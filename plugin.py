"""
Sliding Window Anchor
=====================

Injects an end frame at the beginning of a new sliding window to create smooth
transitions between windows.

When WanGP generates a long video in sliding windows, each window is a separate
generation. This plugin takes the last frame of the window just produced and
feeds it back as a condition on the first frame of the next one, so the new
window starts from where the previous one ended.

For MiniMax H3. Other model families place injected frames using different
arithmetic, so the plugin leaves them alone rather than putting a frame
somewhere it was not meant to go.
"""

import atexit
import json
import os
import traceback

import gradio as gr
from PIL import Image

from shared.utils.plugins import WAN2GPPlugin

from .frame_utils import frame_to_rgb_uint8

# (module path, class name, friendly label). Patched independently; a failure
# on one never blocks the others.
#
# MiniMax H3 only, deliberately. The position handed to the model is not a
# plain frame number: H3 subtracts the carried-over frame count from it before
# using it, so the plugin adds that count back on. Other families do not, and
# would read the same number literally and drop the anchor part-way into the
# window instead of at its start -- with no error to say so. Adding a family
# here means checking how it resolves injection positions first.
PATCH_TARGETS = [
    ("models.minimax_h3.pipeline", "MiniMaxH3Pipeline", "MiniMax H3"),
]

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
INSERT_AFTER_TARGET = "video_info_accordion"  # sibling immediately before generate_btn
TAG = "[SlidingWindowAnchor]"

DESCRIPTION = (
    "Sliding Window Anchor injects an end frame at the beginning of a new "
    "sliding window to create smooth transitions between windows."
)

SAVE_NOTE = (
    "Keeps a copy of every frame this plugin injects, in the saved_frames "
    "folder inside the plugin folder, grouped by run. Frames are otherwise "
    "discarded when a generation finishes."
)

SETTINGS_WARNING = (
    "Keep **Discard Last Frames of a Window** and **Trim First Frames** at 0. "
    "Both trim frames after generation, which would leave the anchor pointing "
    "at a frame that is no longer where the next window continues from."
)


class SlidingWindowAnchorPlugin(WAN2GPPlugin):

    def __init__(self):
        super().__init__()
        self.name = "Sliding Window Anchor"
        self.version = "1.0.0"
        self.description = DESCRIPTION

        self.state_dir = os.path.join(PLUGIN_DIR, "state")
        self.saved_dir = os.path.join(PLUGIN_DIR, "saved_frames")
        os.makedirs(self.state_dir, exist_ok=True)

        # --- settings ---
        self.enabled = True
        self.save_frames = False

        # --- runtime state ---
        # The anchor gets a NEW filename each window. Gradio caches images by
        # path, so reusing one name would leave the panel showing a stale
        # frame even though the file changed underneath it.
        self._anchor_path = None
        self._anchor_seq = 0
        self._window_no = 0          # windows generated so far this run
        self._anchor_frame_no = None  # approximate index in the finished video
        self._output_frames = 0       # running total of retained frames
        self._patched = []
        self._was_in_progress = False
        self._run_stamp = None
        self._warned_missing_kwargs = False

        self._settings_path = os.path.join(self.state_dir, "settings.json")
        self._load_settings()

        # Start every session clean. A frame left on disk belongs to a previous
        # run and injecting it into a new piece would drag in unrelated
        # content. atexit covers clean shutdowns; this covers hard kills.
        self._clear_anchors()
        atexit.register(self._clear_anchors)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def setup_ui(self):
        self.request_component("state")
        self._install_patches()

    def post_ui_setup(self, components):
        # Model modules are imported lazily; retry anything that didn't take
        # at setup_ui time.
        self._install_patches()
        if not self._patched:
            print(f"{TAG} WARNING: MiniMax H3 was not hooked, so the plugin "
                  f"will have no effect. Either this Wan2GP build does not "
                  f"include H3, or it has moved or renamed the pipeline class "
                  f"since this version was written. Generation is "
                  f"unaffected.")

        state_component = components.get("state")

        def build_panel():
            with gr.Accordion(self._panel_title(), open=False) as panel:
                gr.Markdown(DESCRIPTION)

                enabled_cb = gr.Checkbox(value=self.enabled,
                                         label="Enable Sliding Window Anchor")

                with gr.Row():
                    thumb = gr.Image(
                        value=self._anchor_value(), label="Anchor Frame",
                        interactive=False, height=140, width=140,
                    )
                    info_md = gr.Markdown(self._info_text())

                save_cb = gr.Checkbox(
                    value=self.save_frames,
                    label="Save Injected Frames",
                    info=SAVE_NOTE,
                )

                gr.Markdown(SETTINGS_WARNING)

                timer = gr.Timer(3)

                def _toggle(enabled, save_frames):
                    self.enabled = bool(enabled)
                    self.save_frames = bool(save_frames)
                    self._save_settings()
                    return gr.update(label=self._panel_title())

                toggles = [enabled_cb, save_cb]
                for control in toggles:
                    control.change(_toggle, inputs=toggles, outputs=[panel])

                def _poll(state_value):
                    gen = {}
                    if isinstance(state_value, dict):
                        gen = state_value.get("gen", {}) or {}
                    in_progress = bool(gen.get("in_progress"))

                    if self._was_in_progress and not in_progress:
                        # gen["in_progress"] has just been deleted, so the
                        # whole queue is finished, not merely one window.
                        self._reset_run()
                    self._was_in_progress = in_progress

                    return (self._anchor_value(), self._info_text(),
                            gr.update(label=self._panel_title()))

                tick_inputs = [state_component] if state_component is not None else []
                timer.tick(_poll, inputs=tick_inputs,
                           outputs=[thumb, info_md, panel])

            return panel

        try:
            self.insert_after(
                target_component_id=INSERT_AFTER_TARGET,
                new_component_constructor=build_panel,
            )
        except Exception as exc:
            # The panel is only a display; the anchoring itself is already
            # hooked and keeps working without it.
            print(f"{TAG} could not place the panel next to "
                  f"'{INSERT_AFTER_TARGET}' ({exc}). Anchoring still works; "
                  f"watch the console for its output.")
        return {}

    # ------------------------------------------------------------------ #
    # patching
    # ------------------------------------------------------------------ #

    def _install_patches(self):
        import importlib

        for module_path, class_name, label in PATCH_TARGETS:
            if label in self._patched:
                continue
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name, None)
                if cls is None or not hasattr(cls, "generate"):
                    continue
                if getattr(cls.generate, "_sliding_window_anchor_wrapped", False):
                    self._patched.append(label)
                    continue
                cls.generate = self._wrap(cls.generate, label)
                self._patched.append(label)
                print(f"{TAG} hooked {label}")
            except ImportError:
                continue
            except Exception as exc:
                print(f"{TAG} could not hook {label}: {exc}")

    def _wrap(self, original, label):
        plugin = self

        def wrapped(model_self, *args, **kwargs):
            try:
                plugin._inject(kwargs)
            except Exception:
                print(f"{TAG} injection failed, generating without it:")
                traceback.print_exc()

            result = original(model_self, *args, **kwargs)

            try:
                plugin._capture(result, kwargs, label)
            except Exception:
                print(f"{TAG} could not capture the anchor frame:")
                traceback.print_exc()
            return result

        wrapped._sliding_window_anchor_wrapped = True
        wrapped.__name__ = getattr(original, "__name__", "generate")
        wrapped.__doc__ = getattr(original, "__doc__", None)
        return wrapped

    # ------------------------------------------------------------------ #
    # the work
    # ------------------------------------------------------------------ #

    def _inject(self, kwargs):
        """
        Feed the anchor frame in as a condition on the window's first frame.

        Goes through frames_to_inject + frames_relative_positions_list. That
        loop runs for every checkpoint, so this works on FL2VA and Ref2VA
        alike.
        """
        if not self.enabled:
            return
        if not self._anchor_path or not os.path.isfile(self._anchor_path):
            return  # the first window has nothing to anchor to

        if "frames_to_inject" not in kwargs or \
           "frames_relative_positions_list" not in kwargs:
            if not self._warned_missing_kwargs:
                self._warned_missing_kwargs = True
                print(f"{TAG} SKIPPED: this model did not receive "
                      f"frames_to_inject / frames_relative_positions_list, so "
                      f"there is nothing to anchor to. Generation continues "
                      f"unchanged.")
            return

        images = list(kwargs.get("frames_to_inject") or [])
        positions = list(kwargs.get("frames_relative_positions_list") or [])
        if len(images) != len(positions):
            print(f"{TAG} SKIPPED: injected-frame lists are out of step "
                  f"({len(images)} images, {len(positions)} positions); not "
                  f"touching them.")
            return

        # The model subtracts history_count from every position it is given
        # and then bounds-checks the result, so a raw index of 0 would go
        # negative and be silently dropped. Passing history_count instead
        # lands the frame on the window's first generated frame.
        index = self._history_count(kwargs)

        images.append(Image.open(self._anchor_path).convert("RGB"))
        positions.append(index)
        kwargs["frames_to_inject"] = images
        kwargs["frames_relative_positions_list"] = positions
        print(f"{TAG} anchored window {self._window_no + 1} to the end frame "
              f"of window {self._window_no} (raw index {index}).")

    def _capture(self, result, kwargs, label):
        """Save the last frame of the window just generated."""
        if not self.enabled:
            return

        if isinstance(result, dict):
            frames = result.get("x", None)
            if frames is None:
                # Be tolerant of a future pipeline naming the video
                # differently: take the first 4-D, 3-channel tensor present.
                for value in result.values():
                    if hasattr(value, "ndim") and value.ndim == 4 \
                            and getattr(value, "shape", (0,))[0] == 3:
                        frames = value
                        break
        else:
            frames = result
        if frames is None:
            print(f"{TAG} SKIPPED capture: {label} returned no frames "
                  f"(result type {type(result).__name__}).")
            return
        if not hasattr(frames, "ndim") or frames.ndim != 4 or frames.shape[0] != 3:
            shape = tuple(frames.shape) if hasattr(frames, "shape") else "unknown"
            print(f"{TAG} SKIPPED capture: expected a 3-channel (C,T,H,W) "
                  f"tensor, got {shape}.")
            return

        self._window_no += 1

        # WanGP discards the overlap frames from the front of every window
        # after the first, so only the remainder reaches the finished video.
        # Track that to report roughly where the anchor frame ends up.
        overlap = self._overlap(kwargs)
        retained = max(0, int(frames.shape[1]) - overlap)
        self._output_frames += retained
        self._anchor_frame_no = self._output_frames

        rgb = frame_to_rgb_uint8(frames, index=-1)
        self._write_anchor(rgb)
        if self.save_frames:
            self._save_frame(rgb)
        print(f"{TAG} captured the end frame of window {self._window_no} "
              f"(approx. frame {self._anchor_frame_no} of the output).")

    # ------------------------------------------------------------------ #
    # helpers mirroring the pipeline's own arithmetic
    # ------------------------------------------------------------------ #

    def _overlap(self, kwargs):
        """Frames carried over from the previous window, 0 on the first."""
        try:
            value = int(kwargs.get("prefix_frames_count") or 0)
            return max(0, value)
        except Exception:
            return 0

    def _history_count(self, kwargs):
        """
        How many carried-over frames sit in front of the generated content.

        The pipeline normalises the overlap, clamps it to the length of the
        video handed back to it, and keeps all but the last of those frames as
        history; the last one becomes the window's own starting condition.
        """
        try:
            prefix = int(kwargs.get("prefix_frames_count") or 0)
            if prefix <= 0 or kwargs.get("image_start") is not None:
                return 0
            video = kwargs.get("input_video")
            if video is None or not hasattr(video, "shape") or len(video.shape) < 2:
                return 0
            try:
                from shared.utils.frame_scheduler import normalize_overlap
                normalised, error = normalize_overlap(prefix, 17, 1)
                if error is None and normalised:
                    prefix = int(normalised)
            except Exception:
                pass
            return max(0, min(prefix, int(video.shape[1])) - 1)
        except Exception as exc:
            print(f"{TAG} could not read the overlap ({exc}); using 0")
            return 0

    # ------------------------------------------------------------------ #
    # files
    # ------------------------------------------------------------------ #

    def _write_anchor(self, rgb_uint8):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            self._anchor_seq += 1
            path = os.path.join(self.state_dir,
                                f"anchor_{self._anchor_seq:04d}.png")
            tmp = path + ".writing"
            Image.fromarray(rgb_uint8).save(tmp, format="PNG")
            os.replace(tmp, path)
            previous = self._anchor_path
            self._anchor_path = path
            if previous and os.path.isfile(previous):
                try:
                    os.remove(previous)
                except OSError:
                    pass
        except Exception as exc:
            print(f"{TAG} could not write the anchor frame: {exc}")

    def _save_frame(self, rgb_uint8):
        """Keep a copy of an injected frame, grouped by run."""
        try:
            import datetime
            if self._run_stamp is None:
                self._run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(self.saved_dir, f"run_{self._run_stamp}")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"window_{self._window_no:02d}.png")
            Image.fromarray(rgb_uint8).save(path, format="PNG")
            print(f"{TAG} saved {path}")
        except Exception as exc:
            print(f"{TAG} could not save the injected frame: {exc}")

    def _clear_anchors(self):
        try:
            if os.path.isdir(self.state_dir):
                for name in os.listdir(self.state_dir):
                    if name.startswith("anchor_"):
                        try:
                            os.remove(os.path.join(self.state_dir, name))
                        except OSError:
                            pass
        except Exception:
            pass
        self._anchor_path = None

    def _reset_run(self):
        self._clear_anchors()
        self._window_no = 0
        self._anchor_frame_no = None
        self._output_frames = 0
        self._run_stamp = None
        self._warned_missing_kwargs = False

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        try:
            if not os.path.isfile(self._settings_path):
                return
            with open(self._settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if "enabled" in data:
                self.enabled = bool(data["enabled"])
            if "save_frames" in data:
                self.save_frames = bool(data["save_frames"])
        except Exception as exc:
            print(f"{TAG} could not load settings ({exc}); using defaults.")

    def _save_settings(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with open(self._settings_path, "w", encoding="utf-8") as handle:
                json.dump({"enabled": self.enabled,
                           "save_frames": self.save_frames}, handle, indent=2)
        except Exception as exc:
            print(f"{TAG} could not save settings: {exc}")

    # ------------------------------------------------------------------ #
    # panel display
    # ------------------------------------------------------------------ #

    def _panel_title(self):
        return "Sliding Window Anchor" if self.enabled \
            else "Sliding Window Anchor (off)"

    def _anchor_value(self):
        return self._anchor_path if self._anchor_path \
            and os.path.isfile(self._anchor_path) else None

    def _info_text(self):
        window = self._window_no if self._window_no else "N/A"
        frame = self._anchor_frame_no if self._anchor_frame_no else "N/A"
        return f"Sliding Window: {window}  \nFrame (approx): {frame}"
