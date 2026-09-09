"""
MiniMax H3 Sliding Window Anchor
===============================

Injects an end frame at the beginning of a new sliding window to create smooth
transitions between windows.

When WanGP generates a long video in sliding windows, each window is a separate
generation. This plugin takes the last frame of the window just produced and
feeds it back as a condition on the first frame of the next one, so the new
window starts from where the previous one ended.

Everything it carries belongs to one video. Wan2GP runs a queue through the
same process back to back, so the plugin reads the window number out of each
call and drops the anchor and the colour reference whenever the next item, or
the next repeat of the same item, begins.

It can also hold colour, saturation and brightness steady across the join.
The model re-synthesises the anchor frame rather than copying it, and the
small error in that reconstruction is inherited by every window after it, so
a long generation drifts. See colour_match.py.

For MiniMax H3. Other model families place injected frames using different
arithmetic, so the plugin leaves them alone rather than putting a frame
somewhere it was not meant to go.
"""

import atexit
import contextlib
import json
import os
import traceback

import gradio as gr
import torch
from PIL import Image

from shared.utils.plugins import WAN2GPPlugin

from . import colour_match
from .frame_utils import chunk_to_float01, detect_kind, frame_to_rgb_uint8

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

MATCH_DESCRIPTION = (
    "The model re-synthesises the anchor frame rather than copying it, and "
    "the reconstruction is never exact -- blacks lift, contrast softens, "
    "colour desaturates a little. Every window inherits that error from the "
    "one before it, so a long generation drifts. This measures the "
    "difference across each join and corrects it."
)

MATCH_SCOPE_NOTE = (
    "**Whole window** corrects the frames Wan2GP goes on to save, so the "
    "join in the finished video matches. **Anchor frame only** leaves the "
    "video untouched and corrects just the frame fed into the next window, "
    "which stops the drift compounding but does not remove what is already "
    "in the output."
)

MATCH_REFERENCE_NOTE = (
    "**Previous window** matches each window to the one before it. "
    "**First window** matches every window to the opening one, which holds a "
    "consistent look harder but fights any lighting change you actually "
    "wanted."
)

# Gradio radio labels, kept apart from the values stored in settings so the
# wording can change without invalidating a saved settings file.
SCOPE_LABELS = {"Whole window": "window", "Anchor frame only": "anchor"}
REFERENCE_LABELS = {"Previous window": "previous", "First window": "first"}

# Above this share of clipped values a correction is pushing content past the
# ends of the range and flattening it, rather than shifting it.
CLIP_WARN_FRACTION = 0.005

# Consecutive rejections after which a first-window reference is treated as
# describing a scene that is no longer being generated.
STALE_AFTER = 2

# Fewer usable frames than this before a continuation join and the old
# video's tail is not a reference worth matching to.
MIN_CONTINUE_SPAN = 3

# Frames either side of a join that get compared. Not a setting: measured on
# real footage, a real join reads just as strongly at 5 frames as at 18,
# while a quiet join reads more false difference the wider it gets. There is
# no value here worth offering, so there is no control for it.
MATCH_SPAN = 5


class SlidingWindowAnchorPlugin(WAN2GPPlugin):

    def __init__(self):
        super().__init__()
        self.name = "MiniMax H3 Sliding Window Anchor"
        self.version = "1.2.0"
        self.description = DESCRIPTION

        self.state_dir = os.path.join(PLUGIN_DIR, "state")
        self.saved_dir = os.path.join(PLUGIN_DIR, "saved_frames")
        os.makedirs(self.state_dir, exist_ok=True)

        # --- settings ---
        self.enabled = True
        self.save_frames = False

        # Colour matching is off by default. Whole-window scope rewrites the
        # frames Wan2GP saves, and a plugin update should not quietly start
        # altering the output of a setup that was already working.
        self.match_enabled = False
        self.match_scope = "window"
        self.match_reference = "previous"
        self.match_brightness = True
        self.match_contrast = True
        self.match_saturation = True
        self.match_colour = True
        self.match_strength = 1.0
        self.match_limit = 0.06
        self.match_span = MATCH_SPAN
        # The first window of a Continue Video run is matched to the end of
        # the video being continued. That reference comes from a short strip
        # of carried-over frames rather than a whole window, so by default
        # only the anchor is corrected there and the video is left alone.
        self.match_continue_anchor_only = True

        # --- runtime state ---
        # The anchor gets a NEW filename each window. Gradio caches images by
        # path, so reusing one name would leave the panel showing a stale
        # frame even though the file changed underneath it.
        self._anchor_path = None
        self._anchor_seq = 0
        self._window_no = 0          # windows generated so far this run
        self._anchor_frame_no = None  # approximate index in the finished video
        self._output_frames = 0       # running total of retained frames
        # Where the last window sat in its own video, so the next call can be
        # told apart from the first window of the next video in a queue.
        self._last_window_no = None
        self._last_window_start = None
        self._patched = []
        self._was_in_progress = False
        self._run_stamp = None
        self._warned_missing_kwargs = False
        self._warned_dead_scope = False

        # Colour statistics every window is matched onto, and a plain-text
        # note of what the last correction did, for the panel.
        self._reference_stats = None
        self._match_report = None
        self._continuing = False
        self._rejections = 0
        self._effective_scope = self.match_scope

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

                with gr.Accordion("Colour Consistency", open=False):
                    gr.Markdown(MATCH_DESCRIPTION)

                    match_cb = gr.Checkbox(
                        value=self.match_enabled,
                        label="Match Colour Across Windows",
                        info="Works with anchoring switched off as well. "
                             "Nothing here needs an injected frame to exist, "
                             "so the two can be run separately to see what "
                             "each is doing.",
                    )

                    match_md = gr.Markdown(self._match_text())

                    scope_radio = gr.Radio(
                        choices=list(SCOPE_LABELS),
                        value=self._label_for(SCOPE_LABELS, self.match_scope),
                        label="Correction Scope",
                        info=MATCH_SCOPE_NOTE,
                    )
                    reference_radio = gr.Radio(
                        choices=list(REFERENCE_LABELS),
                        value=self._label_for(REFERENCE_LABELS,
                                              self.match_reference),
                        label="Match Against",
                        info=MATCH_REFERENCE_NOTE,
                    )

                    with gr.Row():
                        bright_cb = gr.Checkbox(value=self.match_brightness,
                                                label="Brightness")
                        contrast_cb = gr.Checkbox(value=self.match_contrast,
                                                  label="Contrast")
                        sat_cb = gr.Checkbox(value=self.match_saturation,
                                             label="Saturation")
                        colour_cb = gr.Checkbox(value=self.match_colour,
                                                label="Colour Cast")

                    strength_slider = gr.Slider(
                        0.0, 1.0, value=self.match_strength, step=0.05,
                        label="Correction Strength",
                        info="How much of the measured difference to remove. "
                             "Below 1 leaves some drift in place, which can "
                             "look more natural than a hard match.",
                    )
                    limit_slider = gr.Slider(
                        0.02, 0.5, value=self.match_limit, step=0.01,
                        label="Scene Change Threshold",
                        info="Past this, a difference is read as a cut "
                             "rather than drift and the window is left "
                             "alone. Real drift is around a percent per "
                             "window, so anything much larger is content.",
                    )
                    continue_cb = gr.Checkbox(
                        value=self.match_continue_anchor_only,
                        label="Anchor Only On A Continued Video's First "
                              "Window",
                        info="A Continue Video run matches its first window "
                             "to the end of the video being continued. That "
                             "reference is a short strip of frames rather "
                             "than a whole window, so by default only the "
                             "anchor is corrected there.",
                    )

                gr.Markdown(SETTINGS_WARNING)

                timer = gr.Timer(3)

                def _toggle(enabled, save_frames, match_enabled, scope,
                            reference, brightness, contrast, saturation,
                            colour, strength, limit,
                            continue_anchor_only):
                    self.enabled = bool(enabled)
                    self.save_frames = bool(save_frames)
                    self.match_enabled = bool(match_enabled)
                    self.match_scope = SCOPE_LABELS.get(scope,
                                                        self.match_scope)
                    self.match_reference = REFERENCE_LABELS.get(
                        reference, self.match_reference)
                    self.match_brightness = bool(brightness)
                    self.match_contrast = bool(contrast)
                    self.match_saturation = bool(saturation)
                    self.match_colour = bool(colour)
                    self.match_strength = float(strength)
                    self.match_limit = float(limit)
                    self.match_continue_anchor_only = bool(
                        continue_anchor_only)
                    self._save_settings()
                    return gr.update(label=self._panel_title())

                toggles = [enabled_cb, save_cb, match_cb, scope_radio,
                           reference_radio, bright_cb, contrast_cb, sat_cb,
                           colour_cb, strength_slider, limit_slider,
                           continue_cb]
                for control in toggles:
                    control.change(_toggle, inputs=toggles, outputs=[panel])

                def _poll(state_value):
                    gen = {}
                    if isinstance(state_value, dict):
                        gen = state_value.get("gen", {}) or {}
                    in_progress = bool(gen.get("in_progress"))

                    if self._was_in_progress and not in_progress:
                        # gen["in_progress"] has just been deleted, so the
                        # whole queue is finished. Boundaries between the
                        # videos inside it are found from the generate call
                        # itself; this only clears up after the last one, so
                        # the panel does not sit showing a finished video's
                        # anchor.
                        self._reset_run()
                    self._was_in_progress = in_progress

                    return (self._anchor_value(), self._info_text(),
                            self._match_text(),
                            gr.update(label=self._panel_title()))

                tick_inputs = [state_component] if state_component is not None else []
                timer.tick(_poll, inputs=tick_inputs,
                           outputs=[thumb, info_md, match_md, panel])

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
            # First, because everything below assumes the state it reads
            # belongs to the video being generated now. This call may be the
            # opening window of the next item in a queue rather than the next
            # window of the current video.
            try:
                plugin._note_window(kwargs)
            except Exception:
                print(f"{TAG} could not tell whether this is a new video; "
                      f"carrying on with the state as it stands:")
                traceback.print_exc()

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
    # telling one video from the next
    # ------------------------------------------------------------------ #

    def _starts_new_video(self, kwargs):
        """
        Whether this call opens a new video rather than continuing one.

        Returns (yes, reason), the reason phrased for the log.

        Wan2GP numbers the windows of a video from 1 and starts again at 1
        for the next item in the queue -- and for each repeat of the same
        item -- so the window number answers this outright. The two fallbacks
        below are for a build that does not pass one.
        """
        def whole(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        window = whole(kwargs.get("window_no"))
        if window is not None and window > 0:
            if window == 1:
                return True, "this is window 1"
            previous = self._last_window_no
            if previous is not None and window < previous:
                # Numbering that goes backwards without passing through 1 is
                # not something Wan2GP does, but if it ever did it would mean
                # a restart, and a restart is a new video.
                return True, (f"the window number went from {previous} back "
                              f"to {window}")
            return False, ""

        # No window number, so fall back on the frame the window begins at.
        # That counts up through a video and starts over with the next one,
        # so a number that has not moved forward is a new video. Equality is
        # deliberately not counted: a window that begins where the last one
        # did is more likely a second call for the same window than a new
        # video, and a real new video starts at 0 or a restart.
        start = whole(kwargs.get("window_start_frame_no"))
        if start is not None:
            previous = self._last_window_start
            if start <= 0:
                return True, "the window begins at frame 0"
            if previous is not None and start < previous:
                return True, (f"the window start went from frame {previous} "
                              f"back to {start}")
            return False, ""

        # Neither is available. All that is left is the carried-over frames:
        # a window given none has nothing in front of it to continue from.
        # This misreads the first window of a Continue Video run, which is
        # handed the tail of the video it continues, so on a build this far
        # from the one this was written against a queued continuation may
        # still open on the previous video's anchor.
        if self._overlap(kwargs) <= 0:
            return True, "no frames were carried into this window"
        return False, ""

    def _note_window(self, kwargs):
        """
        Drop the previous video's state when a new video starts.

        Everything carried from one call to the next -- the anchor frame, the
        window count, the colour reference -- describes a single video. A
        queue runs several through the same process back to back, and the
        only thing that used to clear that state was the panel noticing the
        queue as a whole had finished. So the second video in a queue opened
        by injecting the last frame of the first and matching its colour to
        it: one video's ending bled into the next one's beginning.

        Checked here, from the call itself, rather than in the panel. The
        panel polls on a timer, is not there at all if it failed to place,
        and cannot see a boundary the queue does not stop at.
        """
        new_video, reason = self._starts_new_video(kwargs)

        carried = (self._window_no or self._anchor_path
                   or self._reference_stats is not None)
        if new_video and carried:
            print(f"{TAG} a new video has started ({reason}); clearing the "
                  f"anchor frame, window count and colour reference left "
                  f"over from the previous one.")
            self._reset_run()

        self._remember_position(kwargs)

    def _remember_position(self, kwargs):
        """Keep this window's position for the next call to compare against."""
        for key, attribute in (("window_no", "_last_window_no"),
                               ("window_start_frame_no", "_last_window_start")):
            try:
                setattr(self, attribute, int(kwargs[key]))
            except (KeyError, TypeError, ValueError):
                setattr(self, attribute, None)

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
            # No anchor yet. On a continuation the frames handed in carry the
            # end of the video being continued, which is exactly what the
            # next window should start from.
            if not self._seed_anchor_from_input(kwargs):
                return  # a fresh run has nothing to anchor to

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
        source = (f"the end frame of window {self._window_no}"
                  if self._window_no else "the last frame of the video it is "
                  "continuing")
        print(f"{TAG} anchored window {self._window_no + 1} to {source} "
              f"(raw index {index}).")

    def _seed_anchor_from_input(self, kwargs):
        """
        Make the first anchor of a continuation from the video handed in.

        Every other anchor is the last frame of a window this plugin watched
        being generated. The first window of a Continue Video run has no such
        predecessor, so until now it was the one window that started with
        nothing to anchor to -- the join between the old video and the new
        one, which is the most visible join there is.

        The frames carried in are the tail of the video being continued, so
        its last frame is the frame the new window should start from.

        Returns True when an anchor was written.
        """
        if self._window_no > 0:
            return False  # mid-run; a missing anchor here is a real problem
        if kwargs.get("image_start") is not None:
            return False  # starting from a still, not continuing a video

        video = kwargs.get("input_video")
        if video is None or not hasattr(video, "ndim"):
            return False
        if video.ndim != 4 or video.shape[0] != 3 or int(video.shape[1]) < 1:
            shape = tuple(video.shape) if hasattr(video, "shape") else "unknown"
            print(f"{TAG} the video being continued is not a 3-channel "
                  f"(C,T,H,W) tensor ({shape}), so the first window cannot "
                  f"be anchored to it.")
            return False

        try:
            rgb = frame_to_rgb_uint8(video, index=-1)
        except Exception as exc:
            print(f"{TAG} could not read the last frame of the video being "
                  f"continued ({exc}); the first window is unanchored.")
            return False

        self._write_anchor(rgb)
        if self.save_frames:
            self._save_frame(rgb)
        self._continuing = True
        print(f"{TAG} continuing an existing video: anchoring the first "
              f"window to its last frame.")
        return True

    def _capture(self, result, kwargs, label):
        """
        Save the last frame of the window just generated, and correct its
        colour if that is switched on.

        The two are independent. Colour matching measures the difference
        across a join and corrects the window; nothing about that needs an
        injected frame to exist. So this runs when either is on, and only
        the parts each one needs are done.
        """
        if not self.enabled and not self.match_enabled:
            return

        # The key is kept as well as the tensor. A corrected window cannot
        # always be written where it stands, and putting the replacement back
        # under the right name is the only way the pipeline ever sees it.
        frames_key = None
        if isinstance(result, dict):
            frames = result.get("x", None)
            if frames is not None:
                frames_key = "x"
            else:
                # Be tolerant of a future pipeline naming the video
                # differently: take the first 4-D, 3-channel tensor present.
                for key, value in result.items():
                    if hasattr(value, "ndim") and value.ndim == 4 \
                            and getattr(value, "shape", (0,))[0] == 3:
                        frames = value
                        frames_key = key
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

        kind = detect_kind(frames)

        correction = None
        if self.match_enabled:
            try:
                correction, replacement = self._colour_match(frames, kind,
                                                             overlap)
                if replacement is not None:
                    frames = replacement
                    if frames_key is not None:
                        result[frames_key] = replacement
                    else:
                        print(f"{TAG} the window was corrected but this "
                              f"pipeline returns a bare tensor, so the "
                              f"correction cannot be handed back. Only the "
                              f"anchor carries it.")
            except Exception:
                print(f"{TAG} colour matching failed, leaving the window "
                      f"as generated:")
                traceback.print_exc()

        if not self.enabled:
            # Colour matching only. Nothing is anchored, so there is no frame
            # to write and nothing for the next window to be injected with.
            print(f"{TAG} colour matched window {self._window_no} "
                  f"(anchoring is off, so no frame was captured).")
            return

        rgb = frame_to_rgb_uint8(frames, index=-1, kind=kind)

        # In whole-window scope the frame has already been corrected along
        # with the rest of the window. In anchor scope the window was left
        # alone, so the anchor is corrected on its own here.
        if correction is not None and self._effective_scope == "anchor":
            rgb = colour_match.apply_to_frame(rgb, correction)

        self._write_anchor(rgb)
        if self.save_frames:
            self._save_frame(rgb)
        print(f"{TAG} captured the end frame of window {self._window_no} "
              f"(approx. frame {self._anchor_frame_no} of the output).")

    # ------------------------------------------------------------------ #
    # colour, saturation and brightness matching
    # ------------------------------------------------------------------ #

    def _colour_match(self, frames, kind, overlap):
        """
        Hold colour steady across the join, and return the correction used.

        The drift is measured between the last few frames of the previous
        window and the first few *generated* frames of this one. Those cover
        the same moment, so what differs between them is the reconstruction
        error rather than the scene moving on. Comparing whole windows
        instead would read a camera panning from a bright room to a dark one
        as drift and flatten it out.

        The carried-over frames at the front are skipped throughout. They
        were copied from the previous window rather than generated, so they
        carry no drift to measure and correcting them again would move
        frames that are already in the right place.

        Measuring from the overlap count is the cautious reading of where
        those frames end, since it is the same count WanGP itself discards
        from the front. If the generated content actually starts a frame
        earlier, this skips one generated frame, which costs a little
        averaging. Starting a frame earlier and being wrong would fold a
        copied frame into the measurement and understate the drift, which is
        the mistake worth avoiding.
        """
        original = frames
        span = max(1, int(self.match_span))
        total = int(frames.shape[1])
        start = max(0, min(overlap, total - 1))
        self._effective_scope = self.match_scope
        if not self.enabled and self._effective_scope == "anchor":
            # The correction would be applied to a frame that is never
            # written and never injected, so it would do nothing at all.
            if not self._warned_dead_scope:
                self._warned_dead_scope = True
                print(f"{TAG} Correction Scope is 'Anchor frame only' but "
                      f"anchoring is switched off, so a correction would "
                      f"have nowhere to go. Correcting the whole window "
                      f"instead for this run.")
            self._effective_scope = "window"

        # Found up front: it bounds how far the reference span may be
        # widened, and lets pairs of frames straddling it be left out of the
        # noise floor.
        cut = colour_match.find_cut(frames, kind, start)

        # A Continue Video run hands its first window a strip of frames from
        # the end of the video being continued. Those are a real reference,
        # and the join between old and new video is the most visible one
        # there is, so without this it would be the only join never checked.
        if self._reference_stats is None and start >= 1:
            # Widened symmetrically: more frames make a steadier reference,
            # but both sides have to keep covering the same moment, so the
            # span grows equally either way and stops at a cut on either
            # side. Lopsided would compare a swathe of old footage against a
            # sliver of new, and read every camera move in between as drift.
            # The same span as any other join. Measured, widening it buys
            # nothing: the signal at a real join is just as strong at 5
            # frames as at 18, while the false reading at a quiet join grows.
            # Worse, a large span widens the noise floor's own neighbourhood,
            # which is what made it blind to real steps in the first place.
            # A continuation is bounded further by what was carried in.
            want = span
            span = min(want, start, total - start)

            before_cut = colour_match.find_cut(frames, kind, 0, end=start,
                                               last=True)
            if before_cut is not None:
                span = min(span, start - before_cut)
            if cut is not None:
                span = min(span, cut - start)
            if span < MIN_CONTINUE_SPAN:
                # A cut this close to the join means the old video's tail is
                # already a different scene, so it is not a reference for the
                # new window at all. Better to leave the window uncorrected,
                # as it was before continuations were handled, than to match
                # it to a frame or two of unrelated content.
                print(f"{TAG} continuing from an existing video, but only "
                      f"{span} usable frame(s) before the join"
                      + (f" (cut at frame {before_cut})"
                         if before_cut is not None else "")
                      + f"; leaving the first window uncorrected.")
                self._continuing = True
                if self.match_continue_anchor_only:
                    self._effective_scope = "anchor"
                return None, None

            prefix_from = start - span
            self._reference_stats = colour_match.measure(
                chunk_to_float01(frames[:, prefix_from:start], kind))
            self._continuing = True
            if span >= start:
                bounded = "the whole carried-in strip"
            elif span >= total - start:
                bounded = "the length of this window"
            elif before_cut is not None and span == start - before_cut:
                bounded = f"a cut at frame {before_cut} in the old video"
            elif cut is not None and span == cut - start:
                bounded = f"a cut at frame {cut} in the new window"
            else:
                bounded = f"the {want}-frame span"
            print(f"{TAG} continuing from an existing video: {start} frames "
                  f"carried in, measuring {span} either side of the join "
                  f"(limited by {bounded}).")
            if self.match_continue_anchor_only:
                # The reference here is a short strip of carried-over frames
                # rather than a whole window, so the default is to correct
                # only the frame handed to the next window and leave the
                # video itself as generated.
                self._effective_scope = "anchor"
                print(f"{TAG} first window of a continuation: correcting the "
                      f"anchor only, leaving the video as generated.")

        head_end = min(total, start + span)
        head = colour_match.measure(
            chunk_to_float01(frames[:, start:head_end], kind))

        # What the same statistics do inside this window, where there is no
        # join and so no drift. Anything at the join not clearly bigger than
        # this is wobble rather than signal.
        noise = colour_match.estimate_noise(frames, kind, start, span,
                                            cut=None if cut is None
                                            else cut - 1)

        correction = None
        if self._reference_stats is not None:
            correction = colour_match.solve(
                head, self._reference_stats, noise=noise,
                brightness=self.match_brightness,
                contrast=self.match_contrast,
                saturation=self.match_saturation,
                colour=self.match_colour,
                strength=self.match_strength,
                limit=self.match_limit,
            )
            if getattr(correction, "rejected", False):
                # Too large to be drift, so the frames either side of the
                # join are not showing the same moment. Correcting on that
                # measurement would drag a new scene toward the grade of the
                # old one, which is worse than leaving it alone.
                print(f"{TAG} window {self._window_no}: difference across "
                      f"the join is too large to be drift, so it is being "
                      f"read as a scene change and left uncorrected.")
                self._match_report = "skipped, looks like a scene change"
                correction = None
                self._rejections += 1
                if (self.match_reference == "first"
                        and self._rejections >= STALE_AFTER):
                    # Locked to the opening window, a cut leaves the
                    # reference describing a scene that no longer exists, and
                    # every window after it is rejected forever. Re-anchoring
                    # keeps the mode usable on material that cuts.
                    print(f"{TAG} the first-window reference no longer "
                          f"matches anything being generated; re-anchoring "
                          f"it to the current scene.")
                    self._reference_stats = None
                    self._rejections = 0
            elif correction.is_identity():
                self._rejections = 0
                correction = None
            else:
                self._rejections = 0

        if noise is not None:
            print(f"{TAG} window {self._window_no} noise floor: "
                  f"{noise.describe()}")

        if correction is not None:
            self._match_report = correction.describe()
            print(f"{TAG} colour match window {self._window_no}: "
                  f"{self._match_report}")
            stop = None
            if cut is not None and cut > start:
                stop = cut
                print(f"{TAG} the picture cuts at frame {cut} of window "
                      f"{self._window_no}; correcting up to there and "
                      f"leaving the new scene alone, since it has nothing "
                      f"to be matched to yet.")

            if self._effective_scope == "window":
                clipped, replacement = self._apply_to_window(
                    frames, kind, correction, start, stop)
                if replacement is not None:
                    # Everything after this point, the tail statistics and
                    # the anchor included, has to read the corrected frames.
                    frames = replacement
                if clipped > CLIP_WARN_FRACTION:
                    print(f"{TAG} {clipped * 100:.1f}% of values were clipped "
                          f"bringing window {self._window_no} into line. That "
                          f"flattens highlights or shadows rather than "
                          f"shifting them; consider lowering Correction "
                          f"Strength or Maximum Correction.")
        else:
            self._match_report = None

        # The tail becomes what the next window is measured against, so it
        # has to describe the window as it now stands. In whole-window scope
        # the frames were corrected in place and measuring them again reads
        # the corrected values. In anchor scope they were not, so the
        # correction is put through the statistics arithmetically instead.
        tail_start = max(start, total - span)
        tail = colour_match.measure(
            chunk_to_float01(frames[:, tail_start:total], kind))
        if correction is not None and self._effective_scope == "anchor":
            tail = colour_match.transform_stats(tail, correction)

        if self._reference_stats is None or self.match_reference == "previous":
            self._reference_stats = tail

        return correction, (frames if frames is not original else None)

    def _apply_to_window(self, frames, kind, correction, start, stop=None):
        """
        Write the correction into the window Wan2GP will save.

        Returns (clipped fraction, replacement), where a replacement is a new
        tensor the caller has to put back into the pipeline's result, and
        None means the window was corrected where it stood.

        H3 generates under torch.inference_mode, and PyTorch refuses in-place
        writes to the tensors that come out of it -- but only from outside
        inference mode. Re-entering it makes the write legal, which is worth
        doing for two reasons beyond the obvious one of not allocating a
        second copy of the window. Correcting the tensor in place is seen by
        whatever else already holds a reference to it, so the correction
        reaches the saved video whether or not the pipeline re-reads the
        result dictionary. And it does not depend on knowing which key the
        video was returned under.

        The copy remains as a fallback for a tensor that refuses writes for
        some other reason, such as sharing storage with something already
        handed on. Which path to take is settled by a probe write before any
        real work, not by catching a failure part way through: a write that
        died on the fourth chunk would leave the window corrected at one end
        and not the other, and the fallback copy would then correct those
        chunks a second time.
        """
        context = (torch.inference_mode() if torch.is_inference(frames)
                   else contextlib.nullcontext())
        with context:
            try:
                # A self-assignment changes nothing but proves the tensor
                # will accept being written to.
                frames[:, start:start + 1] = frames[:, start:start + 1]
                writable = True
            except RuntimeError as exc:
                print(f"{TAG} window {self._window_no} will not accept an "
                      f"in-place correction ({exc}); using a copy instead.")
                writable = False

            if writable:
                return colour_match.apply_to_window(
                    frames, kind, correction, start=start, end=stop), None

        replacement = torch.empty_like(frames)
        clipped = colour_match.apply_to_window(
            frames, kind, correction, start=start, into=replacement,
            end=stop)
        return clipped, replacement

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
        self._last_window_no = None
        self._last_window_start = None
        self._run_stamp = None
        self._warned_missing_kwargs = False
        self._warned_dead_scope = False
        # Statistics belong to the piece they were measured on. Carrying them
        # into a new one would drag an unrelated look across.
        self._reference_stats = None
        self._match_report = None
        self._continuing = False
        self._rejections = 0
        self._effective_scope = self.match_scope

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    # Each setting is read only if present, so a settings file written by an
    # earlier version keeps working and simply takes the defaults for
    # anything it does not mention.
    BOOL_SETTINGS = ("enabled", "save_frames", "match_enabled",
                     "match_brightness", "match_contrast", "match_saturation",
                     "match_colour", "match_continue_anchor_only")
    FLOAT_SETTINGS = ("match_strength", "match_limit")
    INT_SETTINGS = ()
    CHOICE_SETTINGS = {"match_scope": ("window", "anchor"),
                       "match_reference": ("previous", "first")}

    def _load_settings(self):
        try:
            if not os.path.isfile(self._settings_path):
                return
            with open(self._settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key in self.BOOL_SETTINGS:
                if key in data:
                    setattr(self, key, bool(data[key]))
            for key in self.FLOAT_SETTINGS:
                if key in data:
                    setattr(self, key, float(data[key]))
            for key in self.INT_SETTINGS:
                if key in data:
                    setattr(self, key, int(data[key]))
            for key, allowed in self.CHOICE_SETTINGS.items():
                # An unrecognised value would otherwise sit in the settings
                # and quietly disable the branch that checks for it.
                if data.get(key) in allowed:
                    setattr(self, key, data[key])
        except Exception as exc:
            print(f"{TAG} could not load settings ({exc}); using defaults.")

    def _save_settings(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            keys = (self.BOOL_SETTINGS + self.FLOAT_SETTINGS
                    + self.INT_SETTINGS + tuple(self.CHOICE_SETTINGS))
            data = {key: getattr(self, key) for key in keys}
            with open(self._settings_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:
            print(f"{TAG} could not save settings: {exc}")

    # ------------------------------------------------------------------ #
    # panel display
    # ------------------------------------------------------------------ #

    def _panel_title(self):
        return "H3 Sliding Window Anchor" if self.enabled \
            else "H3 Sliding Window Anchor (off)"

    def _anchor_value(self):
        return self._anchor_path if self._anchor_path \
            and os.path.isfile(self._anchor_path) else None

    def _info_text(self):
        window = self._window_no if self._window_no else "N/A"
        frame = self._anchor_frame_no if self._anchor_frame_no else "N/A"
        return f"Sliding Window: {window}  \nFrame (approx): {frame}"

    def _match_text(self):
        if not self.match_enabled:
            return "Last correction: off"
        if self._match_report is None:
            # Either nothing has been generated yet, or a window came back
            # already matching the one before it, which is the good outcome
            # rather than a failure.
            return "Last correction: none needed yet"
        return f"Last correction: {self._match_report}"

    @staticmethod
    def _label_for(mapping, value):
        """The radio label standing for a stored setting value."""
        for label, stored in mapping.items():
            if stored == value:
                return label
        return next(iter(mapping))
