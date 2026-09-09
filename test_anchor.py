"""
Tests for Sliding Window Anchor.

plugin.py imports gradio and Wan2GP's plugin base, neither of which is
importable outside Wan2GP, so the methods under test are lifted out of the
source with ast and run in isolation. That tests the shipped code rather than
a copy of it that could drift.

Run from inside the plugin folder:

    python3 test_anchor.py
"""
import ast

import numpy as np
import torch

FAILED = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))
    if not condition:
        FAILED.append(name)


def constants_from(tree):
    """
    Literal assignments in plugin.py, at module level and inside the class.

    Lifted methods reach for both -- module constants like TAG, and the class
    attributes listing which settings are booleans -- so they are read out of
    the source rather than restated here, where they could drift.
    """
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def lift(*names, **extra):
    """
    Pull named methods out of plugin.py without importing it.

    `extra` supplies anything a method reaches for that is not a literal --
    the colour_match module, say -- since plugin.py cannot be imported here
    to provide its own.
    """
    source = open("plugin.py").read()
    tree = ast.parse(source)
    constants = constants_from(tree)
    constants.update(extra)

    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            namespace = dict(constants)
            # Decorators cannot be resolved here and none of the lifted
            # methods need theirs applied, so they are dropped.
            plain = ast.FunctionDef(
                name=node.name, args=node.args, body=node.body,
                decorator_list=[], returns=node.returns,
                type_comment=node.type_comment, type_params=[],
                lineno=node.lineno, col_offset=node.col_offset,
            )
            exec(compile(ast.Module(body=[plain], type_ignores=[]),
                         "<extracted>", "exec"), namespace)
            found[node.name] = namespace[node.name]
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"could not lift {missing} from plugin.py")
    return found


def load_sibling(name):
    """
    Import a module from the plugin folder without importing the package.

    plugin.py's relative imports need the package, and the package cannot be
    imported without gradio, so the leaf modules are loaded directly and
    registered under the package name their own relative imports expect.
    """
    import importlib
    import os
    import sys
    import types

    if "swa_under_test" not in sys.modules:
        package = types.ModuleType("swa_under_test")
        package.__path__ = [os.path.dirname(os.path.abspath("plugin.py"))]
        sys.modules["swa_under_test"] = package
    return importlib.import_module(f"swa_under_test.{name}")


def install_frame_scheduler_stub():
    """
    Wan2GP is not importable here, so stand in for the one function the
    plugin borrows from it, matching Wan2GP's own rounding. This checks that
    the plugin calls it and uses the result; it cannot check Wan2GP itself.
    """
    import sys
    import types

    def normalize_overlap(frame_count, step, offset=1):
        if frame_count < 0:
            return None, "negative"
        if frame_count == 0:
            return 0, None
        step = max(1, step)
        offset = max(0, offset)
        value = ((frame_count - offset + step // 2) // step) * step + offset
        return max(step if offset == 0 else offset, value), None

    package = types.ModuleType("shared")
    utils = types.ModuleType("shared.utils")
    scheduler = types.ModuleType("shared.utils.frame_scheduler")
    scheduler.normalize_overlap = normalize_overlap
    utils.frame_scheduler = scheduler
    package.utils = utils
    sys.modules.setdefault("shared", package)
    sys.modules.setdefault("shared.utils", utils)
    sys.modules["shared.utils.frame_scheduler"] = scheduler


install_frame_scheduler_stub()

frame_utils = load_sibling("frame_utils")
colour_match = load_sibling("colour_match")
to_float01 = frame_utils.to_float01
frame_to_rgb_uint8 = frame_utils.frame_to_rgb_uint8


class FakeVideo:
    def __init__(self, frames):
        self.shape = (3, frames, 64, 64)


lifted = lift("_history_count", "_overlap", "_info_text")
history_count = lifted["_history_count"]
overlap = lifted["_overlap"]
SELF = None  # neither helper touches self


# --- injection index --------------------------------------------------------
# The model subtracts history_count from the position it is given, so passing
# history_count lands the frame on the window's first generated frame. Passing
# 0 would go negative and be dropped without an error.

check("overlap 18 gives an injection index of 17",
      history_count(SELF, {"prefix_frames_count": 18,
                           "input_video": FakeVideo(200)}) == 17)
check("overlap 1 gives an injection index of 0",
      history_count(SELF, {"prefix_frames_count": 1,
                           "input_video": FakeVideo(200)}) == 0)
check("an unnormalised overlap is rounded to a legal one",
      history_count(SELF, {"prefix_frames_count": 20,
                           "input_video": FakeVideo(200)}) == 17,
      "20 rounds to 18")
check("no overlap gives an index of 0",
      history_count(SELF, {"prefix_frames_count": 0,
                           "input_video": FakeVideo(200)}) == 0)
check("a start image suppresses the carried-over frames",
      history_count(SELF, {"prefix_frames_count": 18,
                           "input_video": FakeVideo(200),
                           "image_start": object()}) == 0)
check("a short carried-over video clamps the index",
      history_count(SELF, {"prefix_frames_count": 18,
                           "input_video": FakeVideo(5)}) == 4)
check("a missing video gives an index of 0",
      history_count(SELF, {"prefix_frames_count": 18,
                           "input_video": None}) == 0)
check("malformed input does not raise",
      history_count(SELF, {"prefix_frames_count": "nonsense"}) == 0)
check("an index is still produced without the scheduler available",
      history_count(SELF, {"prefix_frames_count": 18,
                           "input_video": FakeVideo(200)}) == 17,
      "the rounding is optional, the index is not")


# --- overlap reading --------------------------------------------------------
check("overlap is read from the window", overlap(SELF, {"prefix_frames_count": 18}) == 18)
check("the first window has no overlap", overlap(SELF, {}) == 0)
check("a negative overlap is clamped",
      overlap(SELF, {"prefix_frames_count": -5}) == 0)
check("a malformed overlap does not raise",
      overlap(SELF, {"prefix_frames_count": None}) == 0)


# --- frame numbering shown in the panel ------------------------------------
class Panel:
    def __init__(self, window, frame):
        self._window_no = window
        self._anchor_frame_no = frame


info = lifted["_info_text"]
check("no anchor yet reads N/A for both",
      info(Panel(0, None)) == "Sliding Window: N/A  \nFrame (approx): N/A")
check("after the first window it reads the window and frame",
      info(Panel(1, 362)) == "Sliding Window: 1  \nFrame (approx): 362")
check("later windows report their own number",
      info(Panel(4, 1400)) == "Sliding Window: 4  \nFrame (approx): 1400")


# --- running frame total ----------------------------------------------------
# Wan2GP drops the overlap frames from the front of every window after the
# first, so only the remainder reaches the finished video.
def output_total(window_lengths, overlaps):
    total = 0
    for length, over in zip(window_lengths, overlaps):
        total += max(0, length - over)
    return total


check("the first window contributes all of its frames",
      output_total([362], [0]) == 362)
check("later windows contribute all but the overlap",
      output_total([362, 362, 362], [0, 18, 18]) == 362 + 344 + 344)
check("a run with no overlap contributes everything",
      output_total([250, 250], [0, 0]) == 500)


# --- frame extraction -------------------------------------------------------
window = (torch.rand(3, 24, 32, 48) * 255).to(torch.uint8)
rgb = frame_to_rgb_uint8(window, index=-1)
check("the anchor is an (H, W, 3) uint8 image",
      rgb.shape == (32, 48, 3) and rgb.dtype == np.uint8, f"shape {rgb.shape}")
check("the anchor is the LAST frame of the window",
      np.array_equal(rgb, window[:, -1].permute(1, 2, 0).numpy()),
      "the frame the next window continues from")


# --- decoded formats Wan2GP can hand back ----------------------------------
signed = torch.rand(3, 8, 16, 16) * 2 - 1
converted = to_float01(signed)
check("signed [-1, 1] converts to [0, 1]",
      float(converted.min()) >= 0.0 and float(converted.max()) <= 1.0)
check("converting does not modify the caller's video",
      float(signed.min()) < -0.05,
      "float32 input is not always copied by .float()")

as_uint8 = (torch.rand(3, 8, 16, 16) * 255).to(torch.uint8)
check("uint8 converts to [0, 1]",
      abs(float(to_float01(as_uint8).max())
          - float(as_uint8.max()) / 255.0) < 1e-5)

as_0_255 = torch.rand(3, 8, 16, 16) * 255
before = as_0_255.clone()
converted = to_float01(as_0_255)
check("float 0..255 converts to [0, 1]", float(converted.max()) <= 1.0)
check("float 0..255 conversion does not modify the caller's video",
      torch.equal(as_0_255, before))

already = torch.rand(3, 8, 16, 16)
check("float already in [0, 1] passes through",
      torch.allclose(to_float01(already), already))


# --- panel title -----------------------------------------------------------
title = lift("_panel_title")["_panel_title"]


class Toggle:
    def __init__(self, enabled):
        self.enabled = enabled


check("the panel names the model family",
      title(Toggle(True)) == "H3 Sliding Window Anchor",
      title(Toggle(True)))
check("the panel shows when it is switched off",
      title(Toggle(False)) == "H3 Sliding Window Anchor (off)")


# --- model scope -----------------------------------------------------------
# The injection index is H3's arithmetic. Adding a family here without
# checking how it resolves positions would place the anchor part-way into the
# window, silently.
source = open("plugin.py").read()
targets = ast.literal_eval(source.split("PATCH_TARGETS = ")[1].split("\n\n")[0].strip())
check("only MiniMax H3 is patched",
      [t[2] for t in targets] == ["MiniMax H3"],
      f"targets: {[t[2] for t in targets]}")
check("the H3 pipeline class is named correctly",
      targets[0][0] == "models.minimax_h3.pipeline"
      and targets[0][1] == "MiniMaxH3Pipeline")


# --- window formats -------------------------------------------------------
# The format is sniffed once and then carried, because correcting a window
# changes the range it would be sniffed from. A window read as signed and
# written back as unit-range would come out half as bright, with no error.
FORMATS = {
    frame_utils.UINT8: (torch.rand(3, 4, 8, 8) * 255).to(torch.uint8),
    frame_utils.SIGNED: torch.rand(3, 4, 8, 8) * 2 - 1,
    frame_utils.SCALED: torch.rand(3, 4, 8, 8) * 255,
    frame_utils.UNIT: torch.rand(3, 4, 8, 8),
}

for expected, sample in FORMATS.items():
    check(f"a {expected} window is recognised",
          frame_utils.detect_kind(sample) == expected,
          frame_utils.detect_kind(sample))

for kind, sample in FORMATS.items():
    original = sample.clone()
    restored = frame_utils.from_float01(
        frame_utils.chunk_to_float01(sample, kind), kind, sample.dtype)
    check(f"a {kind} window survives the round trip unchanged",
          torch.equal(restored, original))
    check(f"reading a {kind} window does not modify the caller's video",
          torch.equal(sample, original))


# --- the colour space ------------------------------------------------------
check("Y'CbCr and its inverse are exact",
      float(np.abs(colour_match.RGB_TO_YCC @ colour_match.YCC_TO_RGB
                   - np.eye(3)).max()) < 1e-12)

for name, rgb in [("black", [0, 0, 0]), ("white", [1, 1, 1]),
                  ("mid grey", [0.5, 0.5, 0.5])]:
    luma, cb, cr = colour_match.RGB_TO_YCC @ np.array(rgb, dtype=np.float64)
    check(f"{name} carries no chroma", abs(cb) < 1e-12 and abs(cr) < 1e-12,
          "a neutral must not pick up a cast")
    check(f"{name} keeps its luma", abs(luma - rgb[0]) < 1e-12)


# --- measuring and undoing a known drift -----------------------------------
# The whole feature rests on this: a window that has drifted by a known
# amount must be brought back to where it started.
torch.manual_seed(0)
CLEAN = (torch.rand(3, 1, 40, 60).repeat(1, 12, 1, 1) * 0.6 + 0.2).clamp(0, 1)


def drift(video, brightness=0.0, contrast=1.0, saturation=1.0,
          cast_cb=0.0, cast_cr=0.0):
    """Stand in for the reconstruction error a VAE round trip introduces."""
    forward = torch.as_tensor(colour_match.RGB_TO_YCC, dtype=torch.float32)
    back = torch.as_tensor(colour_match.YCC_TO_RGB, dtype=torch.float32)
    shape = video.shape
    ycc = forward @ video.reshape(3, -1)
    means = ycc.mean(dim=1, keepdim=True)
    ycc[0] = (ycc[0] - means[0]) * contrast + means[0] + brightness
    ycc[1] = (ycc[1] - means[1]) * saturation + means[1] + cast_cb
    ycc[2] = (ycc[2] - means[2]) * saturation + means[2] + cast_cr
    return (back @ ycc).reshape(shape).clamp(0, 1)


REFERENCE = colour_match.measure(CLEAN)

DRIFTS = {
    "washing out": dict(brightness=0.03, contrast=0.95, saturation=0.93),
    "a warm cast": dict(cast_cr=0.02, cast_cb=-0.015),
    "going dark and contrasty": dict(brightness=-0.04, contrast=1.08),
    "losing saturation": dict(saturation=0.80),
}

for name, amount in DRIFTS.items():
    drifted = drift(CLEAN.clone(), **amount)
    before = float((drifted - CLEAN).abs().mean())
    correction = colour_match.solve(colour_match.measure(drifted), REFERENCE,
                                    limit=0.5)
    colour_match.apply_to_window(drifted, frame_utils.UNIT, correction)
    after = float((drifted - CLEAN).abs().mean())
    check(f"{name} is corrected back out",
          after < before * 0.05,
          f"error {before:.5f} -> {after:.5f}")


# --- the correction does nothing when nothing has drifted ------------------
steady = colour_match.measure(CLEAN)
check("matching a window to itself is a no-op",
      colour_match.solve(steady, steady).is_identity())

untouched = (np.random.rand(16, 24, 3) * 255).astype(np.uint8)
check("a no-op correction leaves every pixel exactly as it was",
      np.array_equal(
          colour_match.apply_to_frame(untouched,
                                      colour_match.solve(steady, steady)),
          untouched))


# --- each axis moves only what it is meant to ------------------------------
mixed = drift(CLEAN.clone(), brightness=0.04, contrast=0.9, saturation=0.85,
              cast_cb=0.01)
source = colour_match.measure(mixed)

only_brightness = colour_match.solve(source, REFERENCE, contrast=False,
                                     saturation=False, colour=False,
                                     limit=0.5)
check("brightness alone moves the luma mean and nothing else",
      abs(only_brightness.luma_gain - 1.0) < 1e-9
      and abs(only_brightness.chroma_gain - 1.0) < 1e-9
      and abs(only_brightness.report()["brightness"]) > 0.01)

only_saturation = colour_match.solve(source, REFERENCE, brightness=False,
                                     contrast=False, colour=False, limit=0.5)
check("saturation alone scales chroma and nothing else",
      abs(only_saturation.luma_gain - 1.0) < 1e-9
      and only_saturation.chroma_gain > 1.05
      and abs(only_saturation.report()["brightness"]) < 1e-9)

only_cast = colour_match.solve(source, REFERENCE, brightness=False,
                               contrast=False, saturation=False, limit=0.5)
check("colour cast alone shifts chroma without scaling it",
      abs(only_cast.chroma_gain - 1.0) < 1e-9
      and abs(only_cast.report()["cast_cb"]) > 1e-4)

check("saturation uses one scale for both chroma channels",
      True, "a per-channel scale would shift hue while claiming not to")


# --- bounds, strength and guards -------------------------------------------
dark = colour_match.measure(torch.rand(3, 4, 16, 16) * 0.1)
bright = colour_match.measure(torch.rand(3, 4, 16, 16) * 0.9 + 0.1)
# A difference this size is not drift, it is two different scenes. Capping
# it and applying it anyway would drag the new one toward the grade of the
# old one, which is exactly the wash-out seen on a live run.
rejected = colour_match.solve(dark, bright, limit=0.15)
check("a scene-sized difference is rejected, not capped",
      rejected.rejected and rejected.is_identity(),
      "real drift never reaches that far")
check("rejection is reported so it can be logged",
      colour_match.looks_like_content(dark, bright, 0.15))
check("a drift-sized difference is not rejected",
      not colour_match.looks_like_content(
          steady, colour_match.Stats(steady.luma_mean + 0.01,
                                     steady.luma_spread * 1.01,
                                     steady.cb_mean, steady.cr_mean,
                                     steady.chroma_spread * 0.99), 0.06))
check("rejection can be switched off for known-continuous material",
      not colour_match.solve(dark, bright, limit=0.15,
                             reject=False).is_identity())

close = colour_match.Stats(dark.luma_mean + 0.03, dark.luma_spread,
                           dark.cb_mean, dark.cr_mean, dark.chroma_spread)
check("zero strength leaves the window alone",
      colour_match.solve(dark, close, strength=0.0, limit=0.15).is_identity())
half = colour_match.solve(dark, close, strength=0.5, limit=0.15)
check("half strength removes half the difference",
      abs(half.report()["brightness"] - 0.015) < 1e-6,
      f"brightness {half.report()['brightness']:.4f}")

flat = colour_match.measure(torch.full((3, 4, 16, 16), 0.5))
check("a flat frame does not produce a runaway gain",
      colour_match.solve(flat, bright).luma_gain == 1.0,
      "a fade to black has no spread to divide by")
check("a flat frame does not produce a runaway saturation",
      colour_match.solve(flat, bright).chroma_gain == 1.0)


# --- applying to a window --------------------------------------------------
for kind, sample in FORMATS.items():
    work = sample.clone()
    stats = colour_match.measure(frame_utils.chunk_to_float01(work, kind))
    target = colour_match.Stats(stats.luma_mean * 0.95, stats.luma_spread,
                                stats.cb_mean, stats.cr_mean,
                                stats.chroma_spread * 1.05)
    colour_match.apply_to_window(work, kind,
                                 colour_match.solve(stats, target), start=1)
    check(f"correcting a {kind} window keeps its dtype and shape",
          work.dtype == sample.dtype and work.shape == sample.shape)
    check(f"correcting a {kind} window skips the carried-over frames",
          torch.equal(work[:, 0], sample[:, 0]),
          "they were copied from the previous window, not generated")
    check(f"correcting a {kind} window does change the generated frames",
          not torch.equal(work[:, 1:], sample[:, 1:]))

blown = torch.full((3, 4, 8, 8), 0.98)
# reject=False because this deliberately asks for a correction large enough
# to run off the end of the range, which the scene-change test would refuse.
lift_it = colour_match.solve(colour_match.measure(blown),
                             colour_match.Stats(1.5, 0.0, 0.0, 0.0, 0.0),
                             limit=0.5, reject=False)
check("clipping is reported when a correction runs off the end of the range",
      colour_match.apply_to_window(blown.clone(), frame_utils.UNIT,
                                   lift_it) > 0.5,
      "the signal that a correction is flattening rather than shifting")


# --- predicting statistics instead of re-measuring -------------------------
# Anchor-only scope leaves the window alone, so the statistics handed to the
# next window have to be put through the correction arithmetically.
drifted = drift(CLEAN.clone(), brightness=0.03, saturation=0.9)
correction = colour_match.solve(colour_match.measure(drifted), REFERENCE,
                                limit=0.5)
predicted = colour_match.transform_stats(colour_match.measure(drifted),
                                         correction)
applied = drifted.clone()
colour_match.apply_to_window(applied, frame_utils.UNIT, correction)
measured = colour_match.measure(applied)
check("predicted statistics match measuring the corrected window",
      abs(predicted.luma_mean - measured.luma_mean) < 1e-4
      and abs(predicted.chroma_spread - measured.chroma_spread) < 1e-4,
      f"luma {predicted.luma_mean:.5f} vs {measured.luma_mean:.5f}")


# --- the plugin's own matching step ----------------------------------------
match_step = lift("_colour_match", colour_match=colour_match,
                  chunk_to_float01=frame_utils.chunk_to_float01)["_colour_match"]


class FakeRun:
    """A stand-in plugin carrying just what _colour_match reads."""

    def __init__(self, scope="window", reference="previous", span=5,
                 limit=0.15, strength=1.0):
        self.match_scope = scope
        self.match_reference = reference
        self.match_span = span
        self.match_limit = limit
        self.match_strength = strength
        self.match_brightness = True
        self.match_contrast = True
        self.match_saturation = True
        self.match_colour = True
        self._reference_stats = None
        self._match_report = None
        self._window_no = 1
        self.enabled = True
        self._warned_dead_scope = False
        self._continuing = False
        self._rejections = 0
        self._effective_scope = scope
        self.match_continue_anchor_only = True

    def _apply_to_window(self, frames, kind, correction, start, stop=None):
        import contextlib
        context = (torch.inference_mode() if torch.is_inference(frames)
                   else contextlib.nullcontext())
        with context:
            return colour_match.apply_to_window(frames, kind, correction,
                                                start=start, end=stop), None


run = FakeRun()
first_window = CLEAN.clone()
check("the first window is left as generated",
      match_step(run, first_window, frame_utils.UNIT, 0)[0] is None
      and torch.equal(first_window, CLEAN),
      "there is nothing before it to match against")
check("the first window becomes the reference",
      run._reference_stats is not None)

second = drift(CLEAN.clone(), brightness=0.03, saturation=0.9)
before = float((second - CLEAN).abs().mean())
result, _ = match_step(run, second, frame_utils.UNIT, 0)
after = float((second - CLEAN).abs().mean())
check("a drifted second window is pulled back into line",
      result is not None and after < before * 0.35,
      f"error {before:.5f} -> {after:.5f}")
check("the correction is reported for the panel",
      isinstance(run._match_report, str) and "brightness" in run._match_report)

anchor_run = FakeRun(scope="anchor")
match_step(anchor_run, CLEAN.clone(), frame_utils.UNIT, 0)
anchor_window = drift(CLEAN.clone(), brightness=0.03, saturation=0.9)
kept = anchor_window.clone()
anchor_correction, _ = match_step(anchor_run, anchor_window,
                                  frame_utils.UNIT, 0)
check("anchor scope produces a correction without touching the window",
      anchor_correction is not None and torch.equal(anchor_window, kept),
      "the saved video is left exactly as generated")

locked = FakeRun(reference="first")
match_step(locked, CLEAN.clone(), frame_utils.UNIT, 0)
opening = locked._reference_stats
match_step(locked, drift(CLEAN.clone(), brightness=0.03),
           frame_utils.UNIT, 0)
check("matching against the first window never moves the reference",
      locked._reference_stats is opening)

chained = FakeRun(reference="previous")
match_step(chained, CLEAN.clone(), frame_utils.UNIT, 0)
opening = chained._reference_stats
match_step(chained, drift(CLEAN.clone(), brightness=0.03),
           frame_utils.UNIT, 0)
check("matching against the previous window moves the reference on",
      chained._reference_stats is not opening)

skipped = FakeRun()
match_step(skipped, CLEAN.clone(), frame_utils.UNIT, 0)
carried = drift(CLEAN.clone(), brightness=0.03)
head_before = carried[:, :4].clone()
match_step(skipped, carried, frame_utils.UNIT, 4)
check("the carried-over frames at the front are never corrected",
      torch.equal(carried[:, :4], head_before),
      "they came from the previous window already in the right place")


# --- drift stops compounding over a long run -------------------------------
# The point of the feature. Without correction the error is inherited by
# every window and grows; with it, it does not.
def run_windows(count, correcting):
    reference = None
    carried = dict(brightness=0.0, saturation=1.0)
    worst = 0.0
    for _ in range(count):
        window = drift(CLEAN.clone(), **carried)
        window = drift(window, brightness=0.012, saturation=0.965)
        if correcting:
            head = colour_match.measure(window[:, :5])
            if reference is not None:
                fix = colour_match.solve(head, reference)
                if not fix.is_identity():
                    colour_match.apply_to_window(window, frame_utils.UNIT, fix)
            reference = colour_match.measure(window[:, -5:])
        now = colour_match.measure(window)
        carried = dict(
            brightness=now.luma_mean - REFERENCE.luma_mean,
            saturation=now.chroma_spread / max(REFERENCE.chroma_spread, 1e-6),
        )
        worst = max(worst, abs(now.luma_mean - REFERENCE.luma_mean))
    return worst


loose = run_windows(8, correcting=False)
held = run_windows(8, correcting=True)
check("drift compounds across windows when nothing corrects it",
      loose > 0.08, f"luma drifted {loose * 100:.1f}% by window 8")
check("matching stops it compounding",
      held < loose * 0.35, f"{held * 100:.1f}% instead of {loose * 100:.1f}%")


# --- settings --------------------------------------------------------------
import json as _json
import os as _os

settings = lift("_load_settings", os=_os, json=_json)["_load_settings"]
attrs = constants_from(ast.parse(open("plugin.py").read()))


class FakeSettings:
    BOOL_SETTINGS = attrs["BOOL_SETTINGS"]
    FLOAT_SETTINGS = attrs["FLOAT_SETTINGS"]
    INT_SETTINGS = attrs["INT_SETTINGS"]
    CHOICE_SETTINGS = attrs["CHOICE_SETTINGS"]

    def __init__(self, path):
        self._settings_path = path
        self.enabled = True
        self.save_frames = False
        self.match_enabled = False
        self.match_scope = "window"
        self.match_reference = "previous"
        self.match_brightness = True
        self.match_contrast = True
        self.match_saturation = True
        self.match_colour = True
        self.match_strength = 1.0
        self.match_limit = 0.15
        self.match_span = 5
        self.match_continue_anchor_only = True


def with_settings(data):
    import json
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    obj = FakeSettings(handle.name)
    settings(obj)
    return obj


# A settings file written by 1.0.0 knows nothing about colour matching, and
# has to keep working rather than wiping the new defaults.
old = with_settings({"enabled": False, "save_frames": True})
check("a settings file from before this feature still loads",
      old.enabled is False and old.save_frames is True)
check("settings it does not mention keep their defaults",
      old.match_enabled is False and old.match_scope == "window"
      and old.match_span == 5)

new = with_settings({"match_enabled": True, "match_scope": "anchor",
                     "match_reference": "first", "match_strength": 0.5,
                     "match_limit": 0.3, "match_saturation": False,
                     "match_continue_anchor_only": False})
check("the colour matching settings are read back",
      new.match_enabled is True and new.match_scope == "anchor"
      and new.match_reference == "first" and new.match_strength == 0.5
      and new.match_saturation is False
      and new.match_continue_anchor_only is False)

# The span is fixed in code, so a stale value left in an old settings file
# must not resurrect it as a setting.
stale = with_settings({"match_span": 9})
check("a span left in an old settings file is ignored",
      stale.match_span == 5,
      "it is no longer a setting, and the measured value is the only one")

# An unrecognised choice would otherwise sit in the settings and quietly
# disable the branch that checks for it.
junk = with_settings({"match_scope": "sideways", "match_reference": ""})
check("an unrecognised choice falls back to the default",
      junk.match_scope == "window" and junk.match_reference == "previous")

broken = FakeSettings("/nonexistent/settings.json")
settings(broken)
check("a missing settings file does not raise", broken.match_span == 5)


# --- panel readout ---------------------------------------------------------
readout = lift("_match_text", "_label_for")
match_text = readout["_match_text"]
label_for = readout["_label_for"]


class Readout:
    def __init__(self, enabled, report):
        self.match_enabled = enabled
        self._match_report = report


check("the readout says so when matching is off",
      match_text(Readout(False, None)) == "Last correction: off")
check("the readout waits for the first correction",
      match_text(Readout(True, None)) == "Last correction: none needed yet")
check("the readout shows what the last correction did",
      match_text(Readout(True, "brightness +1.20%"))
      == "Last correction: brightness +1.20%")

scope_labels = attrs["SCOPE_LABELS"]
check("a stored scope maps back to its radio label",
      scope_labels[label_for(scope_labels, "anchor")] == "anchor")
check("an unknown stored value falls back to the first label",
      label_for(scope_labels, "nonsense") == next(iter(scope_labels)))


# --- inference tensors -----------------------------------------------------
# H3 generates under torch.inference_mode. PyTorch refuses in-place writes to
# what comes out of it, but only from outside inference mode, so re-entering
# it corrects the window where it stands instead of allocating a copy.
with torch.inference_mode():
    inference_window = torch.rand(3, 20, 16, 24)

check("H3-style output is an inference tensor",
      torch.is_inference(inference_window))

refused = False
try:
    inference_window[:, 0] = 0.5
except RuntimeError:
    refused = True
check("writing to it from outside inference mode is refused", refused,
      "the failure reported from the live run")

inference_stats = colour_match.measure(inference_window)
inference_fix = colour_match.solve(
    inference_stats,
    colour_match.Stats(inference_stats.luma_mean * 0.97,
                       inference_stats.luma_spread, inference_stats.cb_mean,
                       inference_stats.cr_mean,
                       inference_stats.chroma_spread * 1.05))
kept_prefix = inference_window[:, :4].clone()
kept_body = inference_window[:, 4:].clone()
with torch.inference_mode():
    colour_match.apply_to_window(inference_window, frame_utils.UNIT,
                                 inference_fix, start=4)
check("re-entering inference mode allows the correction in place",
      not torch.equal(inference_window[:, 4:], kept_body),
      "no second copy of the window is needed")
check("correcting in place still skips the carried-over frames",
      torch.equal(inference_window[:, :4], kept_prefix))
check("the corrected tensor is the same object the pipeline holds",
      torch.is_inference(inference_window),
      "so the correction reaches the saved video without a rebind")

probe_ok = True
try:
    with torch.inference_mode():
        inference_window[:, 0:1] = inference_window[:, 0:1]
except RuntimeError:
    probe_ok = False
check("the writability probe succeeds where the real write would",
      probe_ok, "so the copy fallback is only taken when it is needed")


# --- finding a cut inside a window -----------------------------------------
# A correction is worked out from the frames at the start of a window, so it
# describes only the scene those frames belong to. If the picture cuts part
# way through, everything after has no valid reference and is left alone.
torch.manual_seed(11)
scene_a = (torch.rand(3, 1, 24, 32) * 0.4 + 0.10).repeat(1, 30, 1, 1)
scene_b = (torch.rand(3, 1, 24, 32) * 0.5 + 0.45).repeat(1, 30, 1, 1)
scene_a += torch.randn(3, 30, 24, 32) * 0.01
scene_b += torch.randn(3, 30, 24, 32) * 0.01
spliced = torch.cat([scene_a, scene_b], dim=1).clamp(0, 1)

found = colour_match.find_cut(spliced, frame_utils.UNIT, 0)
check("a cut inside a window is found", found == 30, f"found {found}")
check("continuous footage reports no cut",
      colour_match.find_cut(scene_a.clamp(0, 1), frame_utils.UNIT, 0) is None,
      "ordinary motion must not read as a cut")

moving = (torch.rand(3, 1, 24, 32) * 0.5 + 0.2).repeat(1, 40, 1, 1)
moving += torch.cumsum(torch.randn(3, 40, 24, 32) * 0.03, dim=1)
check("heavy motion alone does not read as a cut",
      colour_match.find_cut(moving.clamp(0, 1), frame_utils.UNIT, 0) is None)

check("a cut is not looked for in too short a window",
      colour_match.find_cut(spliced[:, :3], frame_utils.UNIT, 0) is None)


# --- correcting only up to the cut -----------------------------------------
work = spliced.clone()
before_cut = work[:, :30].clone()
after_cut = work[:, 30:].clone()
cut_stats = colour_match.measure(work[:, :5])
colour_match.apply_to_window(
    work, frame_utils.UNIT,
    colour_match.solve(cut_stats,
                       colour_match.Stats(cut_stats.luma_mean + 0.02,
                                          cut_stats.luma_spread,
                                          cut_stats.cb_mean, cut_stats.cr_mean,
                                          cut_stats.chroma_spread),
                       limit=0.06),
    start=0, end=30)
check("frames before the cut are corrected",
      not torch.equal(work[:, :30], before_cut))
check("frames after the cut are left exactly as generated",
      torch.equal(work[:, 30:], after_cut),
      "the new scene has nothing to be matched to yet")

into = torch.empty_like(spliced)
colour_match.apply_to_window(spliced, frame_utils.UNIT,
                             colour_match.solve(cut_stats, cut_stats),
                             start=4, end=30, into=into)
check("the copy path carries across everything outside the corrected span",
      torch.equal(into[:, :4], spliced[:, :4])
      and torch.equal(into[:, 30:], spliced[:, 30:]))


# --- Continue Video --------------------------------------------------------
# The first window of a continuation arrives with frames from the end of the
# video being continued. Those are a real reference, and the old-to-new join
# is the most visible one there is.
continued = FakeRun()
continued.match_continue_anchor_only = True
continued._effective_scope = continued.match_scope
# CLEAN is short, so the strip is built by repeating it: the carried-over
# frames must be long enough to measure a span from, and the generated part
# long enough to hold a head and a tail that do not overlap it.
LONG = CLEAN.repeat(1, 4, 1, 1)
prefix = LONG[:, :20].clone()
fresh = drift(LONG[:, 20:60].clone(), brightness=0.02, saturation=0.95)
first_window = torch.cat([prefix, fresh], dim=1)
kept_video = first_window.clone()

corr, _ = match_step(continued, first_window, frame_utils.UNIT, 20)
check("a continuation's first window is matched to the carried-over frames",
      corr is not None,
      "without this it is the one join never checked")
check("the continuation is recognised", continued._continuing is True)
check("by default only the anchor is corrected there",
      continued._effective_scope == "anchor"
      and torch.equal(first_window, kept_video),
      "the reference is a short strip, not a whole window")

opted_in = FakeRun()
opted_in.match_continue_anchor_only = False
opted_in._effective_scope = opted_in.match_scope
window_video = torch.cat([prefix.clone(),
                          drift(LONG[:, 20:60].clone(), brightness=0.02)],
                         dim=1)
untouched_prefix = window_video[:, :20].clone()
match_step(opted_in, window_video, frame_utils.UNIT, 20)
check("switching it off corrects the continuation's video too",
      opted_in._effective_scope == "window")
check("the carried-over frames are never corrected either way",
      torch.equal(window_video[:, :20], untouched_prefix),
      "they belong to the video being continued")

# A fresh run has no carried-over frames, so nothing to match window 1 to.
scratch = FakeRun()
scratch._effective_scope = scratch.match_scope
scratch_corr, _ = match_step(scratch, CLEAN.clone(), frame_utils.UNIT, 0)
check("a fresh run still leaves its first window as generated",
      scratch_corr is None and scratch._continuing is False)


# --- the cut detector against hand-checked ground truth ---------------------
# Fast motion produces large frame-to-frame changes too. What separates a cut
# is that its neighbours are ordinary, where motion comes in runs. Measured on
# real footage: four hand-checked cuts scored 9.3 to 21.3 against their
# neighbours, the largest motion peak scored 3.6.
def synthetic_run(length, cut_at=None, motion=0.0):
    base = torch.rand(3, 1, 24, 32) * 0.4 + 0.2
    frames = base.repeat(1, length, 1, 1)
    if motion:
        frames = frames + torch.cumsum(
            torch.randn(3, length, 24, 32) * motion, dim=1)
    if cut_at is not None:
        other = (torch.rand(3, 1, 24, 32) * 0.5 + 0.45).repeat(
            1, length - cut_at, 1, 1)
        frames[:, cut_at:] = other + torch.randn(
            3, length - cut_at, 24, 32) * 0.005
    return frames.clamp(0, 1)


torch.manual_seed(21)
check("a cut is found at the right frame",
      colour_match.find_cut(synthetic_run(60, cut_at=25),
                            frame_utils.UNIT, 0) == 25)
check("a run with no cut reports none",
      colour_match.find_cut(synthetic_run(60), frame_utils.UNIT, 0) is None)
check("a fast pan is not mistaken for a cut",
      colour_match.find_cut(synthetic_run(60, motion=0.05),
                            frame_utils.UNIT, 0) is None,
      "motion changes several frames in a row, a cut changes one")
check("the first cut is returned when there are several",
      colour_match.find_cut(
          torch.cat([synthetic_run(30, cut_at=12),
                     synthetic_run(30, cut_at=20)], dim=1),
          frame_utils.UNIT, 0) == 12,
      "the correction has to stop at the first one")
check("cuts before the search start are ignored",
      colour_match.find_cut(synthetic_run(60, cut_at=10),
                            frame_utils.UNIT, 20) is None,
      "carried-over frames are not the plugin's business")
check("too short a window is not searched",
      colour_match.find_cut(synthetic_run(4), frame_utils.UNIT, 0) is None)
check("a still frame does not produce a cut out of noise",
      colour_match.find_cut(torch.full((3, 40, 16, 16), 0.5),
                            frame_utils.UNIT, 0) is None,
      "isolation alone would be enormous here")


# --- a cut must not be counted as noise ------------------------------------
# One pair of frame groups straddling a cut swamps the estimate. On measured
# footage that put the saturation floor at 0.200 where the true figure was
# 0.044 -- the difference between suppressing a real correction and applying
# it.
torch.manual_seed(31)
steady = (torch.rand(3, 1, 20, 28) * 0.3 + 0.25).repeat(1, 80, 1, 1)
steady = (steady + torch.randn(3, 80, 20, 28) * 0.01).clamp(0, 1)
# Inside the neighbourhood the noise floor is sampled from, and positioned so
# that a sampled pair straddles it. A cut beyond that neighbourhood is never
# sampled, so it cannot inflate the estimate in the first place.
with_cut = steady.clone()
with_cut[:, 25:] = (with_cut[:, 25:] * 0.5 + 0.42).clamp(0, 1)

clean_floor = colour_match.estimate_noise(steady, frame_utils.UNIT, 0, 5)
naive_floor = colour_match.estimate_noise(with_cut, frame_utils.UNIT, 0, 5)
aware_floor = colour_match.estimate_noise(with_cut, frame_utils.UNIT, 0, 5,
                                          cut=24)
check("a cut inflates the noise floor if it is counted",
      naive_floor.luma_shift > clean_floor.luma_shift * 2,
      f"{naive_floor.luma_shift:.4f} vs {clean_floor.luma_shift:.4f}")
check("leaving the cut out restores the true floor",
      aware_floor.luma_shift < naive_floor.luma_shift / 2,
      f"{aware_floor.luma_shift:.4f} vs {naive_floor.luma_shift:.4f}")
check("a real correction survives once the cut is excluded",
      not colour_match.solve(
          colour_match.measure(with_cut[:, :5]),
          colour_match.Stats(colour_match.measure(with_cut[:, :5]).luma_mean
                             + 0.015,
                             colour_match.measure(with_cut[:, :5]).luma_spread,
                             colour_match.measure(with_cut[:, :5]).cb_mean,
                             colour_match.measure(with_cut[:, :5]).cr_mean,
                             colour_match.measure(with_cut[:, :5]).chroma_spread),
          noise=aware_floor, limit=0.06).is_identity(),
      "it would have been suppressed by the inflated floor")
# The floor has to describe the join, not the window. Sampling the whole
# window measures content variation over a far longer stretch -- a camera
# move, a different part of the room -- which is a much larger number and
# suppresses real drift.
near = colour_match.estimate_noise(steady, frame_utils.UNIT, 0, 5)
far = colour_match.estimate_noise(steady, frame_utils.UNIT, 0, 5,
                                  neighbourhood=10_000)
check("the noise floor is sampled near the join, not across the window",
      colour_match.NOISE_NEIGHBOURHOOD < 200 and near is not None
      and far is not None,
      f"neighbourhood {colour_match.NOISE_NEIGHBOURHOOD} frames")

drifting = steady.clone()
for f in range(drifting.shape[1]):
    drifting[:, f] = (drifting[:, f] * (1.0 + 0.004 * f)).clamp(0, 1)
local = colour_match.estimate_noise(drifting, frame_utils.UNIT, 0, 5)
whole = colour_match.estimate_noise(drifting, frame_utils.UNIT, 0, 5,
                                    neighbourhood=10_000)
check("a slow change across the window does not inflate the local floor",
      local.luma_shift < whole.luma_shift,
      f"local {local.luma_shift:.5f} vs whole-window {whole.luma_shift:.5f}")

check("too few clear pairs falls back rather than giving up",
      colour_match.estimate_noise(steady[:, :24], frame_utils.UNIT, 0, 5,
                                  cut=10) is not None)


# --- anchoring the first window of a continuation --------------------------
# Every other anchor is the last frame of a window the plugin watched being
# generated. A continuation's first window has no such predecessor, so it was
# the one window that started with nothing to anchor to -- across the join
# between old video and new, which is the most visible join there is.
import os as _os
import tempfile as _tempfile

seed_anchor = lift("_seed_anchor_from_input", os=_os, torch=torch,
                   frame_to_rgb_uint8=frame_utils.frame_to_rgb_uint8)[
    "_seed_anchor_from_input"]


class Seeding:
    def __init__(self, window_no=0):
        self._window_no = window_no
        self.save_frames = False
        self._continuing = False
        self.written = None

    def _write_anchor(self, rgb):
        self.written = rgb

    def _save_frame(self, rgb):
        pass


carried = (torch.rand(3, 18, 12, 16) * 255).to(torch.uint8)
target = Seeding()
check("a continuation seeds an anchor from the video handed in",
      seed_anchor(target, {"input_video": carried}) is True
      and target.written is not None)
check("the anchor is the LAST frame of that video",
      np.array_equal(target.written,
                     carried[:, -1].permute(1, 2, 0).numpy()),
      "the frame the new window continues from")
check("the run is marked as a continuation", target._continuing is True)

check("a fresh run seeds nothing",
      seed_anchor(Seeding(), {}) is False,
      "there is no earlier video to anchor to")
check("starting from a still image seeds nothing",
      seed_anchor(Seeding(), {"input_video": carried,
                              "image_start": object()}) is False,
      "that is a new piece, not a continuation")
check("a missing anchor mid-run is not papered over",
      seed_anchor(Seeding(window_no=3), {"input_video": carried}) is False,
      "mid-run the anchor should already exist; hiding that would hide a bug")
check("a malformed carried-in video is refused",
      seed_anchor(Seeding(), {"input_video": torch.rand(5, 4)}) is False)
check("an empty carried-in video is refused",
      seed_anchor(Seeding(), {"input_video": torch.rand(3, 0, 8, 8)}) is False)


# --- widening the reference at a continuation join -------------------------
# Both sides of a continuation join sit in the same window, so the span can be
# widened symmetrically. It has to stay symmetric: a swathe of old footage
# against a sliver of new would read every camera move between them as drift.
def continuation(prefix_frames, new_frames, cut_in_prefix=None,
                 cut_in_new=None):
    strip = LONG[:, :prefix_frames].clone()
    fresh = LONG[:, prefix_frames:prefix_frames + new_frames].clone()
    if cut_in_prefix is not None:
        strip[:, cut_in_prefix:] = (strip[:, cut_in_prefix:] * 0.4
                                    + 0.5).clamp(0, 1)
    if cut_in_new is not None:
        fresh[:, cut_in_new:] = (fresh[:, cut_in_new:] * 0.4 + 0.5).clamp(0, 1)
    return torch.cat([strip, fresh], dim=1), prefix_frames


def span_used(run, window, start):
    """The span the run settled on, read back from the reference it built."""
    run._reference_stats = None
    match_step(run, window, frame_utils.UNIT, start)
    return run._reference_stats


# CLEAN is 12 frames, so this gives 360 -- enough for a 120-frame
# carried-in strip and a 120-frame window either side of the join.
LONG = CLEAN.repeat(1, 30, 1, 1)


def spanning_run(maximum):
    r = FakeRun()
    r.match_span = maximum
    r._effective_scope = r.match_scope
    return r


# With more carried-in frames than the maximum, the maximum binds.
wide, s = continuation(120, 120)
r = spanning_run(48)
r._reference_stats = None
match_step(r, wide.clone(), frame_utils.UNIT, s)
check("the maximum bounds how far the reference is widened",
      r._continuing is True,
      "120 frames carried in, 48 allowed")

# With fewer carried-in frames than the maximum, all of them are used.
narrow, s2 = continuation(18, 120)
r2 = spanning_run(48)
match_step(r2, narrow.clone(), frame_utils.UNIT, s2)
check("a short carried-in strip is used in full rather than refused",
      r2._continuing is True and r2._reference_stats is not None,
      "18 frames carried in, 48 allowed")

# The new side can be the binding constraint too, keeping it symmetric.
lopsided, s3 = continuation(120, 12)
r3 = spanning_run(48)
match_step(r3, lopsided.clone(), frame_utils.UNIT, s3)
check("the span never exceeds what the new side can match",
      r3._reference_stats is not None,
      "symmetry is what keeps both sides on the same moment")

# A fresh run is unaffected by any of this.
r4 = spanning_run(48)
fresh_corr, _ = match_step(r4, CLEAN.clone(), frame_utils.UNIT, 0)
check("widening does not apply to a run that starts from scratch",
      fresh_corr is None and r4._continuing is False)

check("find_cut can search backwards for the last cut before a point",
      colour_match.find_cut(
          torch.cat([synthetic_run(30, cut_at=12),
                     synthetic_run(30, cut_at=20)], dim=1),
          frame_utils.UNIT, 0, last=True) == 50,
      "the reference must not reach back past a cut")
check("find_cut can be bounded to a range",
      colour_match.find_cut(synthetic_run(60, cut_at=40),
                            frame_utils.UNIT, 0, end=30) is None,
      "a cut outside the range searched is not reported")


# --- a cut cannot be called at the edge of the range searched ---------------
# At either end there is nothing on one side to compare against, so a value
# merely rising towards the edge scores as an isolated spike. That is how an
# 18-frame carried-in strip reported a cut on its own last frame and collapsed
# the reference to a single frame.
rising = torch.rand(3, 1, 16, 20).repeat(1, 18, 1, 1)
for f in range(18):
    rising[:, f] = (rising[:, f] * (1.0 + 0.09 * f)).clamp(0, 1)
check("a trend running into the end of the strip is not called a cut",
      colour_match.find_cut(rising, frame_utils.UNIT, 0, last=True) is None,
      "no right-hand neighbours there to judge it against")

edge = synthetic_run(18)
edge[:, 17:] = (edge[:, 17:] * 0.4 + 0.5).clamp(0, 1)
check("a change on the very last frame is not called a cut",
      colour_match.find_cut(edge, frame_utils.UNIT, 0, last=True) is None)

edge2 = synthetic_run(18)
edge2[:, 1:] = (edge2[:, 1:] * 0.4 + 0.5).clamp(0, 1)
check("a change on the very first frame is not called a cut either",
      colour_match.find_cut(edge2, frame_utils.UNIT, 0) is None,
      "the rejection gate handles a scene change at the join")

check("a cut with room either side is still found",
      colour_match.find_cut(synthetic_run(30, cut_at=15),
                            frame_utils.UNIT, 0) == 15)
check("a range too short to have any interior is refused",
      colour_match.find_cut(synthetic_run(5, cut_at=2),
                            frame_utils.UNIT, 0) is None,
      "nothing in it has neighbours on both sides")


# --- a cut too close to a continuation join ---------------------------------
# Then the old video's tail is already a different scene, so it is not a
# reference at all and the window is better left alone.
# Because a cut needs two frames of margin to be called at all, a detected
# cut always leaves at least three usable frames. So the floor bites on a
# carried-in strip that is simply too short to measure.
tight = FakeRun()
tight.match_span = 48
tight._effective_scope = tight.match_scope
tight_window = torch.cat([LONG[:, :2].clone(), LONG[:, 2:62].clone()], dim=1)
kept = tight_window.clone()
tight_corr, tight_repl = match_step(tight, tight_window, frame_utils.UNIT, 2)
check("too few frames before the join leaves the window uncorrected",
      tight_corr is None and tight_repl is None
      and torch.equal(tight_window, kept),
      "one or two frames is not a reference")
check("it is still recognised as a continuation",
      tight._continuing is True)


# --- the noise floor must not widen to suit a large span -------------------
# A large span used to push the floor's own neighbourhood out to several
# times its length, so it went back to describing a whole window's content
# variation rather than the join -- which is what made it blind to real
# steps. The neighbourhood is fixed regardless of span.
drifty = CLEAN.repeat(1, 20, 1, 1).clone()
for f in range(drifty.shape[1]):
    drifty[:, f] = (drifty[:, f] * (1.0 + 0.002 * f)).clamp(0, 1)

small_span = colour_match.estimate_noise(drifty, frame_utils.UNIT, 0, 5)
big_span = colour_match.estimate_noise(drifty, frame_utils.UNIT, 0, 16)
check("a large span does not push the noise floor out over the window",
      big_span is not None and small_span is not None
      and big_span.luma_shift < 0.02,
      f"span 5 gives {small_span.luma_shift:.5f}, "
      f"span 16 gives {big_span.luma_shift:.5f}")
check("the neighbourhood stays put regardless of span",
      colour_match.NOISE_NEIGHBOURHOOD == 40,
      "it describes the join, not the window")


# --- colour matching without anchoring -------------------------------------
# The two are independent: matching measures drift across a join and corrects
# the window, which needs no injected frame. Anchoring off must not switch it
# off as well.
class Detached(FakeRun):
    def __init__(self, scope="window", enabled=False):
        FakeRun.__init__(self, scope=scope)
        self.enabled = enabled


detached = Detached()
match_step(detached, CLEAN.clone(), frame_utils.UNIT, 0)
drifted_alone = drift(CLEAN.clone(), brightness=0.03, saturation=0.95)
kept_alone = drifted_alone.clone()
alone_corr, _ = match_step(detached, drifted_alone, frame_utils.UNIT, 0)
check("colour matching still runs with anchoring switched off",
      alone_corr is not None,
      "it needs a join, not an injected frame")
check("and it still corrects the window",
      not torch.equal(drifted_alone, kept_alone))

# Anchor scope with anchoring off would correct a frame that is never
# written and never injected, so it would do nothing at all.
dead = Detached(scope="anchor")
match_step(dead, CLEAN.clone(), frame_utils.UNIT, 0)
check("anchor scope falls back to whole window when anchoring is off",
      dead._effective_scope == "window",
      "otherwise the correction would have nowhere to go")
check("the fallback is warned about once, not every window",
      dead._warned_dead_scope is True)

# With anchoring on, anchor scope is honoured as asked.
live = Detached(scope="anchor", enabled=True)
match_step(live, CLEAN.clone(), frame_utils.UNIT, 0)
check("anchor scope is left alone when anchoring is on",
      live._effective_scope == "anchor")


# --- one video at a time ----------------------------------------------------
# A queue runs several videos through the same process back to back, and the
# anchor frame, the window count and the colour reference all describe one
# video. Wan2GP numbers the windows of each video from 1, so the opening
# window of the next item in a queue can be told apart from the next window of
# the current video. Without that, the second video in a queue opened by
# injecting the last frame of the first.
boundary = lift("_starts_new_video", "_note_window", "_remember_position",
                "_overlap")


class Queued:
    _starts_new_video = boundary["_starts_new_video"]
    _note_window = boundary["_note_window"]
    _remember_position = boundary["_remember_position"]
    _overlap = boundary["_overlap"]

    def __init__(self, window_no=0, anchor=None):
        self._window_no = window_no
        self._anchor_path = anchor
        self._reference_stats = None
        self._last_window_no = None
        self._last_window_start = None
        self.resets = 0

    def _reset_run(self):
        self.resets += 1
        self._window_no = 0
        self._anchor_path = None
        self._reference_stats = None
        self._last_window_no = None
        self._last_window_start = None


def play(windows, start=None):
    """Run a sequence of window numbers through a fresh plugin."""
    plugin = start if start is not None else Queued()
    for number in windows:
        plugin._note_window({"window_no": number})
        # Stand in for the capture that follows a real window.
        plugin._window_no += 1
        plugin._anchor_path = f"anchor_{plugin._window_no:04d}.png"
    return plugin


check("the first window of a session clears nothing",
      play([1]).resets == 0,
      "there is nothing carried over to clear")
check("later windows of the same video are not new videos",
      play([1, 2, 3, 4]).resets == 0,
      "this is the case the plugin exists to serve")
check("the next item in a queue is a new video",
      play([1, 2, 3, 1, 2]).resets == 1)
queued = Queued()
play([1, 2, 3], start=queued)
before = queued._anchor_path
queued._note_window({"window_no": 1})
check("the previous video's anchor is gone before the next one is injected",
      before is not None and queued._anchor_path is None,
      "otherwise video two opens on video one's last frame")
check("and its window count starts again",
      queued._window_no == 0)

check("each repeat of the same item is a new video too",
      play([1, 2, 1, 2, 1, 2]).resets == 2,
      "Wan2GP restarts the numbering for every repeat")

# A queued Continue Video run is the case that went wrong twice over: it opens
# with frames carried in, which reads like a mid-video window, and its own
# seeding was refused because the window count had not been cleared.
continued = play([1, 2, 3])
continued._note_window({"window_no": 1, "prefix_frames_count": 18})
check("a queued continuation is a new video despite carrying frames in",
      continued.resets == 1 and continued._anchor_path is None)

check("a window number that never restarts never clears anything",
      play([4, 5, 6, 7], start=Queued()).resets == 0,
      "joining mid-video, as when the plugin is switched on part-way")

# --- the same, on a build that does not pass a window number ----------------
# The frame a window begins at counts up through a video and starts over with
# the next, so it can stand in.
positions = Queued()
positions._window_no = 2
positions._anchor_path = "anchor_0002.png"
for frame in (0, 344, 688):
    positions._note_window({"window_start_frame_no": frame})
check("a window start of 0 opens a new video",
      positions.resets == 1,
      "the first of the three, before the count was carried")
positions._note_window({"window_start_frame_no": 1032})
check("a window start that keeps climbing stays in the same video",
      positions.resets == 1)

restart = Queued(window_no=3, anchor="anchor_0003.png")
restart._note_window({"window_start_frame_no": 900})
restart._note_window({"window_start_frame_no": 344})
check("a window start that goes backwards opens a new video",
      restart.resets == 1)

# With neither, the carried-over frames are all that is left to go on.
bare = Queued(window_no=3, anchor="anchor_0003.png")
bare._note_window({"prefix_frames_count": 18})
check("with nothing else to read, carried-in frames read as the same video",
      bare.resets == 0)
bare._note_window({"prefix_frames_count": 0})
check("and a window with none reads as a new one",
      bare.resets == 1)

# Malformed values fall through to the next signal rather than raising.
messy = Queued(window_no=3, anchor="anchor_0003.png")
messy._note_window({"window_no": "nonsense", "window_start_frame_no": 0})
check("an unreadable window number falls through to the next signal",
      messy.resets == 1)
check("an unreadable value is not remembered as a position",
      messy._last_window_no is None)

check("a missing window number is not mistaken for window 1",
      Queued(window_no=3, anchor="a.png")._starts_new_video(
          {"prefix_frames_count": 18})[0] is False)


print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all checks passed")
