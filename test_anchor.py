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

from frame_utils import to_float01, frame_to_rgb_uint8

FAILED = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))
    if not condition:
        FAILED.append(name)


def lift(*names):
    """Pull named methods out of plugin.py without importing it."""
    source = open("plugin.py").read()
    tree = ast.parse(source)
    # The methods reference module-level constants, so hand them over too.
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            namespace = dict(constants)
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         "<extracted>", "exec"), namespace)
            found[node.name] = namespace[node.name]
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"could not lift {missing} from plugin.py")
    return found


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


print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all checks passed")
