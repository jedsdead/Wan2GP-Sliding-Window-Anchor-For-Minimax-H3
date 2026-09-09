"""
Colour, saturation and brightness matching between sliding windows.

Each window is VAE-encoded and re-synthesised from the anchor frame handed to
it, and that round trip is never exact: blacks lift, contrast softens, colour
desaturates a little. Because every window is conditioned on the end of the
one before it, the error is inherited and compounds, so a long generation
slowly washes out.

This measures that drift on matched content -- the frames either side of a
window join show the same moment, so anything that differs between them is
drift rather than the scene changing -- and undoes it.

The correction is a per-channel affine in Rec. 709 Y'CbCr, which splits into
exactly the axes worth controlling separately:

    Y  mean   brightness
    Y  spread contrast
    CbCr mean colour cast
    CbCr spread saturation

Y'CbCr is a linear transform of RGB, so the whole correction collapses into a
single 3x3 matrix and offset applied straight to RGB. That matters: a window
can be a billion values, and this way it costs one matmul per chunk with no
colour-space round trip and no per-pixel gamma.

CIELAB would track perceived brightness more faithfully, but converting a
whole window through it costs a cube root per value and several full-size
intermediates. On the size of drift being corrected here -- a few percent --
the difference is not worth that.
"""

import numpy as np
import torch

from .frame_utils import chunk_to_float01, from_float01

# Rec. 709 luma weights, matching the primaries the models are trained on.
LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722
CB_SCALE = 1.8556  # normalises B - Y into [-0.5, 0.5]
CR_SCALE = 1.5748  # normalises R - Y into [-0.5, 0.5]

RGB_TO_YCC = np.array([
    [LUMA_R, LUMA_G, LUMA_B],
    [-LUMA_R / CB_SCALE, -LUMA_G / CB_SCALE, (1.0 - LUMA_B) / CB_SCALE],
    [(1.0 - LUMA_R) / CR_SCALE, -LUMA_G / CR_SCALE, -LUMA_B / CR_SCALE],
], dtype=np.float64)

YCC_TO_RGB = np.linalg.inv(RGB_TO_YCC)

# Above this many pixels the statistics are taken from a spatial subsample.
# Means and spreads converge long before this; the cap keeps a 4K window from
# turning a measurement into a memory event.
MAX_STAT_PIXELS = 4_000_000

# Values converted to float at once when correcting a window. Roughly 128 MB
# of float32, and the working set is a small multiple of that.
CHUNK_VALUE_BUDGET = 32_000_000

# A spread below this is a flat frame -- a fade to black, a blown highlight.
# Dividing by it would produce an enormous gain from no real signal.
MIN_SPREAD = 1e-4

IDENTITY_TOLERANCE = 1e-4


class Stats:
    """Colour statistics of a group of frames, in Y'CbCr."""

    __slots__ = ("luma_mean", "luma_spread", "cb_mean", "cr_mean",
                 "chroma_spread")

    def __init__(self, luma_mean, luma_spread, cb_mean, cr_mean,
                 chroma_spread):
        self.luma_mean = float(luma_mean)
        self.luma_spread = float(luma_spread)
        self.cb_mean = float(cb_mean)
        self.cr_mean = float(cr_mean)
        self.chroma_spread = float(chroma_spread)

    def __repr__(self):
        return (f"Stats(luma={self.luma_mean:.4f}+-{self.luma_spread:.4f}, "
                f"cb={self.cb_mean:+.4f}, cr={self.cr_mean:+.4f}, "
                f"chroma={self.chroma_spread:.4f})")


class Correction:
    """
    An affine correction, held both as Y'CbCr parameters and as the RGB
    matrix they collapse into.

    The Y'CbCr form is what gets reported and what transforms statistics
    analytically; the RGB form is what actually touches pixels.
    """

    __slots__ = ("luma_gain", "chroma_gain", "luma_out_mean", "cb_out_mean",
                 "cr_out_mean", "source", "matrix", "offset", "rejected")

    def __init__(self, luma_gain, chroma_gain, luma_out_mean, cb_out_mean,
                 cr_out_mean, source):
        self.luma_gain = float(luma_gain)
        self.chroma_gain = float(chroma_gain)
        self.luma_out_mean = float(luma_out_mean)
        self.cb_out_mean = float(cb_out_mean)
        self.cr_out_mean = float(cr_out_mean)
        self.source = source

        # v' = A v + t in Y'CbCr, so rgb' = (Cinv A C) rgb + Cinv t.
        gains = np.diag([self.luma_gain, self.chroma_gain, self.chroma_gain])
        offsets = np.array([
            self.luma_out_mean - self.luma_gain * source.luma_mean,
            self.cb_out_mean - self.chroma_gain * source.cb_mean,
            self.cr_out_mean - self.chroma_gain * source.cr_mean,
        ], dtype=np.float64)

        self.matrix = (YCC_TO_RGB @ gains @ RGB_TO_YCC).astype(np.float32)
        self.offset = (YCC_TO_RGB @ offsets).astype(np.float32)
        self.rejected = False

    def is_identity(self):
        return (abs(self.luma_gain - 1.0) < IDENTITY_TOLERANCE
                and abs(self.chroma_gain - 1.0) < IDENTITY_TOLERANCE
                and abs(self.luma_out_mean - self.source.luma_mean) < IDENTITY_TOLERANCE
                and abs(self.cb_out_mean - self.source.cb_mean) < IDENTITY_TOLERANCE
                and abs(self.cr_out_mean - self.source.cr_mean) < IDENTITY_TOLERANCE)

    def report(self):
        """The correction in the terms the panel and console use."""
        return {
            "brightness": self.luma_out_mean - self.source.luma_mean,
            "contrast": self.luma_gain,
            "saturation": self.chroma_gain,
            "cast_cb": self.cb_out_mean - self.source.cb_mean,
            "cast_cr": self.cr_out_mean - self.source.cr_mean,
        }

    def describe(self):
        r = self.report()
        return (f"brightness {r['brightness'] * 100:+.2f}%, "
                f"contrast x{r['contrast']:.3f}, "
                f"saturation x{r['saturation']:.3f}, "
                f"cast Cb {r['cast_cb'] * 100:+.2f}% "
                f"Cr {r['cast_cr'] * 100:+.2f}%")


def measure(span01):
    """
    Colour statistics of a float [0, 1] tensor shaped (3, ...).

    Saturation is taken as the RMS of the two chroma channels about their own
    means, as a single figure rather than one per channel. Scaling Cb and Cr
    by different amounts would shift hue while claiming to change only
    saturation; anything genuinely per-channel belongs in the cast term.
    """
    flat = span01.reshape(3, -1).to(torch.float32)

    count = flat.shape[1]
    if count > MAX_STAT_PIXELS:
        flat = flat[:, ::max(1, count // MAX_STAT_PIXELS)]

    matrix = torch.as_tensor(RGB_TO_YCC, dtype=torch.float32,
                             device=flat.device)
    ycc = matrix @ flat

    luma_mean = ycc[0].mean()
    luma_spread = ycc[0].std()
    cb_mean = ycc[1].mean()
    cr_mean = ycc[2].mean()
    chroma_spread = torch.sqrt(
        ((ycc[1] - cb_mean) ** 2 + (ycc[2] - cr_mean) ** 2).mean()
    )

    return Stats(luma_mean.item(), luma_spread.item(), cb_mean.item(),
                 cr_mean.item(), chroma_spread.item())


def _raw_gain(source_spread, target_spread):
    if source_spread < MIN_SPREAD or target_spread < MIN_SPREAD:
        return 1.0
    return float(target_spread / source_spread)


def looks_like_content(source, target, limit):
    """
    Whether a measured difference is too large to be drift.

    Reconstruction drift is small by nature -- a percent or two per window.
    A difference several times that is not severe drift, it is evidence the
    two groups of frames are not showing the same moment: a cut, a hard
    lighting change, a camera whipping to a different part of the scene.

    Capping such a measurement and applying it anyway is the worst of both
    worlds, because the cap is still far larger than any real drift, so a
    new scene gets dragged bodily toward the grade of the old one. Better to
    recognise the measurement as meaningless and leave the window alone.
    """
    if abs(target.luma_mean - source.luma_mean) > limit:
        return True
    if abs(target.cb_mean - source.cb_mean) > limit * 0.5:
        return True
    if abs(target.cr_mean - source.cr_mean) > limit * 0.5:
        return True
    for gain in (_raw_gain(source.luma_spread, target.luma_spread),
                 _raw_gain(source.chroma_spread, target.chroma_spread)):
        if gain > 1.0 + limit or gain < 1.0 / (1.0 + limit):
            return True
    return False


def _clamped_gain(source_spread, target_spread, limit):
    if source_spread < MIN_SPREAD or target_spread < MIN_SPREAD:
        return 1.0
    gain = target_spread / source_spread
    # Bounded multiplicatively, so halving and doubling are equally far from
    # no change. A linear bound would let the gain fall much further than it
    # can rise.
    return float(min(max(gain, 1.0 / (1.0 + limit)), 1.0 + limit))


class Noise:
    """
    How much these statistics wobble where no drift can exist.

    Measured by comparing neighbouring groups of frames inside a single
    window. Any difference found there is measurement error and content
    movement, since there is no window join between them.
    """

    __slots__ = ("luma_shift", "luma_gain", "chroma_gain", "cb_shift",
                 "cr_shift")

    def __init__(self, luma_shift, luma_gain, chroma_gain, cb_shift,
                 cr_shift):
        self.luma_shift = float(luma_shift)
        self.luma_gain = float(luma_gain)
        self.chroma_gain = float(chroma_gain)
        self.cb_shift = float(cb_shift)
        self.cr_shift = float(cr_shift)

    def describe(self):
        return (f"brightness +-{self.luma_shift * 100:.2f}%, "
                f"contrast +-{self.luma_gain:.3f}, "
                f"saturation +-{self.chroma_gain:.3f}")


# How far past the start of the generated content to look when measuring the
# noise floor. The floor has to describe how much these statistics move
# between neighbouring groups of frames, because that is what a join compares.
# Sampling the whole window instead measures content variation over a much
# longer stretch -- a camera move, a different part of the room -- which is a
# far larger number and suppresses real drift. On measured footage the
# whole-window figure was 5.6 times the local one, the difference between
# reading a genuine 4.5-sigma step as significant and dismissing it at 0.8.
NOISE_NEIGHBOURHOOD = 40


def estimate_noise(sample, kind, start, span, samples=8, cut=None,
                   neighbourhood=NOISE_NEIGHBOURHOOD):
    """
    Measure the noise floor inside one window.

    Real footage moves. Grain, a passing highlight and the scene itself all
    shift these statistics from frame to frame, and on dark or low-chroma
    material that wobble can be several times larger than the drift being
    looked for. Comparing neighbouring groups within a window, where no join
    exists and the true answer is therefore no change, says how much of a
    measured difference at a join is worth believing.

    Pairs that straddle a cut are left out. A cut is not noise, and a single
    pair spanning one swamps the estimate: on measured footage, including it
    put the saturation floor at 0.200 where the true figure was 0.044, which
    is the difference between suppressing a real correction and applying it.

    Returns None when the window is too short to sample.
    """
    total = int(sample.shape[1])
    if span < 1 or total - start < 2 * span + 1:
        return None

    # Kept near the join. Deliberately NOT widened to suit a large span:
    # that is how the floor came to describe a whole window's content
    # variation instead of the join, which made it blind to real steps.
    limit = min(total, start + max(int(neighbourhood), 2 * span + 1))
    if limit - start < 2 * span + 1:
        limit = total

    last = limit - 2 * span
    count = max(2, min(samples, (last - start) + 1))
    positions = [int(p) for p in np.linspace(start, last, count)]
    if cut is not None:
        clear = [p for p in positions if not p <= cut < p + 2 * span]
        # Fall back to sampling everything rather than return nothing: a
        # window shorter than a few spans past the cut has nowhere else to
        # look, and a poor estimate beats none.
        if len(clear) >= 2:
            positions = clear

    luma_shift, luma_gain, chroma_gain, cb_shift, cr_shift = [], [], [], [], []
    for begin in positions:
        first = measure(chunk_to_float01(
            sample[:, begin:begin + span], kind))
        second = measure(chunk_to_float01(
            sample[:, begin + span:begin + 2 * span], kind))
        luma_shift.append(first.luma_mean - second.luma_mean)
        luma_gain.append(first.luma_spread
                         / max(second.luma_spread, MIN_SPREAD))
        chroma_gain.append(first.chroma_spread
                           / max(second.chroma_spread, MIN_SPREAD))
        cb_shift.append(first.cb_mean - second.cb_mean)
        cr_shift.append(first.cr_mean - second.cr_mean)

    return Noise(
        luma_shift=np.std(luma_shift),
        luma_gain=np.std(luma_gain),
        chroma_gain=np.std(chroma_gain),
        cb_shift=np.std(cb_shift),
        cr_shift=np.std(cr_shift),
    )


def _shrink(measured, sigma):
    """
    Hold back the part of a measurement that could be noise.

    A difference the size of the noise is thrown away entirely; one far
    larger passes through almost untouched. Without this the correction
    chases the wobble, and since it is applied every window and each window
    inherits the last, chasing noise does not average out -- it accumulates
    into exactly the drift the feature exists to prevent.
    """
    if sigma is None or sigma <= 0.0:
        return measured
    if measured == 0.0:
        return 0.0
    keep = 1.0 - (sigma * sigma) / (measured * measured)
    return measured * max(0.0, keep)


def solve(source, target, brightness=True, contrast=True, saturation=True,
          colour=True, strength=1.0, limit=0.06, noise=None,
          reject=True):
    """
    Build the correction taking `source` statistics onto `target`.

    `noise` is the wobble measured where no drift exists, and anything not
    clearly larger than it is held back. Pass None to correct the raw
    difference, which is only safe on material known to be steady.

    `limit` is the point beyond which a measurement stops being read as
    drift at all. The measurement assumes the frames either side of a join
    show the same moment; a cut breaks that, and the giveaway is the size of
    the result, since real drift never reaches that far. Anything past the
    limit is rejected outright rather than capped, and `reject=False` turns
    that off for material known to be continuous.
    """
    strength = float(min(max(strength, 0.0), 1.0))
    limit = float(max(limit, 0.0))

    if reject and looks_like_content(source, target, limit):
        identity = Correction(1.0, 1.0, source.luma_mean, source.cb_mean,
                              source.cr_mean, source)
        identity.rejected = True
        return identity

    luma_gain = _clamped_gain(source.luma_spread, target.luma_spread,
                              limit) if contrast else 1.0
    chroma_gain = _clamped_gain(source.chroma_spread, target.chroma_spread,
                                limit) if saturation else 1.0

    # Chroma spans half the range luma does, so it gets half the headroom.
    luma_shift = _clamped_shift(source.luma_mean, target.luma_mean,
                                limit) if brightness else 0.0
    cb_shift = _clamped_shift(source.cb_mean, target.cb_mean,
                              limit * 0.5) if colour else 0.0
    cr_shift = _clamped_shift(source.cr_mean, target.cr_mean,
                              limit * 0.5) if colour else 0.0

    if noise is not None:
        luma_gain = 1.0 + _shrink(luma_gain - 1.0, noise.luma_gain)
        chroma_gain = 1.0 + _shrink(chroma_gain - 1.0, noise.chroma_gain)
        luma_shift = _shrink(luma_shift, noise.luma_shift)
        cb_shift = _shrink(cb_shift, noise.cb_shift)
        cr_shift = _shrink(cr_shift, noise.cr_shift)

    luma_gain = 1.0 + (luma_gain - 1.0) * strength
    chroma_gain = 1.0 + (chroma_gain - 1.0) * strength

    return Correction(
        luma_gain=luma_gain,
        chroma_gain=chroma_gain,
        luma_out_mean=source.luma_mean + luma_shift * strength,
        cb_out_mean=source.cb_mean + cb_shift * strength,
        cr_out_mean=source.cr_mean + cr_shift * strength,
        source=source,
    )


def _clamped_shift(source_mean, target_mean, limit):
    return float(min(max(target_mean - source_mean, -limit), limit))


def transform_stats(stats, correction):
    """
    The statistics a group of frames would have after correction.

    Exact for the affine itself, and slightly optimistic where correction
    pushes values past the ends of the range and they are clipped back.
    """
    return Stats(
        luma_mean=(correction.luma_gain * stats.luma_mean
                   + correction.luma_out_mean
                   - correction.luma_gain * correction.source.luma_mean),
        luma_spread=correction.luma_gain * stats.luma_spread,
        cb_mean=(correction.chroma_gain * stats.cb_mean
                 + correction.cb_out_mean
                 - correction.chroma_gain * correction.source.cb_mean),
        cr_mean=(correction.chroma_gain * stats.cr_mean
                 + correction.cr_out_mean
                 - correction.chroma_gain * correction.source.cr_mean),
        chroma_spread=correction.chroma_gain * stats.chroma_spread,
    )


def apply_to_frame(rgb_uint8, correction):
    """Correct a single (H, W, 3) uint8 frame."""
    values = rgb_uint8.astype(np.float32) / 255.0
    corrected = values @ correction.matrix.T + correction.offset
    return np.round(np.clip(corrected, 0.0, 1.0) * 255).astype(np.uint8)


# A cut is a single frame unlike the one before it whose neighbours are
# ordinary. Fast motion also produces large frame-to-frame changes, but they
# come in runs of several frames, so measuring a peak against its immediate
# neighbours separates the two where measuring it against the median of the
# whole window does not.
#
# Ground truth from 1376x576 footage containing four hand-checked cuts: the
# cuts scored 9.3, 13.6, 20.6 and 21.3 against their neighbours, while the
# largest change caused by motion alone scored 3.6.
CUT_ISOLATION = 6.0

# A peak also has to be large in its own right. Without this, a nearly still
# scene would produce enormous isolation ratios out of sensor-level noise.
CUT_PROMINENCE = 3.0

# How many frames either side count as neighbours. Wide enough to span a run
# of fast motion, narrow enough that two nearby cuts do not mask each other.
CUT_NEIGHBOURHOOD = 2

# Frames are compared at every Nth pixel. A cut changes the whole frame, so
# there is no need to look at all of it, and this keeps the pass cheap enough
# to run on every window.
CUT_STRIDE = 8


def find_cut(sample, kind, start=0, isolation=CUT_ISOLATION,
             prominence=CUT_PROMINENCE, end=None, last=False):
    """
    The first frame at or after `start` where the picture cuts, or None.

    A correction is worked out from the frames at the start of a window, so
    it only describes the scene those frames belong to. If the picture cuts
    part way through, everything after the cut has no valid reference to be
    corrected toward and is better left alone -- and the seam that leaves
    falls exactly on the cut, where nothing can be seen.

    A peak has to clear two bars: it must stand well above its immediate
    neighbours, which is what a cut does and a fast pan does not, and it
    must be large against the window as a whole.
    """
    total = int(sample.shape[1]) if end is None else min(int(end),
                                                        int(sample.shape[1]))
    if total - start < 5:
        return None

    small = chunk_to_float01(
        sample[:, start:total, ::CUT_STRIDE, ::CUT_STRIDE], kind)
    change = (small[:, 1:] - small[:, :-1]).abs().mean(dim=(0, 2, 3))
    count = int(change.numel())
    if count < 5:
        return None

    # A floor rather than a bail-out: a locked-off shot of a still subject
    # has almost no frame-to-frame change, and a cut in the middle of one is
    # exactly the case worth catching. The isolation test still has to pass,
    # which uniform noise cannot do.
    middle = max(float(change.median()), 1e-4)

    values = change.tolist()
    # Only frames with ordinary neighbours on BOTH sides can be judged. At
    # either end of the range there is nothing on one side to compare
    # against, so a value merely rising towards the edge scores as an
    # isolated spike -- which is how an 18-frame carried-in strip reported a
    # cut on its own last frame.
    first = CUT_NEIGHBOURHOOD
    stop_at = count - CUT_NEIGHBOURHOOD
    if stop_at <= first:
        return None
    order = (range(stop_at - 1, first - 1, -1) if last
             else range(first, stop_at))
    for index in order:
        value = values[index]
        if value < middle * prominence:
            continue
        low = index - CUT_NEIGHBOURHOOD
        high = index + CUT_NEIGHBOURHOOD + 1
        neighbours = [values[j] for j in range(low, high) if j != index]
        if value / max(max(neighbours), 1e-9) >= isolation:
            return start + index + 1
    return None


def apply_to_window(sample, kind, correction, start=0, chunk=None, into=None,
                    end=None):
    """
    Correct frames `start` onward of a (3, T, H, W) window.

    Writes into `sample` itself unless `into` is given, in which case
    `sample` is left untouched and `into` receives the whole window --
    frames before `start` copied across as they are, the rest corrected.
    A pipeline running under inference mode hands back tensors that refuse
    in-place writes, and that is the way round it.

    Worked through in chunks rather than all at once. A window at any real
    resolution runs to hundreds of millions of values, and converting the
    whole of it to float would need a second copy the same size again.

    The chunk is sized by value count rather than frame count, so the working
    set stays put as the resolution rises. A fixed number of frames would
    quietly grow from a couple of hundred megabytes at 720p to a couple of
    gigabytes at 4K, on a card already holding the model.

    Returns the fraction of values that had to be clipped back into range,
    which is the signal that a correction was too strong for the material.
    """
    total = int(sample.shape[1])
    start = max(0, min(start, total))
    stop = total if end is None else max(start, min(int(end), total))
    target = sample if into is None else into

    # Anything outside the corrected span is carried across untouched.
    if into is not None:
        if start > 0:
            into[:, :start] = sample[:, :start]
        if stop < total:
            into[:, stop:] = sample[:, stop:]
    if start >= stop:
        return 0.0

    if chunk is None:
        per_frame = max(1, int(sample.shape[0] * sample.shape[2]
                               * sample.shape[3]))
        chunk = max(1, min(stop - start, CHUNK_VALUE_BUDGET // per_frame))

    matrix = torch.as_tensor(correction.matrix, dtype=torch.float32,
                             device=sample.device)
    offset = torch.as_tensor(correction.offset, dtype=torch.float32,
                             device=sample.device).reshape(3, 1, 1, 1)

    clipped = 0
    counted = 0

    for begin in range(start, stop, chunk):
        end = min(begin + chunk, stop)
        block = chunk_to_float01(sample[:, begin:end], kind)
        shape = block.shape

        block = (matrix @ block.reshape(3, -1)).reshape(shape)
        block += offset

        out_of_range = (block < 0.0) | (block > 1.0)
        clipped += int(out_of_range.sum().item())
        counted += out_of_range.numel()

        block.clamp_(0.0, 1.0)
        target[:, begin:end] = from_float01(block, kind, sample.dtype)

    return clipped / counted if counted else 0.0
