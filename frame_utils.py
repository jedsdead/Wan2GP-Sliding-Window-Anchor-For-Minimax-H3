"""
Frame conversion helpers.

Pulls a single frame out of a decoded window as an image the plugin can save
and feed back in on the next window, and converts frames to and from the
[0, 1] float form that colour matching works in.

WanGP hands windows back in several formats. The format is sniffed once per
window and then carried around as a `kind`, rather than sniffed again on the
way back. Sniffing works on the range of the values present, and correcting a
window changes that range -- a window read as signed could easily be written
back as though it were unit-range, silently halving the video.
"""
import numpy as np
import torch

UINT8 = "uint8"      # 0..255 integers
SIGNED = "signed"    # float, -1..1
SCALED = "float255"  # float, 0..255
UNIT = "unit"        # float, 0..1


def detect_kind(sample):
    """
    Work out which of the four formats a decoded window is in.

    The range is sniffed rather than assumed, because a wrong guess here
    silently destroys the image.
    """
    if sample.dtype == torch.uint8:
        return UINT8
    if float(sample.min()) < -0.05:
        return SIGNED
    if float(sample.max()) > 1.5:
        return SCALED
    return UNIT


def _owned_float(sample):
    """
    A float32 tensor that is safe to modify in place.

    CAREFUL: .float() on a tensor that is ALREADY float32 returns the same
    tensor, not a copy. Scaling that in place would corrupt the caller's
    video.
    """
    out = sample.float()
    return out.clone() if out is sample else out


def chunk_to_float01(sample, kind):
    """
    Convert frames of a known format to float32 in [0, 1].

    Always returns a tensor the caller owns, so the result can be worked on
    in place without reaching back into the video it came from.
    """
    if kind == UINT8:
        # .float() on uint8 always allocates, so the scaling can be done in
        # place instead of allocating a second full-size copy.
        return sample.float().div_(255.0)
    if kind == SIGNED:
        return _owned_float(sample).add_(1.0).div_(2.0)
    if kind == SCALED:
        return _owned_float(sample).div_(255.0)
    return _owned_float(sample)


def from_float01(values01, kind, dtype):
    """
    Convert float [0, 1] frames back to the format they were read in.

    `values01` is consumed: it is assumed to be a tensor the caller owns, as
    everything from chunk_to_float01 is.
    """
    if kind == UINT8:
        return values01.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype)
    if kind == SIGNED:
        return values01.mul_(2.0).sub_(1.0).to(dtype)
    if kind == SCALED:
        return values01.mul_(255.0).to(dtype)
    return values01.to(dtype)


def to_float01(sample):
    """Normalise a decoded window to float32 in [0, 1]."""
    return chunk_to_float01(sample, detect_kind(sample))


def frame_to_rgb_uint8(sample, index=-1, kind=None):
    """
    Pull one frame out of a decoded window as an (H, W, 3) uint8 array.

    The frame is sliced out before conversion, so grabbing one frame does not
    allocate a float copy of the whole window. Pass `kind` when it is already
    known to skip the sniff as well.
    """
    if kind is None:
        kind = detect_kind(sample)
    frame01 = chunk_to_float01(sample[:, index], kind)
    frame = frame01.permute(1, 2, 0).cpu().numpy()
    return np.round(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
