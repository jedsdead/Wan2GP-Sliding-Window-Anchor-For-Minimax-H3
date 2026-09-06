"""
Frame conversion helpers.

Pulls a single frame out of a decoded window as an image the plugin can save
and feed back in on the next window.
"""
import numpy as np
import torch


def to_float01(sample):
    """
    Normalise a decoded window to float32 in [0, 1].

    WanGP hands back either uint8 (0..255) or float. Float is normally in
    [-1, 1] but the range is sniffed rather than assumed, because a wrong
    guess here silently destroys the image.
    """
    if sample.dtype == torch.uint8:
        # .float() on a uint8 tensor always allocates, so the scaling can be
        # done in place instead of allocating a second full-size copy of the
        # window.
        return sample.float().div_(255.0)

    lo = float(sample.min())
    hi = float(sample.max())

    if lo < -0.05:  # signed, treat as [-1, 1]
        # CAREFUL: .float() on a tensor that is ALREADY float32 returns the
        # same tensor, not a copy. Scaling that in place would corrupt the
        # caller's video. Let the first op allocate when that happens.
        out = sample.float()
        out = out.add_(1.0) if out is not sample else out.add(1.0)
        return out.div_(2.0)

    if hi > 1.5:  # float but 0..255
        out = sample.float()
        return out.div_(255.0) if out is not sample else out.div(255.0)

    return sample.float()


def frame_to_rgb_uint8(sample, index=-1):
    """Pull one frame out of a decoded window as an (H, W, 3) uint8 array."""
    video01 = to_float01(sample)
    frame = video01[:, index].permute(1, 2, 0).cpu().numpy()
    return np.round(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
