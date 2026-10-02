"""Run-length encoding for element masks (the contract's `mask_rle`)."""
from __future__ import annotations

import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    """Row-major run lengths, alternating, starting with a run of zeros (possibly 0 long): "12,5,300,..."."""
    flat = np.asarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return ""
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return ",".join(map(str, runs))


def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    runs = [int(r) for r in rle.split(",")] if rle else []
    if sum(runs) != shape[0] * shape[1]:
        raise ValueError("RLE does not cover the given shape")
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    pos, val = 0, False
    for r in runs:
        if val:
            out[pos : pos + r] = True
        pos += r
        val = not val
    return out.reshape(shape)
