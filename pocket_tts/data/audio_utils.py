"""Sample rate / channel conversion utilities (numpy)."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


def convert_audio(
    wav: np.ndarray, from_rate: int | float, to_rate: int | float, to_channels: int
) -> np.ndarray:
    """Convert audio to a new sample rate (and assert channel count)."""
    if from_rate != to_rate:
        gcd = math.gcd(int(from_rate), int(to_rate))
        up = int(to_rate // gcd)
        down = int(from_rate // gcd)
        wav = resample_poly(wav, up, down, axis=-1).astype(wav.dtype, copy=False)

    assert wav.shape[-2] == to_channels
    return wav
