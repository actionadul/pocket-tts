"""Conditioner base types."""

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    tokens: np.ndarray  # int64 tensor of shape (B, T).
