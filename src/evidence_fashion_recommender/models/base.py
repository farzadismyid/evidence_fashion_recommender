"""Interfaces shared by model providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Generator(Protocol):
    model_id: str

    def generate(self, prompt: str) -> str: ...
