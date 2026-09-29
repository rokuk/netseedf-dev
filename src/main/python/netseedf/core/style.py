"""Colour scaling shared by the heatmap and both maps."""

from dataclasses import dataclass

import numpy as np

COLORMAPS = ["viridis", "inferno", "magma", "plasma", "cividis", "turbo", "coolwarm", "RdBu_r", "BrBG",
             "Blues", "YlGnBu", "Greys", "terrain", "twilight"]


@dataclass
class Style:
    cmap: str = "viridis"
    auto: bool = True  # the range is the data's min–max, else vmin..vmax
    vmin: float = 0.0
    vmax: float = 1.0

    def limits(self, values: np.ndarray) -> tuple[float, float]:
        if self.auto:
            return data_limits(values)
        return _widen(self.vmin, self.vmax)


def data_limits(values) -> tuple[float, float]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return 0.0, 1.0
    return _widen(float(finite.min()), float(finite.max()))


def _widen(lo, hi):
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        pad = abs(lo) * 0.01 or 0.5
        lo, hi = lo - pad, hi + pad
    return lo, hi
