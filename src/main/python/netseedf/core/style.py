"""Colour scaling shared by the heatmap and both maps."""

from dataclasses import dataclass

import numpy as np

COLORMAPS = ["viridis", "inferno", "magma", "plasma", "cividis", "turbo", "coolwarm", "RdBu_r", "BrBG",
             "Blues", "YlGnBu", "Greys", "terrain", "twilight"]

RANGE_AUTO = "Min–max"
RANGE_ROBUST = "Robust (2–98 %)"
RANGE_MANUAL = "Manual"
RANGE_MODES = [RANGE_AUTO, RANGE_ROBUST, RANGE_MANUAL]


@dataclass
class Style:
    cmap: str = "viridis"
    range_mode: str = RANGE_AUTO
    vmin: float = 0.0  # used in manual mode
    vmax: float = 1.0

    def limits(self, values: np.ndarray) -> tuple[float, float]:
        if self.range_mode == RANGE_MANUAL:
            return _widen(self.vmin, self.vmax)
        return data_limits(values, robust=self.range_mode == RANGE_ROBUST)


def data_limits(values, robust=False) -> tuple[float, float]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return 0.0, 1.0
    if robust:
        lo, hi = np.percentile(finite, [2, 98])
    else:
        lo, hi = finite.min(), finite.max()
    return _widen(float(lo), float(hi))


def _widen(lo, hi):
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        pad = abs(lo) * 0.01 or 0.5
        lo, hi = lo - pad, hi + pad
    return lo, hi
