"""When to reload the visible part of a zoomed-in view at a finer resolution.

Views draw a thinned-out overview of large data. After zooming in, they load
just the visible area (plus a margin, so small pans need no new reads) at the
resolution it can afford, and draw it over the overview.
"""

PAD = 0.25  # margin loaded around the visible area, as a fraction of its size


class DetailTracker:
    """Remembers what area was loaded in detail and says when that's out of date.

    Boxes are (x0, x1, y0, y1) in whatever coordinates the view uses.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.visible = None  # visible box when the detail was loaded
        self.loaded = None  # box that was loaded (visible + margin)

    def up_to_date(self, visible) -> bool:
        """Does the loaded detail still cover `visible` at about the same zoom?"""
        if self.loaded is None:
            return False
        vx0, vx1, vy0, vy1 = _ordered(visible)
        lx0, lx1, ly0, ly1 = self.loaded
        inside = lx0 <= vx0 and vx1 <= lx1 and ly0 <= vy0 and vy1 <= ly1
        ratio = _area(visible) / _area(self.visible)
        return inside and 0.5 <= ratio <= 2  # zooming further may allow a finer step

    def remember(self, visible, loaded):
        self.visible, self.loaded = _ordered(visible), _ordered(loaded)


def padded(box, fraction=PAD):
    x0, x1, y0, y1 = _ordered(box)
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return x0 - dx, x1 + dx, y0 - dy, y1 + dy


def _ordered(box):
    x0, x1, y0, y1 = box
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def _area(box):
    x0, x1, y0, y1 = _ordered(box)
    return max(x1 - x0, 1e-12) * max(y1 - y0, 1e-12)


def resolution_note(base, detail) -> str:
    """Status text for a view showing Slice `base`, maybe with a `detail` Slice on top."""
    if not base.downsampled:
        return ""
    if detail is not None:
        if not detail.downsampled:
            return "  full resolution in view"
        return f"  downsampled ×{detail.max_step} in view (zoom in for more)"
    return f"  downsampled ×{base.max_step} (zoom in for full resolution)"
