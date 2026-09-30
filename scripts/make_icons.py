"""Draw the app icon and write it in the sizes fbs expects (src/main/icons).

    python scripts/make_icons.py
"""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

ICONS = Path(__file__).resolve().parents[1] / "src" / "main" / "icons"
SIZE = 1024
SIZES = {"base": [16, 24, 32, 48, 64], "linux": [128, 256, 512, 1024], "mac": [128, 256, 512, 1024]}
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def draw(size=SIZE):
    """A globe filled with a colour-mapped field on a rounded dark tile."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=size * 0.18, fill=(22, 50, 79, 255))

    c, r = size / 2, size * 0.34
    y, x = np.mgrid[0:size, 0:size]
    u, v = (x - c) / r, (y - c) / r
    inside = u ** 2 + v ** 2 <= 1
    field = np.sin(3.2 * u + 1.0) * np.cos(2.4 * v - 0.4) + 0.6 * (1 - v)
    field = (field - field[inside].min()) / np.ptp(field[inside])
    rgba = (matplotlib.colormaps["viridis"](field) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0)
    img.alpha_composite(Image.fromarray(rgba, "RGBA"))

    line = max(2, round(size * 0.018))
    white = (255, 255, 255, 235)
    d = ImageDraw.Draw(img)
    d.ellipse((c - r, c - r, c + r, c + r), outline=white, width=line)
    d.ellipse((c - r * 0.45, c - r, c + r * 0.45, c + r), outline=white, width=max(1, line // 2))
    d.line((c - r, c, c + r, c), fill=white, width=max(1, line // 2))
    return img


def main():
    full = draw()
    # macOS icons carry a transparent margin, otherwise they look too big in the Dock.
    mac = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    inner = round(SIZE * 0.8)
    mac.paste(full.resize((inner, inner), Image.Resampling.LANCZOS), ((SIZE - inner) // 2,) * 2)
    for folder, sizes in SIZES.items():
        (ICONS / folder).mkdir(parents=True, exist_ok=True)
        source = mac if folder == "mac" else full
        for s in sizes:
            source.resize((s, s), Image.Resampling.LANCZOS).save(ICONS / folder / f"{s}.png")
    full.save(ICONS / "Icon.ico", sizes=[(s, s) for s in ICO_SIZES])
    print(f"Icons written to {ICONS}")


if __name__ == "__main__":
    main()
