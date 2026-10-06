"""Generate assets/Myna.ico: a mosaic-tile hill myna, perched, facing right.

Built entirely from flat geometric primitives (no external art, no randomness
beyond a fixed seed), so the same source always produces the same icon.

The mosaic effect comes from tiling small inset squares across a grid and only
drawing a tile when its centre falls inside one of the bird's zones (body,
head, tail, beak, eye-patch). The gaps between inset tiles are what create the
grout lines - they are simply the background colour showing through, so no
separate mask or grout-drawing pass is needed. At small sizes the tiles blend
together and the icon reads as a clean flat-colour bird silhouette; at large
sizes the mosaic texture is visible.

Colours: near-black charcoal for the plumage, warm gold for the bare skin
patch behind the eye and the beak/legs (both real hill myna markings), on a
light off-white background - matching the app's theme in ui/main_window.py.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
SEED = 20260910  # fixed, so the icon is reproducible across runs

OFF_WHITE = (248, 246, 241)       # matches PAPER/background in the app theme
CHARCOAL = (30, 27, 24)           # plumage
GOLD = (196, 145, 41)             # bare eye-patch skin + beak base
GOLD_LEG = (176, 122, 28)         # legs, slightly deeper
EYE = (18, 16, 15)

TILE_CELL = 27
# A thin inset reads as mosaic grout at full size; too thin and downsampling
# to 16-32px (where the icon is actually seen most) washes the fill colour
# out to a grey mush, since the light gaps dominate the averaged pixels.
TILE_INSET = 2            # shrink each cell by this many px on every side
JITTER_POS = 1            # random position wobble per tile
JITTER_SIZE = 1           # random size wobble per tile
JITTER_COLOR = 14         # +/- per RGB channel


def _clamp(value: int) -> int:
    return max(0, min(255, value))


def _jittered(rng: random.Random, base: tuple[int, int, int]) -> tuple[int, int, int]:
    delta = rng.randint(-JITTER_COLOR, JITTER_COLOR)
    return tuple(_clamp(channel + delta) for channel in base)


def _in_ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float) -> bool:
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _in_triangle(px: float, py: float, tri: list[tuple[float, float]]) -> bool:
    (x1, y1), (x2, y2), (x3, y3) = tri

    def sign(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    d1 = sign(px, py, x1, y1, x2, y2)
    d2 = sign(px, py, x2, y2, x3, y3)
    d3 = sign(px, py, x3, y3, x1, y1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# Bird geometry: side profile, perched, facing right. All coordinates are for
# a 512x512 canvas.
HEAD = {"cx": 330.0, "cy": 168.0, "rx": 68.0, "ry": 66.0}
BODY = {"cx": 262.0, "cy": 322.0, "rx": 122.0, "ry": 138.0}
TAIL = [(172.0, 418.0), (78.0, 466.0), (196.0, 470.0)]
BEAK = [(388.0, 118.0), (470.0, 168.0), (388.0, 218.0)]
EYE_PATCH = {"cx": 366.0, "cy": 152.0, "rx": 28.0, "ry": 24.0}
WING_PATCH = {"cx": 236.0, "cy": 276.0, "rx": 54.0, "ry": 46.0}
EYE_CENTER = (372.0, 148.0)
EYE_RADIUS = 7.0
LEGS = [
    ((208.0, 452.0), (208.0, 486.0)),
    ((296.0, 452.0), (296.0, 486.0)),
]
LEG_WIDTH = 15


def _zone_for(x: float, y: float) -> str | None:
    """Which part of the bird a point falls in, or None if outside it."""
    if _in_ellipse(x, y, **WING_PATCH):
        return "off_white"  # drawn, not skipped, so it reads as a marking
    if _in_ellipse(x, y, **EYE_PATCH):
        return "gold"
    if _in_triangle(x, y, BEAK):
        return "gold"
    if _in_ellipse(x, y, **HEAD):
        return "charcoal"
    if _in_ellipse(x, y, **BODY):
        return "charcoal"
    if _in_triangle(x, y, TAIL):
        return "charcoal"
    return None


def _rounded_background() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    layer = Image.new("RGBA", (SIZE, SIZE), OFF_WHITE + (255,))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, SIZE - 1, SIZE - 1], radius=int(SIZE * 0.22), fill=255
    )
    image.paste(layer, (0, 0), mask)
    return image


def _draw_mosaic(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    rng = random.Random(SEED)

    columns = SIZE // TILE_CELL + 2
    rows = SIZE // TILE_CELL + 2

    for row in range(rows):
        for col in range(columns):
            cell_x = col * TILE_CELL
            cell_y = row * TILE_CELL
            center_x = cell_x + TILE_CELL / 2
            center_y = cell_y + TILE_CELL / 2

            zone = _zone_for(center_x, center_y)
            if zone is None:
                continue

            base = {"charcoal": CHARCOAL, "gold": GOLD, "off_white": OFF_WHITE}[zone]
            # The wing-patch marking is meant to read as a clean, deliberate
            # spot of colour, not a textured one, so skip its jitter.
            color = base if zone == "off_white" else _jittered(rng, base)

            wobble_x = rng.randint(-JITTER_POS, JITTER_POS)
            wobble_y = rng.randint(-JITTER_POS, JITTER_POS)
            wobble_w = rng.randint(-JITTER_SIZE, JITTER_SIZE)

            x0 = cell_x + TILE_INSET + wobble_x
            y0 = cell_y + TILE_INSET + wobble_y
            x1 = cell_x + TILE_CELL - TILE_INSET + wobble_x + wobble_w
            y1 = cell_y + TILE_CELL - TILE_INSET + wobble_y + wobble_w
            if x1 <= x0 or y1 <= y0:
                continue

            draw.rounded_rectangle([x0, y0, x1, y1], radius=3, fill=color)


def _draw_face_and_legs(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    ex, ey = EYE_CENTER
    draw.ellipse(
        [ex - EYE_RADIUS, ey - EYE_RADIUS, ex + EYE_RADIUS, ey + EYE_RADIUS],
        fill=EYE,
    )
    # A tiny highlight keeps the eye from reading as a flat dead dot.
    hl = EYE_RADIUS * 0.35
    draw.ellipse(
        [ex - hl - 2, ey - hl - 2, ex - 2, ey - 2],
        fill=OFF_WHITE,
    )

    for (x0, y0), (x1, y1) in LEGS:
        draw.line([x0, y0, x1, y1], fill=GOLD_LEG, width=LEG_WIDTH)
        # Rounded caps: PIL's line join is square, so cap each end with a disc.
        for x, y in ((x0, y0), (x1, y1)):
            r = LEG_WIDTH / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=GOLD_LEG)


def build_icon() -> Image.Image:
    image = _rounded_background()
    _draw_mosaic(image)
    _draw_face_and_legs(image)
    return image


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/Myna.ico")
    target.parent.mkdir(parents=True, exist_ok=True)

    image = build_icon()
    image.save(
        target,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"Wrote {target}")

    # A flat PNG alongside is handy for previewing without an .ico viewer, and
    # is not required by the app itself.
    preview = target.with_suffix(".png")
    image.save(preview, format="PNG")
    print(f"Wrote {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
