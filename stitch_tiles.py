# flake8: noqa: E501
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageColor


TILE_FILENAME_RE = re.compile(r"^(-?\d+)_(-?\d+)\.png$")


def discover_tiles(folder: Path) -> Dict[Tuple[int, int], Path]:
    """Scan a folder for tile images named like `x_y.png`.

    Returns a mapping from (x, y) -> file path.
    """
    tiles: Dict[Tuple[int, int], Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = TILE_FILENAME_RE.match(path.name)
        if not match:
            continue
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        tiles[(x, y)] = path
    return tiles


def compute_bounding_box(
    coords: Dict[Tuple[int, int], Path]
) -> Tuple[int, int, int, int]:
    xs = [x for (x, _y) in coords.keys()]
    ys = [y for (_x, y) in coords.keys()]
    return min(xs), min(ys), max(xs), max(ys)


def stitch_folder(
    folder: Path,
    output_path: Path,
    *,
    tile_size: int = 1000,
    background_rgba: Tuple[int, int, int, int] | None = None,
) -> None:
    tiles = discover_tiles(folder)
    if not tiles:
        raise SystemExit(
            f"No tiles found in {folder}. "
            f"Expected files like '602_763.png'."
        )

    min_x, min_y, max_x, max_y = compute_bounding_box(tiles)

    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    width = cols * tile_size
    height = rows * tile_size

    print(
        f"Stitching {len(tiles)} tiles from {folder} into a "
        f"{cols}x{rows} grid ({width}x{height} px) "
        f"with tile size {tile_size}."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    # Place tiles: x increases to the right, y increases downward
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            dst_x = (x - min_x) * tile_size
            dst_y = (y - min_y) * tile_size

            tile_path = tiles.get((x, y))
            if tile_path is None:
                # No tile file: leave transparent block
                continue

            try:
                img = Image.open(tile_path).convert("RGBA")
            except Exception as exc:
                print(
                    f"Warning: failed to open {tile_path}: {exc}. "
                    f"Using transparency."
                )
                continue

            if img.size != (tile_size, tile_size):
                # Resize to expected tile size to keep alignment consistent
                img = img.resize((tile_size, tile_size), Image.BICUBIC)

            canvas.paste(img, (dst_x, dst_y))

    # Optionally place the stitched image on a solid background color
    final_image = canvas
    if background_rgba is not None:
        bg_img = Image.new("RGBA", (width, height), background_rgba)
        final_image = Image.alpha_composite(bg_img, canvas)

    final_image.save(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch tiles named 'x_y.png' in a folder into one PNG. Missing "
            "tiles are filled with transparency."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help=(
            "Path to the folder containing tile PNGs "
            "(e.g., tiles/20250810_162535)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to <folder>/stitched.png"
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1000,
        help="Tile size in pixels (default: 1000)",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default=None,
        help=(
            "Background color, e.g., '#000000' or 'white'. If omitted, the "
            "output remains transparent where tiles are missing."
        ),
    )

    args = parser.parse_args()
    folder: Path = args.folder
    output: Path | None = args.output
    tile_size: int = args.tile_size
    background: str | None = args.background

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(
            f"Folder does not exist or is not a directory: {folder}"
        )

    if output is None:
        output = folder / "stitched.png"

    background_rgba: Tuple[int, int, int, int] | None = None
    if background is not None:
        try:
            # Parse using Pillow's color parser to support names and hex
            color = ImageColor.getcolor(background, "RGBA")
            # Ensure 4-tuple
            if isinstance(color, tuple) and len(color) == 3:
                r, g, b = color
                background_rgba = (r, g, b, 255)
            else:
                background_rgba = color  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - arg validation
            raise SystemExit(
                f"Invalid background color '{background}': {exc}"
            )

    stitch_folder(
        folder,
        output,
        tile_size=tile_size,
        background_rgba=background_rgba,
    )


if __name__ == "__main__":
    main()
