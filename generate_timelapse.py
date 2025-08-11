# flake8: noqa: E501
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Local imports
import stitch_tiles as st
from PIL import Image


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def discover_timestamp_dirs(tiles_root: Path) -> List[Path]:
    """Return sorted subdirectories of tiles_root that look like capture folders.

    We include any subdir that contains any tile file matching stitcher's pattern
    or an existing stitched.png.
    """
    if not tiles_root.exists() or not tiles_root.is_dir():
        return []
    candidates: List[Path] = []
    for child in tiles_root.iterdir():
        if not child.is_dir():
            continue
        # Is there already a stitched image?
        if (child / "stitched.png").exists():
            candidates.append(child)
            continue
        # Or at least one tile file?
        try:
            for p in child.iterdir():
                if p.is_file() and st.TILE_FILENAME_RE.match(p.name):
                    candidates.append(child)
                    break
        except Exception:
            # Ignore unreadable folders
            continue
    # Sort by folder name (timestamps are lexicographically sortable)
    return sorted(candidates, key=lambda p: p.name)


def stitch_missing(
    capture_dirs: Sequence[Path],
    *,
    tile_size: int,
    background: Optional[Tuple[int, int, int, int]],
    force: bool,
    global_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> List[Path]:
    """Stitch tiles in each capture dir, returning paths to stitched images."""
    stitched_paths: List[Path] = []
    for folder in capture_dirs:
        output_path = folder / "stitched.png"
        if output_path.exists() and not force:
            print(f"Found existing stitched image, keeping: {output_path}")
            stitched_paths.append(output_path)
            continue

        try:
            tiles = st.discover_tiles(folder)
        except Exception as exc:
            print(f"Warning: failed to scan {folder}: {exc}")
            continue

        if not tiles:
            print(f"Skipping {folder}: no tiles found")
            continue

        print(f"Stitching tiles in {folder} -> {output_path}")
        try:
            st.stitch_folder(
                folder,
                output_path,
                tile_size=tile_size,
                background_rgba=background,
                global_bbox=global_bbox,
            )
            stitched_paths.append(output_path)
        except SystemExit as exc:
            print(f"Warning: stitch error in {folder}: {exc}")
        except Exception as exc:
            print(f"Warning: unexpected stitch error in {folder}: {exc}")
    return stitched_paths


def parse_bbox_arg(bbox: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not bbox:
        return None
    parts = bbox.split(",")
    if len(parts) != 4:
        raise SystemExit("--bbox must be minX,minY,maxX,maxY")
    try:
        min_x, min_y, max_x, max_y = map(int, parts)
    except Exception:
        raise SystemExit("--bbox values must be integers")
    if min_x > max_x or min_y > max_y:
        raise SystemExit("--bbox must have min <= max for both axes")
    return (min_x, min_y, max_x, max_y)


def restrict_dirs_by_ts(dirs: Sequence[Path], min_ts: Optional[str], max_ts: Optional[str]) -> List[Path]:
    def ok(name: str) -> bool:
        if min_ts and name < min_ts:
            return False
        if max_ts and name > max_ts:
            return False
        return True
    return [d for d in dirs if ok(d.name)]


def bbox_from_center(center_x: Optional[int], center_y: Optional[int], buffer_size: Optional[int]) -> Optional[Tuple[int, int, int, int]]:
    if center_x is None or center_y is None or buffer_size is None:
        return None
    b = int(buffer_size)
    cx = int(center_x)
    cy = int(center_y)
    return (cx - b, cy - b, cx + b, cy + b)


def detect_tile_size_from_any(capture_dirs: Sequence[Path]) -> Optional[int]:
    """Inspect one tile image to infer natural tile size.

    Returns the width (assuming square tiles). If detection fails, returns None.
    """
    for folder in capture_dirs:
        try:
            tiles = st.discover_tiles(folder)
        except Exception:
            continue
        if not tiles:
            continue
        for _coord, path in tiles.items():
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    if w <= 0 or h <= 0:
                        continue
                    return int(w)
            except Exception:
                continue
    return None


def copy_frames_in_order(
    stitched_images: Sequence[Path],
    frames_dir: Path,
    *,
    stage_scale_width: Optional[int] = None,
    progress_interval: int = 50,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(stitched_images)
    for idx, img_path in enumerate(stitched_images, start=1):
        dst = frames_dir / f"frame_{idx:06d}.png"
        try:
            if stage_scale_width and stage_scale_width > 0:
                # Only re-encode when we actually scale to reduce work and avoid slow PNG optimize passes
                with Image.open(img_path) as im:
                    im = im.convert("RGBA")
                    w, h = im.size
                    if w > stage_scale_width:
                        new_h = int(round(h * (stage_scale_width / float(w))))
                        im = im.resize((stage_scale_width, max(1, new_h)), Image.LANCZOS)
                    # Fast save params: avoid optimize (very slow); use low compression level
                    im.save(dst, format="PNG", compress_level=1)
            else:
                # No scaling requested: hard copy for speed
                shutil.copy2(img_path, dst)
        except Exception:
            # Ensure no half-written file remains locked on Windows
            try:
                if dst.exists():
                    dst.unlink(missing_ok=True)
            except Exception:
                pass
            # Fallback to copy
            shutil.copy2(img_path, dst)

        if progress_interval > 0 and (idx % progress_interval == 0 or idx == total):
            print(f"Staged {idx}/{total} frames")


def _build_ffmpeg_input_args(fr_string: str, frames_dir: Path) -> list[str]:
    # Use numeric pattern for better cross-platform compatibility
    return [
        "-framerate",
        fr_string,
        "-i",
        str(frames_dir / "frame_%06d.png"),
    ]


def _maybe_scale_filter(scale_width: Optional[int]) -> list[str]:
    """Return a -vf chain for MP4 output.

    - If user provides --scale-width, use it.
    - Else, clamp to a safe maximum width (MP4_MAX_WIDTH, default 3840).
    - Always enforce even dimensions for yuv420p/H.264.
    """
    vf_chain: list[str] = []
    if scale_width and scale_width > 0:
        vf_chain.append(f"scale={scale_width}:-2:flags=lanczos")
    else:
        mp4_max_width = _read_env_int("MP4_MAX_WIDTH", 3840)
        # Downscale only when input width exceeds the safe maximum
        vf_chain.append(f"scale=min(iw\,{mp4_max_width}):-2:flags=lanczos")
    # Ensure even dimensions regardless of previous step
    vf_chain.append("scale=ceil(iw/2)*2:ceil(ih/2)*2")
    return ["-vf", ",".join(vf_chain)]


def run_ffmpeg_mp4(
    *,
    ffmpeg_bin: str,
    frames_dir: Path,
    framerate_str: str,
    output_path: Path,
    scale_width: Optional[int],
    video_codec: str,
    preset: Optional[str],
    x264_params: Optional[str],
    crf: int,
    pix_fmt: str,
    threads: Optional[int] = None,
    loglevel: str = "info",
    show_stats: bool = True,
    progress: Optional[str] = None,
) -> None:
    args = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        loglevel,
        *( ["-stats"] if show_stats else [] ),
        *( ["-progress", "pipe:1"] if (progress == "pipe") else (["-progress", progress] if progress else []) ),
    ]
    if threads and threads > 0:
        args += ["-threads", str(threads)]
    args += [
        *_build_ffmpeg_input_args(framerate_str, frames_dir),
        *_maybe_scale_filter(scale_width),
        "-c:v",
        video_codec,
        *( ["-preset", preset] if preset else []),
        *( ["-x264-params", x264_params] if x264_params and video_codec in ("libx264", "h264") else []),
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
    ]

    # Strengthen H.264 compatibility for common players (Windows, Discord)
    h264_like_codecs = ("libx264", "h264", "h264_nvenc", "h264_qsv", "h264_amf")
    if any(c in video_codec for c in h264_like_codecs):
        # Permit overrides via environment
        h264_profile = os.environ.get("H264_PROFILE", "high")
        h264_level = os.environ.get("H264_LEVEL", "4.1")
        args += [
            "-profile:v", h264_profile,
            "-level:v", h264_level,
            # Force MP4 fourcc to avc1 for broader compatibility
            "-tag:v", "avc1",
        ]

    # Set explicit output frame rate and use constant frame rate pacing
    args += [
        "-r", framerate_str,
        "-vsync", "cfr",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(args, check=True)


def run_ffmpeg_gif(
    *,
    ffmpeg_bin: str,
    frames_dir: Path,
    framerate_str: str,
    output_path: Path,
    scale_width: Optional[int],
    threads: Optional[int] = None,
    loglevel: str = "info",
    show_stats: bool = True,
    progress: Optional[str] = None,
    force_size: Optional[Tuple[int, int]] = None,
    pad_color: str = "black",
) -> None:
    # Two-pass GIF: generate palette, then apply palette. This reduces memory vs single-pass split.
    palette_path = frames_dir / "__palette.png"
    palettegen_opts = f"stats_mode={os.environ.get('GIF_STATS_MODE', 'full')}:reserve_transparent=1"
    dither = os.environ.get('GIF_DITHER', 'sierra2_4a')
    diff_mode = os.environ.get('GIF_DIFF_MODE', 'rectangle')
    paletteuse_opts = f"dither={dither}:diff_mode={diff_mode}:new=1"

    # Pass 1: palette generation
    pass1 = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        loglevel,
        *( ["-stats"] if show_stats else [] ),
        *( ["-progress", "pipe:1"] if (progress == "pipe") else (["-progress", progress] if progress else []) ),
    ]
    if threads and threads > 0:
        pass1 += ["-threads", str(threads)]
    pass1 += [
        *_build_ffmpeg_input_args(framerate_str, frames_dir),
    ]
    vf1 = []
    if scale_width and scale_width > 0:
        vf1.append(f"scale={scale_width}:-1:flags=lanczos")
    if force_size and force_size[0] > 0 and force_size[1] > 0:
        w, h = force_size
        # Pad to exact canvas; keep centered
        vf1.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color}")
    vf1.append(f"palettegen={palettegen_opts}")
    pass1 += ["-vf", ",".join(vf1)]
    pass1 += [str(palette_path)]

    # Pass 2: apply palette
    pass2 = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        loglevel,
        *( ["-stats"] if show_stats else [] ),
        *( ["-progress", "pipe:1"] if (progress == "pipe") else (["-progress", progress] if progress else []) ),
    ]
    if threads and threads > 0:
        pass2 += ["-threads", str(threads)]
    pass2 += [
        "-framerate",
        framerate_str,
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-i",
        str(palette_path),
    ]
    if scale_width and scale_width > 0:
        scale_chain = f"[0:v]scale={scale_width}:-1:flags=lanczos[s]"
        src_label = "[s]"
    else:
        scale_chain = ""
        src_label = "[0:v]"

    pad_chain = ""
    if force_size and force_size[0] > 0 and force_size[1] > 0:
        w, h = force_size
        # Build chain: (optional scale) -> (optional pad) -> paletteuse
        if scale_chain:
            pad_chain = f";{src_label}pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color}[padded]"
            src_label = "[padded]"
        else:
            pad_chain = f"{src_label}pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color}[padded]"
            src_label = "[padded]"

    lavfi = "".join(filter(None, [scale_chain, pad_chain, f";{src_label}[1:v]paletteuse={paletteuse_opts}"]))
    if not lavfi.startswith("["):
        # If no scale, ensure we start with the source label
        lavfi = f"{src_label}[1:v]paletteuse={paletteuse_opts}"
    pass2 += ["-lavfi", lavfi]
    pass2 += [str(output_path)]

    try:
        subprocess.run(pass1, check=True)
        subprocess.run(pass2, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "GIF render failed. This often indicates your frames are very large and ffmpeg ran out of memory. "
            "Try reducing size via --gif-scale-width (e.g., 800) or lowering the number of frames.\n"
            f"ffmpeg exit: {exc}"
        )


def compute_global_bbox(capture_dirs: Sequence[Path]) -> Optional[Tuple[int, int, int, int]]:
    """Compute a global bounding box across all capture folders (union)."""
    have_any = False
    min_x = min_y = 0
    max_x = max_y = 0
    for folder in capture_dirs:
        try:
            tiles = st.discover_tiles(folder)
        except Exception:
            continue
        if not tiles:
            continue
        bx = st.compute_bounding_box(tiles)
        if not have_any:
            min_x, min_y, max_x, max_y = bx
            have_any = True
        else:
            min_x = min(min_x, bx[0])
            min_y = min(min_y, bx[1])
            max_x = max(max_x, bx[2])
            max_y = max(max_y, bx[3])
    if not have_any:
        return None
    return (min_x, min_y, max_x, max_y)


def compute_intersection_bbox(capture_dirs: Sequence[Path]) -> Optional[Tuple[int, int, int, int]]:
    """Compute intersection of available tiles across all captures (may be smaller)."""
    first = True
    min_x = min_y = max_x = max_y = 0
    for folder in capture_dirs:
        try:
            tiles = st.discover_tiles(folder)
        except Exception:
            continue
        if not tiles:
            continue
        bx = st.compute_bounding_box(tiles)
        if first:
            min_x, min_y, max_x, max_y = bx
            first = False
        else:
            min_x = max(min_x, bx[0])
            min_y = max(min_y, bx[1])
            max_x = min(max_x, bx[2])
            max_y = min(max_y, bx[3])
            if min_x > max_x or min_y > max_y:
                return None
    if first:
        return None
    return (min_x, min_y, max_x, max_y)


def compute_reference_bbox(capture_dirs: Sequence[Path], reference_name: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not reference_name:
        return None
    for folder in capture_dirs:
        if folder.name == reference_name:
            try:
                tiles = st.discover_tiles(folder)
                if not tiles:
                    return None
                return st.compute_bounding_box(tiles)
            except Exception:
                return None
    return None


def parse_background(color_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not color_str:
        return None
    from PIL import ImageColor

    color = ImageColor.getcolor(color_str, "RGBA")
    if isinstance(color, tuple) and len(color) == 3:
        r, g, b = color
        return (r, g, b, 255)
    return color  # type: ignore[return-value]


def validate_ffmpeg(ffmpeg_bin: str) -> None:
    try:
        subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as exc:
        raise SystemExit(
            "ffmpeg not found. Install ffmpeg and/or set FFMPEG_BINARY to its path. "
            f"Tried '{ffmpeg_bin}': {exc}"
        )


def compute_framerate_string(env_fps: Optional[str], env_frame_duration: Optional[str], cli_fps: Optional[float], cli_frame_duration: Optional[float]) -> str:
    # Priority: CLI frame-duration -> CLI framerate -> ENV frame-duration -> ENV framerate -> default
    if cli_frame_duration and cli_frame_duration > 0:
        return f"1/{cli_frame_duration}"
    if cli_fps and cli_fps > 0:
        return str(cli_fps)
    if env_frame_duration:
        try:
            dur = float(env_frame_duration)
            if dur > 0:
                return f"1/{dur}"
        except Exception:
            pass
    if env_fps:
        try:
            fps = float(env_fps)
            if fps > 0:
                return str(fps)
        except Exception:
            pass
    return "2"  # default 2 fps


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch tiles from timestamped folders and render a timelapse using ffmpeg. "
            "Frames are ordered by folder name and displayed at a configurable rate."
        )
    )
    parser.add_argument("--tiles-root", type=Path, default=Path(os.environ.get("TILES_ROOT", "tiles")), help="Root folder containing timestamped subfolders with tiles (default: tiles)")
    parser.add_argument("--format", choices=["mp4", "gif", "both"], default=os.environ.get("OUTPUT_FORMAT", "mp4"), help="Output format to render (default: mp4)")
    parser.add_argument("--output", type=Path, default=None, help="Output file path. Defaults to tiles/timelapse.<ext>")
    parser.add_argument("--framerate", type=float, default=None, help="Frames per second. Alternative: set FRAME_DURATION_SECONDS to control seconds per frame")
    parser.add_argument("--frame-duration", type=float, default=None, help="Seconds per frame (overrides --framerate if provided)")
    parser.add_argument("--scale-width", type=int, default=_read_env_int("SCALE_WIDTH", 0), help="Optional output width to scale to (height auto-calculated) for MP4 (and GIF if --gif-scale-width not set)")
    parser.add_argument("--gif-scale-width", type=int, default=_read_env_int("GIF_SCALE_WIDTH", 0), help="Optional GIF output width. If not set, GIF will auto-scale to a safe default to avoid OOM")
    parser.add_argument("--gif-width", type=int, default=_read_env_int("GIF_WIDTH", 0), help="Force GIF canvas width (used with --gif-height to pad to exact size)")
    parser.add_argument("--gif-height", type=int, default=_read_env_int("GIF_HEIGHT", 0), help="Force GIF canvas height (used with --gif-width to pad to exact size)")
    parser.add_argument("--gif-pad-color", type=str, default=os.environ.get("GIF_PAD_COLOR", "black"), help="Pad color for GIF when forcing size (e.g., black, white, #000000)")
    parser.add_argument("--gif-dither", type=str, choices=["none", "bayer", "floyd_steinberg", "sierra2", "sierra2_4a"], default=os.environ.get("GIF_DITHER", "sierra2_4a"), help="Dithering algorithm for paletteuse (default: sierra2_4a)")
    parser.add_argument("--gif-stats-mode", type=str, choices=["full", "single"], default=os.environ.get("GIF_STATS_MODE", "full"), help="Palettegen stats mode (full or single). 'single' reduces memory")
    parser.add_argument("--gif-diff-mode", type=str, choices=["none", "rectangle"], default=os.environ.get("GIF_DIFF_MODE", "rectangle"), help="paletteuse diff_mode (default: rectangle)")
    parser.add_argument("--force-stitch", action="store_true", default=_bool_from_env(os.environ.get("FORCE_STITCH"), default=False), help="Re-stitch even if stitched.png exists")
    parser.add_argument("--tile-size", type=int, default=_read_env_int("TILE_SIZE", 1000), help="Expected tile size in pixels (default: 1000)")
    parser.add_argument("--background", type=str, default=os.environ.get("STITCH_BACKGROUND", None), help="Optional background color for stitching (e.g., white or #000000)")
    parser.add_argument("--ffmpeg-binary", type=str, default=os.environ.get("FFMPEG_BINARY", "ffmpeg"), help="ffmpeg binary name or full path (default: ffmpeg)")
    parser.add_argument("--ffmpeg-loglevel", type=str, default=os.environ.get("FFMPEG_LOGLEVEL", "info"), help="ffmpeg -loglevel (quiet, panic, fatal, error, warning, info, verbose, debug, trace)")
    parser.add_argument("--ffmpeg-stats", action="store_true", default=_bool_from_env(os.environ.get("FFMPEG_STATS"), default=True), help="Show ffmpeg -stats progress output")
    parser.add_argument("--ffmpeg-progress", type=str, default=os.environ.get("FFMPEG_PROGRESS", None), help="ffmpeg -progress target: 'pipe' for stdout or a file path")
    parser.add_argument("--video-codec", type=str, default=os.environ.get("VIDEO_CODEC", "libx264"), help="Video codec for mp4 (default: libx264)")
    parser.add_argument("--preset", type=str, default=os.environ.get("PRESET", "veryfast"), help="Encoder preset for mp4 (x264), e.g., ultrafast, veryfast, medium")
    parser.add_argument("--x264-params", type=str, default=os.environ.get("X264_PARAMS", None), help="Extra x264 params, e.g., 'ref=1:bframes=0' to lower memory")
    parser.add_argument("--crf", type=int, default=_read_env_int("CRF", 20), help="CRF quality for mp4 (lower is higher quality, default: 20)")
    parser.add_argument("--pix-fmt", type=str, default=os.environ.get("PIX_FMT", "yuv420p"), help="Pixel format for mp4 (default: yuv420p)")
    parser.add_argument("--threads", type=int, default=_read_env_int("THREADS", 0), help="Threads to use for ffmpeg (0 = auto)")
    parser.add_argument("--min-ts", type=str, default=os.environ.get("MIN_TIMESTAMP", None), help="Include only capture folders with name >= this (lexicographic)")
    parser.add_argument("--max-ts", type=str, default=os.environ.get("MAX_TIMESTAMP", None), help="Include only capture folders with name <= this (lexicographic)")
    parser.add_argument("--bbox", type=str, default=os.environ.get("BBOX", None), help="Explicit tile bbox as minX,minY,maxX,maxY (overrides automatic)")
    parser.add_argument("--center-x", type=int, default=os.environ.get("CENTER_X", None), help="Center tile X for bbox (used with --center-y and --buffer-size)")
    parser.add_argument("--center-y", type=int, default=os.environ.get("CENTER_Y", None), help="Center tile Y for bbox (used with --center-x and --buffer-size)")
    parser.add_argument("--buffer-size", type=int, default=os.environ.get("BUFFER_SIZE", None), help="Buffer radius for bbox (produces (2b+1)x(2b+1) tiles)")
    parser.add_argument("--bbox-mode", choices=["union", "intersection", "reference"], default=os.environ.get("BBOX_MODE", "union"), help="How to compute bbox across timestamps when not explicitly provided")
    parser.add_argument("--reference-ts", type=str, default=os.environ.get("REFERENCE_TIMESTAMP", None), help="Timestamp folder name to use as reference when --bbox-mode=reference")
    parser.add_argument("--max-total-pixels", type=int, default=_read_env_int("MAX_TOTAL_PIXELS", 80_000_000), help="Safety limit: abort if (width*height) exceeds this (default ~80MP)")
    parser.add_argument("--limit-frames", type=int, default=_read_env_int("LIMIT_FRAMES", 0), help="Use only the first N frames (0 = all). Useful to test settings quickly")
    parser.add_argument("--auto-downscale", action="store_true", default=_bool_from_env(os.environ.get("AUTO_DOWNSCALE"), default=True), help="Automatically reduce tile size to fit --max-total-pixels")
    parser.add_argument("--stage-scale-width", type=int, default=_read_env_int("STAGE_SCALE_WIDTH", 0), help="Scale frames during staging before ffmpeg (reduces ffmpeg memory). 0 = disabled")

    args = parser.parse_args(argv)

    tiles_root: Path = args.tiles_root
    out_format: str = args.format
    output_path: Optional[Path] = args.output
    framerate_str = compute_framerate_string(
        os.environ.get("FRAMERATE"),
        os.environ.get("FRAME_DURATION_SECONDS"),
        args.framerate,
        args.frame_duration,
    )
    # Compute per-format scale widths. GIF defaults to a safe downscale if no width provided
    scale_width: Optional[int] = args.scale_width if args.scale_width and args.scale_width > 0 else None
    gif_scale_width_env = _read_env_int("GIF_SCALE_WIDTH", 0)
    gif_scale_width_cli = args.gif_scale_width if args.gif_scale_width and args.gif_scale_width > 0 else None
    gif_scale_width: Optional[int]
    if gif_scale_width_cli:
        gif_scale_width = gif_scale_width_cli
    elif gif_scale_width_env > 0:
        gif_scale_width = gif_scale_width_env
    else:
        gif_scale_width = 1000  # sensible default to avoid OOM on large canvases
    force_stitch: bool = args.force_stitch
    # If tile size wasn't provided explicitly, try to auto-detect
    tile_size: int = args.tile_size if args.tile_size else (detect_tile_size_from_any(capture_dirs) or 1000)
    background_rgba = parse_background(args.background)
    ffmpeg_bin: str = args.ffmpeg_binary
    ffmpeg_loglevel: str = args.ffmpeg_loglevel
    ffmpeg_show_stats: bool = args.ffmpeg_stats
    ffmpeg_progress: Optional[str] = args.ffmpeg_progress
    video_codec: str = args.video_codec
    preset: Optional[str] = args.preset
    x264_params: Optional[str] = args.x264_params
    crf: int = args.crf
    pix_fmt: str = args.pix_fmt
    threads: Optional[int] = args.threads if args.threads and args.threads > 0 else None
    gif_width: Optional[int] = args.gif_width if args.gif_width and args.gif_width > 0 else None
    gif_height: Optional[int] = args.gif_height if args.gif_height and args.gif_height > 0 else None
    gif_pad_color: str = args.gif_pad_color
    gif_dither: str = args.gif_dither
    gif_stats_mode: str = args.gif_stats_mode
    gif_diff_mode: str = args.gif_diff_mode

    validate_ffmpeg(ffmpeg_bin)

    capture_dirs = discover_timestamp_dirs(tiles_root)
    # Limit by timestamp range if provided
    if args.min_ts or args.max_ts:
        capture_dirs = restrict_dirs_by_ts(capture_dirs, args.min_ts, args.max_ts)
    if not capture_dirs:
        raise SystemExit(f"No capture folders found under {tiles_root}. Run main.py first to download tiles.")

    # Determine bbox, allow overrides
    global_bbox = parse_bbox_arg(args.bbox)
    if global_bbox is None:
        center_bbox = bbox_from_center(args.center_x, args.center_y, args.buffer_size)
    else:
        center_bbox = None
    if global_bbox is None and center_bbox is not None:
        global_bbox = center_bbox
    if global_bbox is None:
        if args.bbox_mode == "reference":
            global_bbox = compute_reference_bbox(capture_dirs, args.reference_ts)
        elif args.bbox_mode == "intersection":
            global_bbox = compute_intersection_bbox(capture_dirs)
        else:
            global_bbox = compute_global_bbox(capture_dirs)
    if global_bbox is None:
        raise SystemExit("No tiles found in any capture folders.")

    stitched_paths = stitch_missing(
        capture_dirs,
        tile_size=tile_size,
        background=background_rgba,
        force=force_stitch,
        global_bbox=global_bbox,
    )

    # Keep only those that actually exist
    stitched_paths = [p for p in stitched_paths if p.exists()]
    # Optionally limit frames to reduce memory/CPU load
    if args.limit_frames and args.limit_frames > 0:
        stitched_paths = stitched_paths[: args.limit_frames]
    if not stitched_paths:
        raise SystemExit("No stitched images produced.")

    # Prepare frames in a temp directory and render
    default_output = (tiles_root / f"timelapse.{out_format if out_format != 'both' else 'mp4'}")
    if output_path is None:
        output_path = default_output

    # If 'both', derive mp4/gif paths
    output_paths: List[Tuple[str, Path]] = []
    if out_format == "both":
        output_paths.append(("mp4", tiles_root / "timelapse.mp4"))
        output_paths.append(("gif", tiles_root / "timelapse.gif"))
    else:
        output_paths.append((out_format, output_path))

    # Safety: compute pixel dimensions and bail out or auto-downscale if excessive
    min_x, min_y, max_x, max_y = global_bbox
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1
    width_px = cols * tile_size
    height_px = rows * tile_size
    total_pixels = width_px * height_px
    if args.max_total_pixels and total_pixels > args.max_total_pixels:
        if args.auto_downscale and total_pixels > 0 and tile_size > 1:
            # Compute scale factor to fit under limit
            import math

            scale_factor = math.sqrt(args.max_total_pixels / float(total_pixels))
            new_tile = max(1, int(tile_size * scale_factor))
            if new_tile < tile_size:
                print(
                    f"Auto-downscaling tile size from {tile_size} to {new_tile} to fit under max pixels {args.max_total_pixels}."
                )
                tile_size = new_tile
                width_px = cols * tile_size
                height_px = rows * tile_size
                total_pixels = width_px * height_px
        if total_pixels > args.max_total_pixels:
            raise SystemExit(
                f"Aborting: frame size {width_px}x{height_px} (~{total_pixels:,} px) exceeds --max-total-pixels={args.max_total_pixels}.\n"
                "Reduce area (use --min-ts/--max-ts, --bbox, or --center-* with --buffer-size), or lower --tile-size."
            )

    frames_dir = Path(tempfile.mkdtemp(prefix="wplace_timelapse_frames_"))
    try:
        stage_scale_width = args.stage_scale_width if args.stage_scale_width and args.stage_scale_width > 0 else None
        copy_frames_in_order(
            stitched_paths,
            frames_dir,
            stage_scale_width=stage_scale_width,
            progress_interval=max(1, len(stitched_paths) // 20),  # ~5% progress updates
        )

        for fmt, out_path in output_paths:
            _ensure_dir(out_path)
            print(f"Rendering {fmt.upper()} -> {out_path} at framerate {framerate_str}")
            if fmt == "mp4":
                run_ffmpeg_mp4(
                    ffmpeg_bin=ffmpeg_bin,
                    frames_dir=frames_dir,
                    framerate_str=framerate_str,
                    output_path=out_path,
                    scale_width=scale_width,
                    video_codec=video_codec,
                    preset=preset,
                    x264_params=x264_params,
                    crf=crf,
                    pix_fmt=pix_fmt,
                    threads=threads,
                    loglevel=ffmpeg_loglevel,
                    show_stats=ffmpeg_show_stats,
                    progress=ffmpeg_progress,
                )
            elif fmt == "gif":
                # Pass user-selected GIF options via env to the runner
                os.environ['GIF_DITHER'] = gif_dither
                os.environ['GIF_STATS_MODE'] = gif_stats_mode
                os.environ['GIF_DIFF_MODE'] = gif_diff_mode
                run_ffmpeg_gif(
                    ffmpeg_bin=ffmpeg_bin,
                    frames_dir=frames_dir,
                    framerate_str=framerate_str,
                    output_path=out_path,
                    scale_width=gif_scale_width,
                    threads=threads,
                    loglevel=ffmpeg_loglevel,
                    show_stats=ffmpeg_show_stats,
                    progress=ffmpeg_progress,
                    force_size=(gif_width, gif_height) if (gif_width and gif_height) else None,
                    pad_color=gif_pad_color,
                )
            else:
                raise SystemExit(f"Unknown format: {fmt}")

        print("Done.")
        print(f"Staged frames kept at: {frames_dir} (delete manually when done)")
    except Exception:
        print(f"Error during render; staged frames kept at: {frames_dir}")
        raise


if __name__ == "__main__":
    main()


