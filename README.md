## wplace-timelapse

Small script to download one or more map tiles from `backend.wplace.live`, with retry/backoff and optional browser-impersonating client. Includes a helper to stitch downloaded tiles into a single image.

### Requirements

- Python 3.10+

### Setup

Windows (PowerShell):

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Set environment variables or put them in a `.env` file (auto-loaded).

- `TILE_TARGET_X`, `TILE_TARGET_Y`: center tile to download (defaults: 603, 763)
- `TILE_BUFFER_SIZE`: radius around the center tile to include; 3 downloads a 7×7 grid (default: 3)
- `COOKIE`: full `Cookie` header from your browser (optional)
- `CF_CLEARANCE`: only the Cloudflare `cf_clearance` token (optional)
- `HTTP_CLIENT`: `curl_cffi`, `httpx`, or `requests` (default: prefers `curl_cffi` if installed)
- `CURL_IMPERSONATE`: browser TLS fingerprint when using `curl_cffi` (default: `chrome`)
- `REQUEST_DELAY_SECONDS`: fixed delay before each request (default: 0)
- `MAX_RETRIES`: max retries for 429/network errors with exponential backoff (default: 6)

Example `.env`:

```dotenv
# Target area (603, 763) with a 3‑tile buffer (7x7 grid)
TILE_TARGET_X=603
TILE_TARGET_Y=763
TILE_BUFFER_SIZE=3

# If the site works in your browser but not via script, pass cookies
CF_CLEARANCE=your_cf_clearance_token
# Or the full Cookie header:
COOKIE=cf_clearance=...; other_cookie=...

# Optional client selection / fingerprint
HTTP_CLIENT=curl_cffi
CURL_IMPERSONATE=chrome

# Throttling and retries
REQUEST_DELAY_SECONDS=0.25
MAX_RETRIES=6
```

### Usage

Run the downloader:

Windows (PowerShell):

```powershell
$env:TILE_TARGET_X = "603"
$env:TILE_TARGET_Y = "763"
$env:TILE_BUFFER_SIZE = "3"
python main.py
```

Linux/macOS:

```bash
TILE_TARGET_X=603 TILE_TARGET_Y=763 TILE_BUFFER_SIZE=3 python main.py
```

If your browser works but the script gets 403/503, pass cookies:

```powershell
# Only the Cloudflare token
$env:CF_CLEARANCE = "<token>"
python main.py

# Or the full Cookie header
$env:COOKIE = "cf_clearance=...; other_cookie=..."
python main.py
```

Switch HTTP client if 403 persists:

```powershell
$env:HTTP_CLIENT = "curl_cffi"  # or httpx or requests
$env:CURL_IMPERSONATE = "chrome" # or chrome110, edge101, etc.
python main.py
```

Handling rate limiting (HTTP 429):

```powershell
$env:REQUEST_DELAY_SECONDS = "0.25"  # add fixed delay before each request
$env:MAX_RETRIES = "6"               # control retry attempts
python main.py
```

### Output

Each run saves PNGs to `tiles/<timestamp>/` named `x_y.png`.

### Scheduling (Linux/macOS)

The provided script activates a venv and runs the downloader from cron every 10 minutes:

```bash
# Make executable once
chmod +x run_timelapse.sh

# Edit crontab
crontab -e

# Add a line like this (adjust path):
*/10 * * * * /usr/bin/env bash /path/to/wplace-timelapse/run_timelapse.sh >> /path/to/wplace-timelapse/cron.log 2>&1
```

If you don't have a venv yet:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Stitching tiles into one image

Use `stitch_tiles.py` to build a single image; missing tiles become transparent.

```powershell
# Example: stitch the output of a previous run
python stitch_tiles.py tiles/20250810_162535 --output tiles/20250810_162535/stitched.png

# Customize tile size if needed (defaults to 1000)
python stitch_tiles.py tiles/20250810_162535 --tile-size 1000

# Optional: place the stitched image on a solid background (e.g., white)
python stitch_tiles.py tiles/20250810_162535 --background white

# Or use hex color
python stitch_tiles.py tiles/20250810_162535 --background "#000000"
```

Show help:

```bash
python stitch_tiles.py -h
```

### Render a timelapse (MP4 or GIF)

Note: GIF rendering will be lower quality than MP4.

Use `generate_timelapse.py` to stitch each timestamped folder under `tiles/` into a frame, then render a timelapse with ffmpeg.

Windows (PowerShell):

```powershell
# Render MP4 with 2 fps (default)
python generate_timelapse.py --format mp4

# Show each frame for 1.5s using a rational framerate
$env:FRAME_DURATION_SECONDS = "1.5"
python generate_timelapse.py --format mp4

# Or set explicit FPS
$env:FRAMERATE = "5"
python generate_timelapse.py --format gif --scale-width 1200

# Use a custom ffmpeg path (if not on PATH)
$env:FFMPEG_BINARY = "C:\\ffmpeg\\bin\\ffmpeg.exe"
python generate_timelapse.py --format both
```

Linux/macOS:

```bash
# Render default MP4
python generate_timelapse.py --format mp4

# Hold each frame for 2 seconds
FRAME_DURATION_SECONDS=2 python generate_timelapse.py --format gif

# Scale output width to 1600px, keep aspect ratio
SCALE_WIDTH=1600 python generate_timelapse.py --format mp4
```

By default the script will:

- Stitch each `tiles/<timestamp>/` to `tiles/<timestamp>/stitched.png`, skipping ones that already exist.
- Ensure all frames share a consistent canvas using a global bounding box across timestamps, so the timelapse doesn’t wobble.
- Render frames in chronological order (sorted by folder name).

You can force re-stitching with `--force-stitch` or `FORCE_STITCH=1`.

### MP4 compatibility (Windows/Discord)

The MP4 encoder is configured for broad compatibility:

- H.264 `yuv420p`, `-profile:v high`, `-level:v 4.1`, `-tag:v avc1`, `-movflags +faststart`.
- Constant frame pacing is enforced and output dimensions are made even.
- If you do not pass `--scale-width`, the width is auto-clamped to a safe maximum (`MP4_MAX_WIDTH`, default 3840). Override via env or pass `--scale-width` explicitly.

Examples:

```powershell
# Default compatible MP4
python .\generate_timelapse.py --format mp4

# Force 1080p and a conservative profile/level
$env:MP4_MAX_WIDTH = "1920"; $env:H264_PROFILE = "main"; $env:H264_LEVEL = "4.0"
python .\generate_timelapse.py --format mp4
```

Environment variables (can also be set via `--flags` where provided):

- `FRAMERATE`: input framerate for frames (e.g., `5`). Alternative to `FRAME_DURATION_SECONDS`.
- `FRAME_DURATION_SECONDS`: seconds per frame (e.g., `1.5`). Takes precedence over `FRAMERATE`.
- `FFMPEG_BINARY`: ffmpeg binary name or path (default: `ffmpeg`).
- `OUTPUT_FORMAT`: `mp4`, `gif`, or `both` (default: `mp4`).
- `SCALE_WIDTH`: optional width to scale output to (height auto-calculated).
- `GIF_SCALE_WIDTH`: optional GIF width. Defaults to 1000 if unset to avoid OOM on large frames.
- `TILE_SIZE`: expected tile size when stitching (default: 1000).
- `STITCH_BACKGROUND`: optional solid background color used when stitching (e.g., `white`, `#000`).
- `FORCE_STITCH`: re-stitch even if `stitched.png` exists (`1`/`0`).
- `VIDEO_CODEC`: mp4 video codec (default: `libx264`).
- `CRF`: mp4 CRF quality (default: `20`, lower is higher quality).
- `PIX_FMT`: mp4 pixel format (default: `yuv420p`).
- `MP4_MAX_WIDTH`: maximum width when no `--scale-width` is provided; auto-downscales if wider (default: `3840`).
- `H264_PROFILE`: H.264 profile for MP4 (default: `high`).
- `H264_LEVEL`: H.264 level for MP4 (default: `4.1`).
- `TILES_ROOT`: root folder containing timestamped subfolders (default: `tiles`).
- `MIN_TIMESTAMP` / `MAX_TIMESTAMP`: include only folders in this lexicographic range.
- `BBOX`: explicit tile bbox `minX,minY,maxX,maxY` to constrain canvas.
- `CENTER_X`, `CENTER_Y`, `BUFFER_SIZE`: alternative bbox definition centered on a tile.
- `MAX_TOTAL_PIXELS`: safety limit for frame area (default ~80MP); aborts if exceeded.
- `LIMIT_FRAMES`: only use first N frames (useful for testing and to reduce load).

Show help:

```bash
python generate_timelapse.py -h
```

### License

MIT. See `LICENSE`.

