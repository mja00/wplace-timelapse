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

### License

MIT. See `LICENSE`.

