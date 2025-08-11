# flake8: noqa: E501
import os
import time
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict
import requests as std_requests

# Load .env file if it exists using dotenv
from dotenv import load_dotenv

load_dotenv()

try:
    # Optional: use a browser-impersonating HTTP client to avoid bot blocks
    from curl_cffi import requests as curl_requests  # type: ignore

    HAVE_CURL_CFFI = True
except Exception:  # pragma: no cover - optional dependency
    curl_requests = None  # type: ignore
    HAVE_CURL_CFFI = False

try:
    # Optional: HTTP/2 client with different TLS/ALPN fingerprint
    import httpx  # type: ignore

    HAVE_HTTPX = True
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore
    HAVE_HTTPX = False


def parse_cookie_header(raw_cookie_header: str) -> Dict[str, str]:
    """Parse a `Cookie:` header string into a dict for requests.

    Accepts a semicolon-delimited cookie string like
    "name=value; other=value2" and returns {"name": "value", ...}.
    """
    cookie_dict: Dict[str, str] = {}
    if not raw_cookie_header:
        return cookie_dict
    for part in raw_cookie_header.split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookie_dict[name.strip()] = value.strip()
    return cookie_dict


def get_cookies_from_env() -> Dict[str, str]:
    """Build a cookie dict from environment variables, if provided.

    Supported env vars:
    - COOKIE: full `Cookie:` header string copied from the browser
    - CF_CLEARANCE: just the Cloudflare clearance token value
    """
    env_cookie = os.environ.get("COOKIE", "").strip()
    cookies: Dict[str, str] = parse_cookie_header(env_cookie)

    cf_clearance = os.environ.get("CF_CLEARANCE", "").strip()
    if cf_clearance and "cf_clearance" not in cookies:
        cookies["cf_clearance"] = cf_clearance

    return cookies


def create_http_session():
    """Create an HTTP session.

    Prefers `curl_cffi` (Chrome impersonation) if available to better match
    a real browser TLS/HTTP2 fingerprint. Falls back to std `requests`.
    You can force the client via env var `HTTP_CLIENT` (values: `curl_cffi`,
    `requests`).
    """
    preferred = os.environ.get("HTTP_CLIENT", "curl_cffi").lower()
    if HAVE_CURL_CFFI and preferred == "curl_cffi":
        impersonate = os.environ.get("CURL_IMPERSONATE", "chrome")
        session = curl_requests.Session(impersonate=impersonate)  # type: ignore
        session.headers.update(COMMON_HEADERS)
        return session

    if HAVE_HTTPX and preferred in ("httpx", "curl_cffi"):
        # If curl_cffi isn't available, try HTTP/2 via httpx
        session = httpx.Client(http2=True, headers=COMMON_HEADERS)  # type: ignore
        return session

    session = std_requests.Session()
    session.headers.update(COMMON_HEADERS)
    return session


COMMON_HEADERS = {
    # Mirror Postman minimal headers that worked in your test
    "User-Agent": "PostmanRuntime/7.45.0",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

base_url = "https://backend.wplace.live/files/s0/tiles/%s/%s.png"

# WANTED_TILES: All tiles in a configurable buffer around a single target tile.
# The target tile can be set via the TILE_TARGET_X and TILE_TARGET_Y environment variables (default: 603, 763).
# The buffer size can be set via the TILE_BUFFER_SIZE environment variable (default: 3).
# This will include all tiles from (cx-buffer, cy-buffer) to (cx+buffer, cy+buffer) for the target.
TILE_BUFFER_SIZE = int(os.environ.get("TILE_BUFFER_SIZE", "3"))
TILE_TARGET_X = int(os.environ.get("TILE_TARGET_X", "603"))
TILE_TARGET_Y = int(os.environ.get("TILE_TARGET_Y", "763"))
WANTED_TILES = [
    (x, y)
    for x in range(TILE_TARGET_X - TILE_BUFFER_SIZE, TILE_TARGET_X + TILE_BUFFER_SIZE + 1)
    for y in range(TILE_TARGET_Y - TILE_BUFFER_SIZE, TILE_TARGET_Y + TILE_BUFFER_SIZE + 1)
]
WANTED_TILES = list(sorted(set(WANTED_TILES)))

print(f"Downloading {len(WANTED_TILES)} tiles")

# Download the tiles into a folder of the current time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure output directory exists (tiles/<timestamp>/)
output_dir = os.path.join("tiles", current_time)
os.makedirs(output_dir, exist_ok=True)

session = create_http_session()

cookies = get_cookies_from_env()

def _parse_retry_after_seconds(header_value: str) -> float:
    """Parse Retry-After header value to seconds.

    Supports either a delay in seconds or an HTTP-date.
    Returns 0.0 if parsing fails or would be in the past.
    """
    if not header_value:
        return 0.0
    header_value = header_value.strip()
    # Numeric seconds
    if header_value.isdigit():
        try:
            seconds = float(header_value)
            return max(0.0, seconds)
        except Exception:
            return 0.0
    # HTTP-date
    try:
        dt = parsedate_to_datetime(header_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return 0.0


def download_tile_with_retries(
    session,
    cookies: Dict[str, str],
    tile: tuple,
    *,
    max_retries: int,
    base_delay_seconds: float,
) -> bool:
    """Download one tile with 429-aware backoff and optional base delay.

    Returns True on success, False otherwise.
    """
    url = base_url % (tile[0], tile[1])
    attempt_index = 0
    while True:
        if base_delay_seconds > 0:
            time.sleep(base_delay_seconds)
        try:
            response = session.get(url, cookies=cookies, timeout=20)
        except Exception as exc:
            if attempt_index >= max_retries:
                print(f"Failed to download tile {tile}: network error after retries: {exc}")
                return False
            # Backoff on transient network issues
            wait_seconds = min(2 ** attempt_index, 60) + random.uniform(0, 0.5)
            print(
                f"Network error for tile {tile}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s (attempt {attempt_index + 1}/{max_retries})."
            )
            time.sleep(wait_seconds)
            attempt_index += 1
            continue

        # HTTP 429: Too Many Requests
        if getattr(response, "status_code", None) == 429:
            if attempt_index >= max_retries:
                print(f"Failed to download tile {tile}: HTTP 429 after {max_retries} retries")
                return False
            retry_after_header = response.headers.get("Retry-After", "")
            retry_after = _parse_retry_after_seconds(retry_after_header)
            if retry_after <= 0:
                retry_after = min(2 ** attempt_index, 60)
            retry_after += random.uniform(0, 0.5)  # jitter
            print(
                f"Rate limited (429) for tile {tile}. "
                f"Retrying in {retry_after:.1f}s (attempt {attempt_index + 1}/{max_retries})."
            )
            time.sleep(retry_after)
            attempt_index += 1
            continue

        # Non-200 and not 429
        if response.status_code != 200:
            print(f"Failed to download tile {tile}: HTTP {response.status_code}")
            # Cloudflare/anti-bot often returns 403/503 or a 1020 body
            # Using .text for compatibility across clients
            try:
                snippet = response.text[:500].replace("\n", " ")
            except Exception:
                snippet = "<no text>"
            print(f"Response snippet: {snippet}")
            if response.status_code in (401, 403, 503) or "1020" in snippet:
                print(
                    "Hint: If it works in your browser, provide cookies via the "
                    "COOKIE env var (full 'Cookie' header) or CF_CLEARANCE."
                )
            return False

        # Content-Type validation
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/png"):
            print(
                f"Failed to download tile {tile}: unexpected content-type "
                f"{content_type}"
            )
            return False

        # Save file
        file_path = os.path.join(output_dir, f"{tile[0]}_{tile[1]}.png")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded tile {tile} -> {file_path}")
        return True


# Allow users to throttle requests to avoid 429
BASE_DELAY_SECONDS = float(os.environ.get("REQUEST_DELAY_SECONDS", "0"))
# How many times to retry on 429/network errors
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))

for tile in WANTED_TILES:
    download_tile_with_retries(
        session,
        cookies,
        tile,
        max_retries=MAX_RETRIES,
        base_delay_seconds=BASE_DELAY_SECONDS,
    )
