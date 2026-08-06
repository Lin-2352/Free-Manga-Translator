from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

# Mirrors run_extension_live_smoke.py's bootstrap, but this script has no in-process
# pipeline import to make -- it only needs PROJECT_ROOT for default paths.
_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
PROJECT_ROOT = _PROJECT_ROOT_FOR_IMPORTS
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS

# Every other "backend" test in this repo (run_extension_live_smoke.py,
# test_extension_backend_contract.py, test_backend_concurrency_stress.py) imports
# backend_api.app.main and calls translate_image() IN-PROCESS -- none of them ever
# make an actual HTTP request, let alone to a remote host. That means nothing in this
# repo's test suite has ever exercised the one path real users hit: a Chrome extension
# on their own machine, talking over the public internet to a Kaggle-hosted backend
# through an ngrok tunnel. The 403 (auth-token race) and interstitial-HTML (missing
# ngrok-skip-browser-warning) failures diagnosed earlier this session both lived
# exactly in that untested gap. This script closes it: same request shape the real
# extension sends, over real HTTP, to whatever URL you give it.


def _data_url_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _write_data_url(path: Path, data_url: str) -> None:
    if "," not in data_url:
        raise ValueError("Response image is not a data URL")
    encoded = data_url.split(",", 1)[1]
    path.write_bytes(base64.b64decode(encoded))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real HTTP smoke test against a remote (e.g. Kaggle+ngrok) backend -- "
        "the same request shape and headers the Chrome extension actually sends."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Full translate-image endpoint, e.g. https://your-domain.ngrok-free.dev/v1/translate-image "
        "-- must include the path, a bare origin will 404 (see docs/KAGGLE_USER_MANUAL.md section 10).",
    )
    parser.add_argument(
        "--auth-token",
        default="",
        help="Must match the backend's FMT_AUTH_TOKEN exactly. Omit only if the backend has no token set.",
    )
    parser.add_argument(
        "--image",
        default=str(PROJECT_ROOT / "samples" / "sample1" / "sample.jpg"),
        help="Input manga image path.",
    )
    parser.add_argument("--source-language", default="ja", choices=["ja", "ko", "zh"])
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Seconds. 600 matches the notebook's own Cell 5b -- a cold Kaggle CPU-bound OCR "
        "pass can genuinely take several minutes.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Write the translated image to quality_reports/extension_runtime/remote_smoke_output.png "
        "for manual inspection. Off by default -- this decodes/validates the image either way.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    payload = json.dumps(
        {
            "imageData": _data_url_for_image(image_path),
            "sourceLanguage": args.source_language,
            "targetLanguage": "en",
            "qualityProfile": "strict",
        }
    ).encode("utf-8")

    # The exact three headers background.js's fmtHeaders() sends on every real request
    # (background.js) -- X-Fmt-Client always, ngrok-skip-browser-warning always
    # (bypasses the free-tier interstitial), X-Fmt-Auth only when a token is set,
    # matching how a blank popup token field behaves for real.
    headers = {
        "Content-Type": "application/json",
        "X-Fmt-Client": "free-manga-translator-extension",
        "ngrok-skip-browser-warning": "1",
    }
    if args.auth_token:
        headers["X-Fmt-Auth"] = args.auth_token

    request = urllib.request.Request(args.url, data=payload, method="POST", headers=headers)

    print(f"POST {args.url}")
    print(f"auth token: {'set (' + str(len(args.auth_token)) + ' chars)' if args.auth_token else 'not set'}")
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} from {args.url} after {time.time() - start:.1f}s -- body: {body[:500]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach {args.url}: {error.reason}") from error
    elapsed = time.time() - start

    # A real, distinguishable failure mode: ngrok's free-tier interstitial page comes
    # back as text/html with a 200, not an HTTP error -- indistinguishable from success
    # by status code alone. Catch it explicitly instead of letting json.loads() produce
    # a confusing generic parse error.
    if "text/html" in content_type.lower():
        raise RuntimeError(
            f"Received an HTML response instead of JSON (Content-Type: {content_type}) after "
            f"{elapsed:.1f}s -- this is ngrok's browser-warning interstitial. The "
            "ngrok-skip-browser-warning header was sent; if you still see this, the tunnel or "
            f"backend is running an older build. Body preview: {raw_body[:300]!r}"
        )

    try:
        result = json.loads(raw_body)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Response was not valid JSON after {elapsed:.1f}s -- body preview: {raw_body[:300]!r}"
        ) from error

    print(f"HTTP 200, {elapsed:.1f}s, Content-Type: {content_type}")
    print("status:", result.get("status"))
    if result.get("status") != "pass":
        raise RuntimeError(f"Remote translate did not pass: {result.get('error')}")

    report = result.get("report", {})
    print(
        "summary:",
        {
            "ocrItems": report.get("ocrItems"),
            "translations": report.get("translations"),
            "renderedRegions": report.get("renderedRegions"),
            "totalSeconds": report.get("totalSeconds"),
        },
    )
    stage_timings = report.get("stageTimings") or []
    if stage_timings:
        print("stageTimings:", ", ".join(f"{s['stage']}={s['seconds']}s" for s in stage_timings))
    serving_providers = report.get("translationServingProviders") or []
    print(f"translation provider: {', '.join(serving_providers) if serving_providers else 'local NLLB fallback'}")

    out_data_url = result.get("translatedImageDataUrl") or result.get("imageDataUrl")
    if not out_data_url or "," not in out_data_url:
        raise RuntimeError("Response had status 'pass' but no translated image data URL was returned.")
    out_bytes = base64.b64decode(out_data_url.split(",", 1)[1])

    with PILImage.open(BytesIO(out_bytes)) as out_image:
        out_image.load()  # raises on a truncated/corrupt image
        print(f"output image OK: {out_image.width}x{out_image.height}, {len(out_bytes) / 1024:.0f} KB")

    if args.save_output:
        out_dir = PROJECT_ROOT / "quality_reports" / "extension_runtime"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "remote_smoke_output.png"
        _write_data_url(output_path, out_data_url)
        print(f"output saved: {output_path}")

    print("remote_backend_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
