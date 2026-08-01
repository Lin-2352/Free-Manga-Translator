"""
Download legal/open external CJK comic samples for pipeline stress testing.

The sample set intentionally uses Wikimedia Commons files so the test can be
reproduced without piracy sites or copyrighted scanlation pages.
"""
from __future__ import annotations

# --- Clean-copy path bootstrap ---
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_BOOTSTRAP_FILE = _BootstrapPath(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in (
    "python/common",
    "python/steps",
    "python/validation",
    "python/runtime",
    "python/downloaders",
    "python/reference",
    "python/diagnostics",
):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _path)
del _BootstrapPath, _bootstrap_sys, _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path
# --- End clean-copy path bootstrap ---

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from pipeline_paths import EXTERNAL_CJK_ROOT, PROJECT_ROOT


USER_AGENT = "TranslatorPipelineValidation/1.0 (local reproducible QA)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SAMPLES_ROOT = EXTERNAL_CJK_ROOT
MANIFEST_PATH = PROJECT_ROOT / "external_cjk_samples_manifest.json"


SAMPLE_SPECS = [
    {
        "sample_name": "external_ja_1",
        "language": "Japanese",
        "title": "File:Wikipe-tan manga page1.jpg",
        "challenge": "Small page, multiple bubbles, mixed layout.",
    },
    {
        "sample_name": "external_ja_2",
        "language": "Japanese",
        "title": "File:Wikipe-tan manga page2.jpg",
        "challenge": "Page continuation with dense speech regions.",
    },
    {
        "sample_name": "external_ko_1",
        "language": "Korean",
        "title": "File:Manhwa-Yu.Gil-jun-Yahak-01.jpg",
        "challenge": "Historical manhwa scan with low-resolution Korean lettering.",
    },
    {
        "sample_name": "external_ko_2",
        "language": "Korean",
        "encoded_filename": "%EB%B4%84%EC%9D%B4_%EC%93%B0%EB%8A%94_%EB%A7%8C%EB%AC%B8_%EB%B4%84%EC%9D%B4_%EA%B7%B8%EB%A6%AC%EB%8A%94_%EB%A7%8C%ED%99%94%28%EC%B5%9C%EC%98%81%EC%88%98%E4%BD%9C%2C_1933%EB%85%84_4%EC%9B%94_3%EC%9D%BC%29.png",
        "challenge": "Historical Korean newspaper cartoon with mixed print quality.",
    },
    {
        "sample_name": "external_zh_1",
        "language": "Chinese",
        "encoded_filename": "%E5%9C%8B%E9%98%B2online%E6%BC%AB%E7%95%AB%E6%95%98%E8%8A%B1%E8%93%AE%E9%9C%87%E7%81%BD_01.jpg",
        "challenge": "Tall manhua-style page with Traditional Chinese bubbles.",
    },
    {
        "sample_name": "external_zh_2",
        "language": "Chinese",
        "encoded_filename": "%E5%9C%8B%E9%98%B2online%E6%BC%AB%E7%95%AB%E6%95%98%E8%8A%B1%E8%93%AE%E9%9C%87%E7%81%BD_02.jpg",
        "challenge": "Tall page continuation with several dialogue regions.",
    },
]


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _title_from_spec(spec: dict[str, str]) -> str:
    if spec.get("title"):
        return spec["title"]
    filename = urllib.parse.unquote(spec["encoded_filename"])
    return f"File:{filename}"


def _request_with_backoff(url: str, **kwargs: Any) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(5):
        response = requests.get(url, headers=_headers(), timeout=60, **kwargs)
        last_response = response
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait_seconds = 5 + attempt * 5
        print(f"  Commons rate-limited; waiting {wait_seconds}s")
        time.sleep(wait_seconds)
    assert last_response is not None
    last_response.raise_for_status()
    return last_response


def _commons_image_info(title: str) -> dict[str, Any]:
    response = _request_with_backoff(
        COMMONS_API,
        params={
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "900",
            "format": "json",
        },
    )
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "imageinfo" not in page:
        raise RuntimeError(f"Commons file not found: {title}")
    image_info = page["imageinfo"][0]
    metadata = image_info.get("extmetadata", {})
    return {
        "title": title,
        "source_url": image_info["descriptionurl"],
        "download_url": image_info["url"],
        "thumb_url": image_info.get("thumburl", ""),
        "mime": image_info.get("mime", ""),
        "width": int(image_info.get("width", 0)),
        "height": int(image_info.get("height", 0)),
        "license": metadata.get("LicenseShortName", {}).get("value", ""),
        "artist": metadata.get("Artist", {}).get("value", ""),
        "attribution": metadata.get("Attribution", {}).get("value", ""),
    }


def _download_binary(url: str, output_path: Path, fallback_url: str = "") -> None:
    try:
        response = _request_with_backoff(url, stream=True)
    except requests.HTTPError as error:
        if not fallback_url:
            raise
        print(f"  Full-size download failed ({error.response.status_code}); using Commons thumbnail.")
        response = _request_with_backoff(fallback_url, stream=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out_file:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                out_file.write(chunk)


def _normalize_image(source_path: Path, output_path: Path) -> tuple[int, int]:
    with Image.open(source_path) as image:
        normalized = image.convert("RGB")
        normalized.save(output_path, quality=97)
        return normalized.size


def download_external_samples() -> dict[str, Any]:
    manifest_samples: list[dict[str, Any]] = []
    for spec in SAMPLE_SPECS:
        sample_name = spec["sample_name"]
        title = _title_from_spec(spec)
        print(f"\nDownloading {sample_name}: {title}")
        sample_dir = SAMPLES_ROOT / sample_name
        cached_info_path = sample_dir / "source_info.json"
        cached_input_path = sample_dir / "input.jpg"
        if cached_input_path.exists() and cached_info_path.exists():
            sample_record = json.loads(cached_info_path.read_text(encoding="utf-8"))
            manifest_samples.append(sample_record)
            print(f"  Reusing existing {cached_input_path}")
            continue

        info = _commons_image_info(title)
        extension = ".png" if "png" in info["mime"].lower() else ".jpg"
        raw_path = sample_dir / f"source{extension}"
        input_file = "input.jpg"
        input_path = sample_dir / input_file
        preferred_url = info.get("thumb_url") or info["download_url"]
        _download_binary(preferred_url, raw_path, fallback_url=info["download_url"])
        normalized_size = _normalize_image(raw_path, input_path)

        sample_record = {
            "sample_name": sample_name,
            "language": spec["language"],
            "input_file": input_file,
            "source_title": info["title"],
            "source_url": info["source_url"],
            "download_url": info["download_url"],
            "license": info["license"],
            "challenge": spec["challenge"],
            "source_size": [info["width"], info["height"]],
            "normalized_size": list(normalized_size),
        }
        manifest_samples.append(sample_record)
        (sample_dir / "source_info.json").write_text(
            json.dumps(sample_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved {input_path} ({normalized_size[0]}x{normalized_size[1]})")

    manifest = {
        "created_for": "external CJK pipeline validation",
        "license_policy": "Only open/legal Wikimedia Commons files are used; no piracy or scanlation sites.",
        "samples": manifest_samples,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    download_external_samples()
