"""
Download modern legal CJK webcomic samples for translation testing.

Source: Pepper&Carrot by David Revoy, licensed under Creative Commons
Attribution 4.0. These samples are modern colored comic pages with translated
Japanese, Korean, and Chinese pages available from the official website.
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
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from pipeline_paths import MODERN_CJK_ROOT, PROJECT_ROOT


SAMPLES_ROOT = MODERN_CJK_ROOT
MANIFEST_PATH = PROJECT_ROOT / "modern_cjk_samples_manifest.json"
USER_AGENT = "TranslatorPipelineModernValidation/1.0"


SAMPLE_SPECS = [
    ("modern_ja_1", "Japanese", "ja", "E01P01", "Episode 1 page 1, modern colored Japanese page."),
    ("modern_ja_2", "Japanese", "ja", "E01P02", "Episode 1 page 2, modern colored Japanese page."),
    ("modern_ko_1", "Korean", "kr", "E01P01", "Episode 1 page 1, modern colored Korean page."),
    ("modern_ko_2", "Korean", "kr", "E01P02", "Episode 1 page 2, modern colored Korean page."),
    ("modern_zh_1", "Chinese", "cn", "E01P01", "Episode 1 page 1, modern colored Simplified Chinese page."),
    ("modern_zh_2", "Chinese", "cn", "E01P02", "Episode 1 page 2, modern colored Simplified Chinese page."),
]


def _source_url(lang_code: str, page_code: str) -> str:
    return (
        "https://www.peppercarrot.com/0_sources/ep01_Potion-of-Flight/low-res/"
        f"{lang_code}_Pepper-and-Carrot_by-David-Revoy_{page_code}.jpg"
    )


def _english_reference_url(page_code: str) -> str:
    return (
        "https://www.peppercarrot.com/0_sources/ep01_Potion-of-Flight/low-res/"
        f"en_Pepper-and-Carrot_by-David-Revoy_{page_code}.jpg"
    )


def _episode_url(lang_code: str) -> str:
    return f"https://www.peppercarrot.com/{lang_code}/webcomic/ep01_Potion-of-Flight.html"


def _download(url: str, output_path: Path) -> None:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def _normalize_image(source_path: Path, output_path: Path) -> tuple[int, int]:
    with Image.open(source_path) as image:
        normalized = image.convert("RGB")
        normalized.save(output_path, quality=97)
        return normalized.size


def download_modern_samples() -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    english_cache: dict[str, Path] = {}

    for sample_name, language, lang_code, page_code, challenge in SAMPLE_SPECS:
        sample_dir = SAMPLES_ROOT / sample_name
        source_path = sample_dir / "source.jpg"
        input_path = sample_dir / "input.jpg"
        reference_path = sample_dir / "english_reference.jpg"

        print(f"\nDownloading {sample_name}: {language} {page_code}")
        if not input_path.exists():
            _download(_source_url(lang_code, page_code), source_path)
            size = _normalize_image(source_path, input_path)
        else:
            with Image.open(input_path) as image:
                size = image.size
            print(f"  Reusing {input_path}")

        if page_code not in english_cache:
            english_tmp = sample_dir / f"english_{page_code}.jpg"
            if not english_tmp.exists():
                _download(_english_reference_url(page_code), english_tmp)
            english_cache[page_code] = english_tmp
        if not reference_path.exists():
            reference_path.write_bytes(english_cache[page_code].read_bytes())

        record = {
            "sample_name": sample_name,
            "language": language,
            "input_file": "input.jpg",
            "english_reference_file": "english_reference.jpg",
            "source_url": _source_url(lang_code, page_code),
            "english_reference_url": _english_reference_url(page_code),
            "source_page": _episode_url(lang_code),
            "license": "CC BY 4.0",
            "attribution": "Pepper&Carrot by David Revoy",
            "challenge": challenge,
            "normalized_size": list(size),
        }
        samples.append(record)
        (sample_dir / "source_info.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved {input_path} ({size[0]}x{size[1]})")

    manifest = {
        "created_for": "modern external CJK translation validation",
        "source": "Pepper&Carrot official website",
        "license_policy": "Modern legal CC BY 4.0 webcomic pages; no piracy or scanlation sources.",
        "samples": samples,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    download_modern_samples()
