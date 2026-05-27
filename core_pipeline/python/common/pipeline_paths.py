from __future__ import annotations

import os
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
if _THIS_DIR.name == "common" and _THIS_DIR.parent.name == "python":
    PROJECT_ROOT = _THIS_DIR.parent.parent
else:
    PROJECT_ROOT = _THIS_DIR
DEFAULT_SAMPLES_ROOT = PROJECT_ROOT / "samples"
VALIDATION_ROOT = PROJECT_ROOT / "validation_samples"
EXTERNAL_CJK_ROOT = VALIDATION_ROOT / "external_cjk"
MODERN_CJK_ROOT = VALIDATION_ROOT / "modern_cjk"
RUNTIME_ROOT = PROJECT_ROOT / "runtime_samples"
EXTENSION_RUNTIME_ROOT = RUNTIME_ROOT / "extension"


def sample_root_from_env(default: Path = DEFAULT_SAMPLES_ROOT) -> Path:
    configured = os.environ.get("PIPELINE_SAMPLES_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return default.resolve()
