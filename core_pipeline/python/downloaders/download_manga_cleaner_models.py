from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "manga_cleaner"
REPO_ID = "chflame163/ComfyUI_LayerStyle"
FILES = [
    "ComfyUI/models/lama/manga_inpaintor.jit",
    "ComfyUI/models/lama/erika.jit",
]


def main() -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(MODEL_ROOT),
            )
        )
        print(f"{filename} -> {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
