"""
Step 5 — OCR & Consolidation (v4: Self-Contained Detection)
==========================================================
1. Check for Step 1-3 results. If missing, RUN detection models.
2. Group and consolidate detections by bubble.
3. Run Japanese OCR on consolidated Red Box regions.
4. Save results for Layout and Translation.
"""

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
# torch must be imported before cv2 in this process: on this platform,
# loading opencv-python's native runtime first and torch's afterward
# causes a hard segfault (not a catchable exception) the first time a
# torch-based model (manga-ocr, NLLB) actually runs -- verified via
# isolated import-order reproduction. Whichever step script imports first
# in a given process determines the load order for everything downstream,
# so every entry point that eventually touches cv2 needs this guard.
import torch  # noqa: F401  (import-order guard, see comment above)
import json
import os
import base64
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import cv2
import numpy as np
from pathlib import Path
from api_manager import API_MANAGER, ApiProviderAuthLocked, ApiProviderUnavailable, ApiQuotaExhausted, ApiRateLimited
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env
from ml_region_lib import (
    MLConfig, load_ocr_model, load_text_model, load_bubble_model, load_semantic_model,
    detect_text, detect_bubbles, detect_semantic_text_regions,
    build_step2_routing_state, consolidate_by_bubble, Box,
    SAMPLE_MAP, classify_text_by_content,
    _safe_bubble_mask, _polygon_from_mask, _bubble_cluster_zones,
)

_EASYOCR_READERS = {}
_PADDLEOCR_READERS = {}
_PADDLEOCR_UNAVAILABLE = set()
_MANGA_OCR_MODEL = None
_TEXT_HANDLE = None
_BUBBLE_MODEL = None
_BUBBLE_DEVICE = None
_SEMANTIC_HANDLE = None
_STEP5_RUN_LOCK = threading.RLock()
ENV_FILE = _PROJECT_ROOT_FOR_IMPORTS / ".env" if "_PROJECT_ROOT_FOR_IMPORTS" in globals() else Path(__file__).resolve().parents[2] / ".env"


def _local_cjk_mode() -> bool:
    return os.environ.get("LOCAL_NLLB_TRANSLATION", "").strip().lower() in {"1", "true", "yes", "on"}


def _sample_cjk_ocr_language(sample_name: str) -> str | None:
    lowered = sample_name.lower()
    chinese_markers = ("_zh_", "zh_", "_chi", "(chi", "_chinese", "(chinese", "_cn", "(cn")
    korean_markers = ("_ko_", "ko_", "_kor", "(ko", "(kor", "_korean", "(korean", "_kr", "(kr")
    if any(marker in lowered for marker in chinese_markers) or lowered.startswith(("external_zh", "modern_zh", "runtime_zh")):
        # Was unconditionally "ch_tra" (Traditional) for every zh sample -- never a detected
        # choice, just an unverified guess, and Simplified is the more common real-world source
        # script. This EasyOCR reader is only ever reached as a last-resort fallback behind
        # PaddleOCR (lang="ch", script-agnostic) -- so this default only matters when paddle is
        # unavailable, but should still default to the more likely case rather than the less
        # likely one.
        return "ch_sim"
    if any(marker in lowered for marker in korean_markers) or lowered.startswith(("external_ko", "modern_ko", "runtime_ko")):
        return "ko"
    return None


def _is_chinese_ocr_language(language: str | None) -> bool:
    return language in {"ch_tra", "ch_sim"}


def _easyocr_reader(language: str):
    if language not in _EASYOCR_READERS:
        import easyocr

        _EASYOCR_READERS[language] = easyocr.Reader([language, "en"], gpu=True, verbose=False)
        print(f"  [Model D2] EasyOCR {language}: LOADED")
    return _EASYOCR_READERS[language]


def _paddleocr_reader(language: str):
    if language not in {"ko", "ch"}:
        return None
    if language in _PADDLEOCR_UNAVAILABLE:
        return None
    if language not in _PADDLEOCR_READERS:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
        # Paddle grows into VRAM on demand instead of grabbing a slab next to torch's.
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        try:
            # torch MUST already be imported before paddle in this process: both frameworks
            # vendor their own cuDNN 9.x sub-DLLs, and Windows resolves DLLs by basename, so
            # whichever loads second can bind the other's cuDNN siblings at the wrong minor
            # version (import paddle first here reproducibly raises WinError 127 on
            # torch\lib\shm.dll). Every real caller already loads torch first (detection/
            # magi/manga-ocr run before this lazy import), so this is a no-op at runtime and
            # a guard against a future caller that isn't.
            import torch  # noqa: F401
            import paddle
            from paddleocr import PaddleOCR
        except (ModuleNotFoundError, ImportError, OSError, RuntimeError) as exc:
            # Was ModuleNotFoundError only -- correct for "paddle isn't installed" but not for
            # a PARTIALLY broken install (a missing libcudnn sub-DLL, a mismatched wheel from
            # an interrupted multi-index install loop -- see the Kaggle notebook's Cell 2 §6e,
            # a best-effort install that can leave paddle importable-but-broken rather than
            # absent). That raised ImportError/OSError/RuntimeError instead, uncaught, crashing
            # step 5 outright for zh/ko requests instead of falling back the way a genuinely
            # missing module already does. Every zh runtime sample in this repo's own fixture
            # tree was produced via this exact path succeeding -- there is zero evidence the
            # fallback below has ever been exercised for zh, so a crash here silently means
            # "Chinese OCR stopped working," not a visible error.
            _PADDLEOCR_UNAVAILABLE.add(language)
            print(f"  [PaddleOCR warn] {language}: paddleocr unavailable ({type(exc).__name__}: {exc}); using non-Paddle OCR fallbacks")
            return None

        paddle_lang = "korean" if language == "ko" else "ch"
        # Explicit, not left to PaddleOCR's own get_default_device() auto-detection:
        # that reports "gpu:0" whenever a GPU is physically present, even when the
        # installed paddlepaddle build has no CUDA kernels at all (e.g. the CPU-only
        # wheel) -- so it doesn't fail loudly, it silently runs on CPU while claiming
        # GPU. Deriving this from PADDLE's own compiled-with-cuda flag, not torch's --
        # torch having working CUDA says nothing about whether THIS installed paddle
        # build does. Verified directly: on a CPU-only paddle build with a working CUDA
        # torch, asking PaddleOCR for device="gpu:0" doesn't silently fall back, it
        # raises ValueError at init ("PaddlePaddle is not compiled with CUDA") -- so the
        # old torch-derived check could crash OCR init outright on exactly the
        # mismatched-build case it was trying to protect against.
        _paddle_device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
        if os.environ.get("MANGA_PADDLE_DEVICE", "").strip().lower() == "cpu":
            _paddle_device = "cpu"
        try:
            _PADDLEOCR_READERS[language] = PaddleOCR(
                lang=paddle_lang,
                device=_paddle_device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            if _paddle_device == "cpu":
                raise
            print(f"  [PaddleOCR warn] {language}: GPU init failed ({exc}); falling back to CPU for this process")
            _paddle_device = "cpu"
            _PADDLEOCR_READERS[language] = PaddleOCR(
                lang=paddle_lang,
                device=_paddle_device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        print(f"  [Model D3] PaddleOCR device: {_paddle_device}")
        print(f"  [Model D3] PaddleOCR {paddle_lang}: LOADED")
    return _PADDLEOCR_READERS[language]


def _load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _csv_env(name: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]


def _extract_json_array(text: str) -> list[object]:
    import re

    stripped = str(text or "").strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", stripped)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _tighten_box_with_segmentation(box: Box, seg_mask: np.ndarray | None, img_w: int, img_h: int) -> Box:
    if seg_mask is None:
        return box
    roi = seg_mask[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return box
    ys, xs = np.nonzero(roi > 0)
    if len(xs) < 20:
        return box
    tightened = Box(
        max(0, box.x1 + int(xs.min()) - 2),
        max(0, box.y1 + int(ys.min()) - 2),
        min(img_w, box.x1 + int(xs.max()) + 3),
        min(img_h, box.y1 + int(ys.max()) + 3),
    )
    if tightened.width < 8 or tightened.height < 8:
        return box
    return tightened


def _region_too_art_heavy(image: np.ndarray, box: Box) -> bool:
    roi = image[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return True
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 80))
    pale_fraction = float(np.mean(gray > 235))
    return dark_fraction > 0.45 and pale_fraction < 0.28


def _gemini_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None = None,
) -> list[dict]:
    if os.environ.get("USE_API_VISION_OCR", "auto").strip().lower() in {"0", "false", "no", "off"}:
        return []
    _load_env_file()
    if not API_MANAGER.provider_keys("gemini") or not text_boxes:
        return []

    img_h, img_w = image.shape[:2]
    usable_boxes = []
    for box in text_boxes:
        tightened = _tighten_box_with_segmentation(box, seg_mask, img_w, img_h)
        if tightened.width < 8 or tightened.height < 8 or tightened.width * tightened.height < 80:
            continue
        if _region_too_art_heavy(image, tightened):
            continue
        usable_boxes.append(tightened)
    usable_boxes = usable_boxes[:24]
    if not usable_boxes:
        return []

    region_payload = [
        {"id": index, "box": [box.x1, box.y1, box.x2, box.y2]}
        for index, box in enumerate(usable_boxes)
    ]
    language_name = {"ko": "Korean", "ja": "Japanese"}.get(language_hint, "Chinese")
    ja_note = (
        " The Japanese may be hand-lettered and printed vertically; read each column "
        "top-to-bottom, columns right-to-left."
        if language_hint == "ja" else ""
    )
    prompt = (
        f"Image size is exactly {img_w}x{img_h} pixels. "
        f"OCR and translate only the boxed {language_name}/CJK text regions listed here: "
        f"{json.dumps(region_payload, ensure_ascii=False)}.{ja_note} "
        "Return valid JSON only in this exact shape: "
        "[{\"id\":0,\"source_text\":\"...\",\"english\":\"...\"}]. "
        "Omit regions that are not readable text. Do not add coordinates."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(os.environ.get("VISION_OCR_MAX_TOKENS", "2048")),
            "responseMimeType": "application/json",
        },
    }
    models = [
        model.strip()
        for model in os.environ.get(
            "GEMINI_VISION_OCR_MODELS",
            # gemini-flash-latest first: gemini-2.5-flash/-lite return 404
            # "no longer available to new users" for newer API keys/projects
            # (confirmed 2026-07-16); the -latest alias works across old and
            # new keys alike. See matching note in run_step7_translate.py.
            "gemini-flash-latest,gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash",
        ).split(",")
        if model.strip()
    ]

    last_error = None
    estimated_tokens = API_MANAGER.estimate_tokens(prompt, output_tokens=int(os.environ.get("VISION_OCR_MAX_TOKENS", "2048")))
    for model in models:
        attempted_hashes: set[str] = set()
        while True:
            try:
                lease = API_MANAGER.reserve_key("gemini", estimated_tokens, capability="vision_ocr")
            except (ApiProviderUnavailable, ApiProviderAuthLocked, ApiQuotaExhausted, ApiRateLimited) as error:
                last_error = str(error)
                break
            if lease.key_hash in attempted_hashes:
                break
            attempted_hashes.add(lease.key_hash)
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(lease.key)}"
            )
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=max(20, int(os.environ.get("VISION_OCR_TIMEOUT_SECONDS", "60")))) as response:
                    raw_payload = json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as error:
                last_error = f"HTTP {error.code}"
                API_MANAGER.mark_failure(lease, error.code, last_error)
                if API_MANAGER.is_terminal_quota_error(error.code, last_error) or error.code in {401, 402, 403, 429}:
                    continue
                break
            except Exception as error:
                last_error = str(error)[:120]
                API_MANAGER.mark_failure(lease, 0, last_error)
                break
            API_MANAGER.mark_success(lease, raw_payload)

            content = ""
            for candidate in raw_payload.get("candidates", []):
                parts = (candidate.get("content") or {}).get("parts", [])
                content += "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
            parsed = _extract_json_array(content)
            final_results = []
            seen_boxes: list[Box] = []
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                try:
                    region_id = int(entry.get("id"))
                except (TypeError, ValueError):
                    continue
                if region_id < 0 or region_id >= len(usable_boxes):
                    continue
                source_text = str(entry.get("source_text") or entry.get("text") or "").strip()
                english_text = str(entry.get("english") or entry.get("en_text") or entry.get("translation") or "").strip()
                if not _vision_source_script_ok(source_text, language_hint):
                    continue
                if not english_text or not any(ch.isalpha() for ch in english_text):
                    continue
                red_box = usable_boxes[region_id].expanded(4, img_w, img_h)
                if any(_boxes_overlap(red_box, existing) > 0.76 for existing in seen_boxes):
                    continue
                seen_boxes.append(red_box)
                green_box = red_box.expanded(max(8, min(24, int(min(red_box.width, red_box.height) * 0.12))), img_w, img_h)
                final_results.append(
                    {
                        "id": len(final_results),
                        "text": source_text,
                        "pretranslated_text": english_text,
                        "ocr_provider": f"gemini_vision_ocr:{model}",
                        "ocr_confidence": None,
                        "box": {k: int(v) for k, v in red_box.to_dict().items()},
                        "erase_boxes": [{k: int(v) for k, v in red_box.to_dict().items()}],
                        "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                        "green_polygon": [
                            [green_box.x1, green_box.y1],
                            [green_box.x2, green_box.y1],
                            [green_box.x2, green_box.y2],
                            [green_box.x1, green_box.y2],
                        ],
                        "route": "floating_dialogue",
                        "bubble_idx": -1,
                        "mask_mode": "stroke",
                        "fallback_source": "gemini_vision_region_ocr",
                        "force_bubble_cleanup": False,
                        "vision_rescue": True,
                    }
                )
            if final_results:
                print(f"  [vision-ocr] Gemini recovered {len(final_results)} regions with {model}")
                return final_results
            break
    if last_error:
        print(f"  [vision-ocr warn] Gemini region OCR failed: {last_error}")
    return []


def _openai_compatible_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None,
    provider: str,
    provider_label: str,
    endpoint: str,
    headers_factory,
    model_env_names: tuple[str, ...],
    default_models: list[str],
    max_tokens_field: str = "max_tokens",
) -> list[dict]:
    if os.environ.get("USE_API_VISION_OCR", "auto").strip().lower() in {"0", "false", "no", "off"}:
        return []
    _load_env_file()
    if not API_MANAGER.provider_keys(provider) or not text_boxes:
        return []

    img_h, img_w = image.shape[:2]
    usable_boxes = []
    for box in text_boxes:
        tightened = _tighten_box_with_segmentation(box, seg_mask, img_w, img_h)
        if tightened.width < 8 or tightened.height < 8 or tightened.width * tightened.height < 80:
            continue
        if _region_too_art_heavy(image, tightened):
            continue
        usable_boxes.append(tightened)
    usable_boxes = usable_boxes[:24]
    if not usable_boxes:
        return []

    region_payload = [
        {"id": index, "box": [box.x1, box.y1, box.x2, box.y2]}
        for index, box in enumerate(usable_boxes)
    ]
    language_name = {"ko": "Korean", "ja": "Japanese"}.get(language_hint, "Chinese")
    script_note = (
        "For hand-lettered or vertical Japanese text, read each column top-to-bottom, "
        "columns right-to-left, and preserve the source_text as read,"
        if language_hint == "ja" else
        "For mixed Hangul/Hanja or historical vertical Korean text, preserve the source_text as read,"
    )
    prompt = (
        f"Image size is exactly {img_w}x{img_h} pixels. "
        f"OCR and translate only the boxed {language_name}/CJK text regions listed here: "
        f"{json.dumps(region_payload, ensure_ascii=False)}. "
        "Return valid JSON only in this exact shape: "
        "[{\"id\":0,\"source_text\":\"...\",\"english\":\"...\"}]. "
        f"{script_note} "
        "then provide concise natural English. Omit unreadable text, SFX, logos, and artwork. "
        "Do not add coordinates."
    )
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    max_tokens = int(os.environ.get("VISION_OCR_MAX_TOKENS", "2048"))
    payload_template = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful manga/manhwa/manhua OCR and translation assistant. "
                    "You only return strict JSON and never invent unreadable text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        max_tokens_field: max_tokens,
    }
    model_env = next((os.environ.get(name, "").strip() for name in model_env_names if os.environ.get(name, "").strip()), "")
    models = [model.strip() for model in (model_env or ",".join(default_models)).split(",") if model.strip()]

    last_error = None
    estimated_tokens = API_MANAGER.estimate_tokens(prompt, output_tokens=max_tokens)
    for model in models:
        attempted_hashes: set[str] = set()
        while True:
            try:
                lease = API_MANAGER.reserve_key(provider, estimated_tokens, capability="vision_ocr")
            except (ApiProviderUnavailable, ApiProviderAuthLocked, ApiQuotaExhausted, ApiRateLimited) as error:
                last_error = str(error)
                break
            if lease.key_hash in attempted_hashes:
                break
            attempted_hashes.add(lease.key_hash)
            payload = dict(payload_template)
            payload["model"] = model
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers=headers_factory(lease.key),
            )
            try:
                with urllib.request.urlopen(request, timeout=max(20, int(os.environ.get("VISION_OCR_TIMEOUT_SECONDS", "60")))) as response:
                    raw_payload = json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = f"HTTP {error.code}: {body or error.reason}"
                API_MANAGER.mark_failure(lease, error.code, last_error)
                if API_MANAGER.is_terminal_quota_error(error.code, last_error) or error.code in {401, 402, 403, 429}:
                    continue
                break
            except Exception as error:
                last_error = str(error)[:180]
                API_MANAGER.mark_failure(lease, 0, last_error)
                break
            API_MANAGER.mark_success(lease, raw_payload)

            content_parts = []
            for choice in raw_payload.get("choices", []):
                message = choice.get("message") if isinstance(choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else ""
                if isinstance(content, str):
                    content_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            content_parts.append(str(part.get("text", "")))
            parsed = _extract_json_array("\n".join(content_parts))
            final_results = []
            seen_boxes: list[Box] = []
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                try:
                    region_id = int(entry.get("id"))
                except (TypeError, ValueError):
                    continue
                if region_id < 0 or region_id >= len(usable_boxes):
                    continue
                source_text = str(entry.get("source_text") or entry.get("text") or "").strip()
                english_text = str(entry.get("english") or entry.get("en_text") or entry.get("translation") or "").strip()
                if not _vision_source_script_ok(source_text, language_hint):
                    continue
                if not english_text or not any(ch.isalpha() for ch in english_text):
                    continue
                red_box = usable_boxes[region_id].expanded(4, img_w, img_h)
                if any(_boxes_overlap(red_box, existing) > 0.76 for existing in seen_boxes):
                    continue
                seen_boxes.append(red_box)
                green_box = red_box.expanded(max(8, min(24, int(min(red_box.width, red_box.height) * 0.12))), img_w, img_h)
                final_results.append(
                    {
                        "id": len(final_results),
                        "text": source_text,
                        "pretranslated_text": english_text,
                        "ocr_provider": f"{provider}_vision_ocr:{model}",
                        "ocr_confidence": None,
                        "box": {k: int(v) for k, v in red_box.to_dict().items()},
                        "erase_boxes": [{k: int(v) for k, v in red_box.to_dict().items()}],
                        "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                        "green_polygon": [
                            [green_box.x1, green_box.y1],
                            [green_box.x2, green_box.y1],
                            [green_box.x2, green_box.y2],
                            [green_box.x1, green_box.y2],
                        ],
                        "route": "floating_dialogue",
                        "bubble_idx": -1,
                        "mask_mode": "stroke",
                        "fallback_source": f"{provider}_vision_region_ocr",
                        "force_bubble_cleanup": False,
                        "vision_rescue": True,
                    }
                )
            if final_results:
                print(f"  [vision-ocr] {provider_label} recovered {len(final_results)} regions with {model}")
                return final_results
            break
    if last_error:
        print(f"  [vision-ocr warn] {provider_label} region OCR failed: {last_error}")
    return []


def _openrouter_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None = None,
) -> list[dict]:
    return _openai_compatible_vision_region_ocr(
        image_path,
        image,
        text_boxes,
        language_hint,
        seg_mask,
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1/chat/completions",
        lambda key: {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "http://127.0.0.1",
            "X-Title": "Free Manga Translator Step 5 Vision OCR",
        },
        ("OPENROUTER_VISION_OCR_MODELS", "OPENROUTER_VISION_OCR_MODEL"),
        ["qwen/qwen2.5-vl-72b-instruct:free", "qwen/qwen2.5-vl-72b-instruct", "google/gemini-2.5-flash"],
    )


def _groq_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None = None,
) -> list[dict]:
    return _openai_compatible_vision_region_ocr(
        image_path,
        image,
        text_boxes,
        language_hint,
        seg_mask,
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        lambda key: {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        ("GROQ_VISION_OCR_MODELS", "GROQ_VISION_OCR_MODEL"),
        ["meta-llama/llama-4-scout-17b-16e-instruct"],
        "max_completion_tokens",
    )


def _github_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None = None,
) -> list[dict]:
    return _openai_compatible_vision_region_ocr(
        image_path,
        image,
        text_boxes,
        language_hint,
        seg_mask,
        "github",
        "GitHub Models",
        "https://models.github.ai/inference/chat/completions",
        lambda key: {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        ("GITHUB_VISION_OCR_MODELS", "GITHUB_VISION_OCR_MODEL"),
        ["openai/gpt-4o-mini", "openai/gpt-4o"],
    )


def _nvidia_vision_region_ocr(
    image_path: Path,
    image: np.ndarray,
    text_boxes: list[Box],
    language_hint: str,
    seg_mask: np.ndarray | None = None,
) -> list[dict]:
    base_url = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    return _openai_compatible_vision_region_ocr(
        image_path,
        image,
        text_boxes,
        language_hint,
        seg_mask,
        "nvidia",
        "NVIDIA NIM",
        f"{base_url}/chat/completions",
        lambda key: {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        ("NVIDIA_NIM_VISION_OCR_MODELS", "NVIDIA_NIM_VISION_OCR_MODEL", "NVIDIA_VISION_OCR_MODELS", "NVIDIA_VISION_OCR_MODEL"),
        [
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "meta/llama-3.2-11b-vision-instruct",
            "meta/llama-3.2-90b-vision-instruct",
        ],
    )


def _script_count(text: str, script: str) -> int:
    import re

    patterns = {
        "hangul": r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]",
        "han": r"[\u3400-\u9fff]",
        "kana": r"[\u3040-\u30ff]",
    }
    return len(re.findall(patterns[script], text or ""))


def _vision_source_script_ok(text: str, language_hint: str) -> bool:
    # A vision-model transcription is only trusted as real source text if it
    # actually contains enough of the expected script -- guards against the
    # model hallucinating/echoing English or noise into source_text. ja uses
    # kana+han (a JA line can be almost entirely kanji, or almost entirely
    # kana); ko/ch keep their existing hangul+han / han-only bars unchanged.
    if language_hint == "ja":
        return _script_count(text, "kana") + _script_count(text, "han") >= 2
    return _script_count(text, "hangul") + _script_count(text, "han") >= 2


def _combine_easyocr_results(results: list, language: str, min_confidence: float | None = None) -> dict:
    if not results:
        return {"text": "", "confidence": 0.0}

    if min_confidence is None:
        # Was 0.35 for zh vs 0.55 for ko -- an asymmetry with no found justification (no
        # comment, no measured tuning note) alongside a zh rescue path that had never been
        # exercised on a real sample (see _paddleocr_reader/rescue changes above). Matched to
        # ko's stricter floor rather than assumed correct as-is.
        min_confidence = 0.55
    kept = []
    for polygon, text, confidence in results:
        clean_text = str(text or "").strip()
        if not clean_text or float(confidence) < min_confidence:
            continue
        if language == "ko" and _script_count(clean_text, "hangul") == 0:
            continue
        if _is_chinese_ocr_language(language) and _script_count(clean_text, "han") == 0:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        kept.append(
            {
                "text": clean_text,
                "confidence": float(confidence),
                "x": float(min(xs)),
                "y": float(min(ys)),
                "height": float(max(ys) - min(ys)),
            }
        )

    if not kept:
        return {"text": "", "confidence": 0.0}

    kept.sort(key=lambda item: (round(item["y"] / max(8.0, item["height"] * 0.7)), item["x"]))
    text = " ".join(item["text"] for item in kept).strip()
    weights = [max(1, len(item["text"])) for item in kept]
    confidence = sum(item["confidence"] * weight for item, weight in zip(kept, weights)) / sum(weights)
    return {"text": text, "confidence": confidence}


def _easyocr_rescue_variants(crop_rgb: np.ndarray) -> list[np.ndarray]:
    if crop_rgb.size == 0:
        return []
    variants = [crop_rgb]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    for scale in (2, 3):
        up_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(cv2.cvtColor(up_gray, cv2.COLOR_GRAY2RGB))
        sharpened = cv2.addWeighted(
            up_gray,
            1.45,
            cv2.GaussianBlur(up_gray, (0, 0), 1.0),
            -0.45,
            0,
        )
        variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        if min(up_gray.shape[:2]) >= 24:
            threshold = cv2.adaptiveThreshold(
                up_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            variants.append(cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB))
    return variants


def _read_easyocr_rescue(crop_rgb: np.ndarray, language: str) -> dict:
    script = "han" if _is_chinese_ocr_language(language) else "hangul"
    # Was 0.18 for zh vs 0.32 for ko -- same unexplained asymmetry as _combine_easyocr_results
    # above, matched to ko's stricter floor.
    min_confidence = 0.32
    best = {"text": "", "confidence": 0.0, "script_count": 0}
    reader = _easyocr_reader(language)

    for variant in _easyocr_rescue_variants(crop_rgb):
        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                batch_size=4,
                contrast_ths=0.01,
                adjust_contrast=1.0,
                text_threshold=0.25,
                low_text=0.1,
                link_threshold=0.2,
            )
        except Exception:
            continue
        combined = _combine_easyocr_results(results, language, min_confidence=min_confidence)
        text = str(combined.get("text") or "").strip()
        script_count = _script_count(text, script)
        if script_count == 0:
            continue
        score = (
            script_count,
            float(combined.get("confidence") or 0.0),
            len(text),
        )
        best_score = (
            int(best.get("script_count") or 0),
            float(best.get("confidence") or 0.0),
            len(str(best.get("text") or "")),
        )
        if score > best_score:
            best = {
                "text": text,
                "confidence": round(float(combined.get("confidence") or 0.0), 4),
                "script_count": script_count,
            }

    if not best["text"]:
        return {"text": "", "provider": f"easyocr_{language}_rescue", "confidence": 0.0}
    return {
        "text": best["text"],
        "provider": f"easyocr_{language}_rescue",
        "confidence": best["confidence"],
    }


def _crop_dark_flat_background(crop_bgr: np.ndarray, seg_crop: np.ndarray | None = None) -> bool:
    """True when a crop's non-glyph background is confidently dark and flat --
    a solid dark UI panel or reverse-polarity bubble fill, not textured dark
    art/shading. OCR here is tuned for dark-text-on-light; feeding it a crop
    like this backwards can produce plausible-looking garbage rather than an
    outright failure (verified: a dark promo banner with white knockout text
    OCR'd to fluent-looking nonsense instead of erroring, so a
    confidence/failure-gated retry can't catch it -- inversion has to be
    decided from the crop's own pixel statistics before the first attempt).
    Requires BOTH a dark majority and low variance among the dark pixels, so
    a real dark textured surface (hair, night sky, screentone) -- which
    varies a lot tonally even though its average is dark -- does not qualify.
    """
    if crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if seg_crop is not None and seg_crop.shape == gray.shape:
        background = gray[seg_crop <= 0]
    else:
        background = gray.reshape(-1)
    if background.size < 40:
        return False
    dark_pixels = background[background < 100]
    dark_fraction = dark_pixels.size / float(background.size)
    if dark_fraction < 0.45:
        return False
    return float(np.std(dark_pixels)) <= 18.0


class LocalCjkOcr:
    def __init__(self, manga_ocr_model, language: str | None):
        self.manga_ocr_model = manga_ocr_model
        self.language = language

    def __call__(self, pil_image):
        return self.read_pil(pil_image)["text"]

    def read_pil(self, pil_image, seg_crop: np.ndarray | None = None) -> dict:
        from PIL import Image

        global _MANGA_OCR_MODEL
        crop_rgb = np.array(pil_image.convert("RGB"))
        if not self.language:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            if _crop_dark_flat_background(crop_bgr, seg_crop):
                inverted = Image.fromarray(255 - crop_rgb)
                text = self.manga_ocr_model(inverted)
                return {"text": text, "provider": "manga_ocr_inverted", "confidence": None}
            text = self.manga_ocr_model(pil_image)
            return {"text": text, "provider": "manga_ocr", "confidence": None}

        def _run_easyocr(rgb_variant):
            try:
                results = _easyocr_reader(self.language).readtext(
                    rgb_variant,
                    detail=1,
                    paragraph=False,
                    batch_size=8,
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                    text_threshold=0.4,
                    low_text=0.2,
                    link_threshold=0.3,
                )
            except Exception as error:
                print(f"  [EasyOCR warn] {self.language}: {str(error)[:100]}")
                return None
            return _combine_easyocr_results(results, self.language)

        combined = _run_easyocr(crop_rgb)
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if _crop_dark_flat_background(crop_bgr, seg_crop):
            inverted_combined = _run_easyocr(255 - crop_rgb)
            if inverted_combined is not None and (
                combined is None
                or float(inverted_combined["confidence"]) > float(combined["confidence"])
            ):
                combined = inverted_combined
        if combined is None:
            return {"text": "", "provider": f"easyocr_{self.language}", "confidence": 0.0}

        script = "han" if _is_chinese_ocr_language(self.language) else "hangul"
        if _script_count(str(combined["text"]), script) == 0:
            rescued = _read_easyocr_rescue(crop_rgb, self.language)
            if rescued["text"]:
                return rescued
            if _is_chinese_ocr_language(self.language):
                # Was manga-ocr -- a JAPANESE model -- accepted at a hardcoded 0.42 confidence
                # just because its output happened to contain Han characters. Chinese text
                # misread as plausible-looking kanji passed this trivially, and unlike Korean's
                # rescue below (real PaddleOCR-ko, gated at real confidence), this path has zero
                # evidence it was ever exercised on a real Chinese sample: every zh runtime
                # sample in this repo's own fixture tree was produced by the primary
                # PaddleOCR-ch path succeeding, never this fallback. Mirrors ko's rescue pattern
                # instead: the request's own language via PaddleOCR, gated on the same real
                # confidence/script checks, not a borrowed model's guess. No landscape-only
                # restriction (unlike ko's, which exists specifically because tall/narrow
                # Korean crops belong to a separate upstream vertical-column pipeline -- no
                # equivalent constraint is documented for Chinese).
                try:
                    paddle_reader = _paddleocr_reader("ch")
                    paddle_results = paddle_reader.predict(crop_rgb[:, :, ::-1]) if paddle_reader is not None else []
                    paddle_text = ""
                    paddle_score = 0.0
                    for payload in paddle_results:
                        texts = payload.get("rec_texts") or []
                        scores = payload.get("rec_scores") or []
                        if texts:
                            paddle_text = str(texts[0] or "").strip()
                            paddle_score = float(scores[0]) if scores else 0.0
                            break
                except Exception as error:
                    print(f"  [PaddleOCR warn] Chinese rescue failed: {str(error)[:100]}")
                    paddle_text = ""
                    paddle_score = 0.0
                if _script_count(paddle_text, "han") > 0 and paddle_score >= 0.55:
                    return {
                        "text": paddle_text,
                        "provider": "paddleocr_ch_rescue",
                        "confidence": round(paddle_score, 4),
                    }
            elif self.language == "ko" and crop_rgb.shape[1] >= crop_rgb.shape[0]:
                # Landscape crops only (width >= height): a horizontal line
                # of modern Korean, matching the confirmed target case
                # (new_sample_14's "썸머스쿨" label, a 207x57 landscape
                # crop). Tall/narrow PORTRAIT crops belong to the separate
                # vertical-column Korean pipeline (external_ko_2's archaic-
                # script columns, e.g. 79x303) which already runs its own
                # dedicated PaddleOCR pass upstream and groups results by
                # provider-name uniformity -- verified directly that
                # rescuing individual sub-segments there with a different
                # provider string ("_rescue" vs the group's plain
                # "paddleocr_ko") broke that pass's grouping and produced
                # WORSE results (A/B tested: disabling this rescue restored
                # the original "paddleocr_ko_grouped" output exactly).
                # Landscape-only keeps this fix scoped to the failure mode
                # it was actually diagnosed against.
                try:
                    paddle_reader = _paddleocr_reader("ko")
                    paddle_results = paddle_reader.predict(crop_rgb[:, :, ::-1]) if paddle_reader is not None else []
                    paddle_text = ""
                    paddle_score = 0.0
                    for payload in paddle_results:
                        texts = payload.get("rec_texts") or []
                        scores = payload.get("rec_scores") or []
                        if texts:
                            paddle_text = str(texts[0] or "").strip()
                            paddle_score = float(scores[0]) if scores else 0.0
                            break
                except Exception as error:
                    print(f"  [PaddleOCR warn] Korean rescue failed: {str(error)[:100]}")
                    paddle_text = ""
                    paddle_score = 0.0
                if _script_count(paddle_text, "hangul") > 0 and paddle_score >= 0.55:
                    return {
                        "text": paddle_text,
                        "provider": "paddleocr_ko_rescue",
                        "confidence": round(paddle_score, 4),
                    }
        return {
            "text": combined["text"],
            "provider": f"easyocr_{self.language}",
            "confidence": round(float(combined["confidence"]), 4),
        }


def _read_ocr_crop(ocr_runtime, crop, seg_crop=None) -> dict:
    from PIL import Image

    if crop.size == 0:
        return {"text": "", "provider": "empty", "confidence": 0.0}
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if hasattr(ocr_runtime, "read_pil"):
        return ocr_runtime.read_pil(pil_crop, seg_crop)
    text = ocr_runtime(pil_crop)
    return {"text": text, "provider": "manga_ocr", "confidence": None}


def _map_paddle_box_to_original(box, angle, img_w: int, img_h: int) -> Box:
    x1, y1, x2, y2 = [float(v) for v in box]
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    try:
        angle = int(angle or 0) % 360
    except Exception:
        angle = 0

    mapped = []
    for x, y in points:
        if angle == 270:
            mapped.append((y, img_h - x))
        elif angle == 90:
            mapped.append((img_w - y, x))
        elif angle == 180:
            mapped.append((img_w - x, img_h - y))
        else:
            mapped.append((x, y))

    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return Box(
        max(0, int(round(min(xs)))),
        max(0, int(round(min(ys)))),
        min(img_w, int(round(max(xs)))),
        min(img_h, int(round(max(ys)))),
    )


def _expanded_vertical_layout_box(red_box: Box, img_w: int, img_h: int) -> Box:
    width = max(1, red_box.width)
    height = max(1, red_box.height)
    if height >= width * 1.7:
        target_width = min(img_w, max(width + 48, 96, int(height * 0.45)))
        target_height = min(img_h, height + max(18, int(height * 0.08)))
    else:
        target_width = min(img_w, max(width + 32, int(width * 1.35)))
        target_height = min(img_h, height + 20)

    center_x = (red_box.x1 + red_box.x2) // 2
    center_y = (red_box.y1 + red_box.y2) // 2
    x1 = max(0, min(img_w - target_width, center_x - target_width // 2))
    y1 = max(0, min(img_h - target_height, center_y - target_height // 2))
    return Box(int(x1), int(y1), int(x1 + target_width), int(y1 + target_height))


def _paddle_result_payload(result) -> dict:
    payload = getattr(result, "json", None)
    if isinstance(payload, dict):
        return payload.get("res", payload)
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return payload.get("res", payload)
    return {}


def _axis_tiles(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    if length <= tile_size:
        return [(0, length)]
    ranges: list[tuple[int, int]] = []
    stride = max(1, tile_size - overlap)
    start = 0
    while start < length:
        end = min(length, start + tile_size)
        ranges.append((start, end))
        if end >= length:
            break
        start = max(0, end - overlap)
        if ranges and start <= ranges[-1][0]:
            start = ranges[-1][0] + stride
    return ranges


def _paddle_page_payloads(reader, image_path: Path, image, label: str):
    img_h, img_w = image.shape[:2]
    try:
        max_side = max(1200, int(os.environ.get("PADDLE_OCR_MAX_TILE_SIDE", "3600")))
    except ValueError:
        max_side = 3600
    try:
        overlap = max(64, int(os.environ.get("PADDLE_OCR_TILE_OVERLAP", "180")))
    except ValueError:
        overlap = 180

    if max(img_w, img_h) <= max_side:
        try:
            for result in reader.predict(str(image_path)):
                yield _paddle_result_payload(result), 0, 0, img_w, img_h
        except Exception as error:
            print(f"  [PaddleOCR warn] {label}: {str(error)[:120]}")
        return

    x_tiles = _axis_tiles(img_w, max_side, overlap)
    y_tiles = _axis_tiles(img_h, max_side, overlap)
    print(f"  [paddle-{label}] tiled native page OCR {len(x_tiles) * len(y_tiles)} tiles max_side={max_side}")
    for x1, x2 in x_tiles:
        for y1, y2 in y_tiles:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tmp_name = ""
            try:
                with tempfile.NamedTemporaryFile(prefix=f"fmt_paddle_{label}_", suffix=".jpg", delete=False) as tmp:
                    tmp_name = tmp.name
                cv2.imwrite(tmp_name, crop)
                for result in reader.predict(tmp_name):
                    yield _paddle_result_payload(result), x1, y1, x2 - x1, y2 - y1
            except Exception as error:
                print(f"  [PaddleOCR warn] {label} tile {x1},{y1}: {str(error)[:120]}")
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass


def _boxes_overlap(left: Box, right: Box) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    smaller = min(max(1, left.width * left.height), max(1, right.width * right.height))
    return intersection / smaller


def _box_from_payload(payload: dict) -> Box:
    return Box(
        int(payload.get("x1", 0)),
        int(payload.get("y1", 0)),
        int(payload.get("x2", 0)),
        int(payload.get("y2", 0)),
    )


def _usable_cjk_text_count(items: list[dict], language: str | None) -> int:
    if language == "ko":
        scripts = ("hangul", "han")
    elif language in {"ch_tra", "ch_sim"}:
        scripts = ("han",)
    elif language == "ja":
        scripts = ("kana", "han")
    else:
        return sum(1 for item in items if str(item.get("text") or "").strip())
    count = 0
    for item in items:
        text = str(item.get("text") or "").strip()
        if sum(_script_count(text, script) for script in scripts) >= 2:
            count += 1
    return count


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _step1_detection_is_stale(image_path: Path, detect_dir: Path) -> bool:
    image_mtime = _safe_mtime(image_path)
    if image_mtime <= 0:
        return False
    required_outputs = [
        detect_dir / "detections.json",
        detect_dir / "seg_mask.png",
        detect_dir / "semantic_detections.json",
    ]
    if any(not output.exists() for output in required_outputs):
        return True
    if any(_safe_mtime(output) + 0.01 < image_mtime for output in required_outputs):
        return True
    bubble_outputs = sorted(detect_dir.glob("bubble_*.png"))
    return any(_safe_mtime(output) + 0.01 < image_mtime for output in bubble_outputs)


def _clear_stale_step1_outputs(detect_dir: Path) -> None:
    for pattern in ("bubble_*.png", "detections.json", "seg_mask.png", "semantic_detections.json"):
        for old_output in detect_dir.glob(pattern):
            try:
                old_output.unlink()
            except OSError:
                pass


def _box_mask_overlap_fraction(mask: np.ndarray, box: Box) -> float:
    if mask is None:
        return 0.0
    img_h, img_w = mask.shape[:2]
    x1 = max(0, min(img_w, int(box.x1)))
    y1 = max(0, min(img_h, int(box.y1)))
    x2 = max(0, min(img_w, int(box.x2)))
    y2 = max(0, min(img_h, int(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2] > 0
    return float(np.count_nonzero(roi)) / float(max(1, roi.size))


def _text_box_supports_bubble(mask_bounds: tuple[int, int, int, int], box: Box, overlap_fraction: float) -> bool:
    if overlap_fraction < 0.14:
        return False
    bx1, by1, bx2, by2 = mask_bounds
    mask_w = max(1, bx2 - bx1)
    mask_h = max(1, by2 - by1)
    mask_area = max(1, mask_w * mask_h)
    box_w = max(1, int(box.x2) - int(box.x1))
    box_h = max(1, int(box.y2) - int(box.y1))
    box_area_ratio = (box_w * box_h) / float(mask_area)
    box_aspect = box_w / float(box_h)
    horizontal_dialogue = (
        box_w >= max(120, int(mask_w * 0.44))
        and box_h <= max(int(mask_h * 0.78), 42)
        and box_aspect >= 1.9
    )
    vertical_dialogue = (
        box_h >= max(64, int(mask_h * 0.32))
        and box_w <= max(int(mask_w * 0.72), 54)
        and box_aspect <= 0.78
    )
    broad_text_block = (
        box_area_ratio >= 0.32
        and (box_w >= int(mask_w * 0.40) or box_h >= int(mask_h * 0.40))
    )
    return horizontal_dialogue or vertical_dialogue or broad_text_block


def _bubble_mask_has_dialogue_text_support(mask: np.ndarray, text_boxes: list[Box]) -> bool:
    bounds = _mask_bounds(mask)
    if bounds is None:
        return False
    bx1, by1, bx2, by2 = bounds
    mask_w = max(1, bx2 - bx1)
    mask_h = max(1, by2 - by1)
    mask_area = max(1, mask_w * mask_h)
    if np.count_nonzero(mask > 0) < max(120, int(mask_area * 0.18)):
        return False
    if not text_boxes:
        return True
    for box in text_boxes:
        overlap = _box_mask_overlap_fraction(mask, box)
        if _text_box_supports_bubble(bounds, box, overlap):
            return True
    return False


def _rewrite_bubble_mask_outputs(detect_dir: Path, bubble_masks: list[np.ndarray]) -> None:
    for old_output in detect_dir.glob("bubble_*.png"):
        try:
            old_output.unlink()
        except OSError:
            pass
    for i, mask in enumerate(bubble_masks):
        if mask is not None:
            cv2.imwrite(str(detect_dir / f"bubble_{i}.png"), mask)


def _filter_and_save_bubble_masks(
    detect_dir: Path,
    bubble_masks: list[np.ndarray],
    text_boxes: list[Box],
) -> list[np.ndarray]:
    if not bubble_masks:
        _rewrite_bubble_mask_outputs(detect_dir, [])
        return []
    kept = [
        mask
        for mask in bubble_masks
        if _bubble_mask_has_dialogue_text_support(mask, text_boxes)
    ]
    if text_boxes and kept and len(kept) < len(bubble_masks):
        print(f"  [bubble-filter] kept {len(kept)}/{len(bubble_masks)} dialogue-supported masks")
        bubble_masks = kept
    _rewrite_bubble_mask_outputs(detect_dir, bubble_masks)
    return bubble_masks


def _korean_script_chars(text: str) -> int:
    return _script_count(text, "hangul") + _script_count(text, "han")


def _korean_ocr_quality_score(items: list[dict]) -> float:
    if not items:
        return -1000.0
    score = 0.0
    usable_count = 0
    short_count = 0
    for item in items:
        text = str(item.get("text") or "").strip()
        chars = _korean_script_chars(text)
        if chars <= 0:
            score -= 1.0
            continue
        if chars == 1:
            short_count += 1
            score -= 0.35
            continue
        usable_count += 1
        score += min(12, chars) * 0.75
        score += min(4, len(text.split())) * 0.6
    score += usable_count * 1.5
    score -= max(0, len(items) - 12) * 0.35
    score -= short_count * 0.45
    return score


def _korean_ocr_is_fragmented(items: list[dict]) -> bool:
    korean_counts = [
        _korean_script_chars(str(item.get("text") or "").strip())
        for item in items
        if _korean_script_chars(str(item.get("text") or "").strip()) > 0
    ]
    if len(korean_counts) < 6:
        return False
    short_count = sum(1 for count in korean_counts if count <= 1)
    usable_count = sum(1 for count in korean_counts if count >= 2)
    short_ratio = short_count / max(1, len(korean_counts))
    return short_ratio >= 0.30 or (len(korean_counts) >= 10 and usable_count < len(korean_counts) * 0.62)


def _korean_paddle_grouping_is_better(current_items: list[dict], paddle_items: list[dict]) -> bool:
    if not paddle_items or not _korean_ocr_is_fragmented(current_items):
        return False
    current_counts = [
        _korean_script_chars(str(item.get("text") or "").strip())
        for item in current_items
        if _korean_script_chars(str(item.get("text") or "").strip()) > 0
    ]
    paddle_counts = [
        _korean_script_chars(str(item.get("text") or "").strip())
        for item in paddle_items
        if _korean_script_chars(str(item.get("text") or "").strip()) > 0
    ]
    if len(paddle_counts) < 2:
        return False
    current_avg = sum(current_counts) / max(1, len(current_counts))
    paddle_avg = sum(paddle_counts) / max(1, len(paddle_counts))
    grouped_enough = len(paddle_counts) <= max(4, int(len(current_counts) * 0.55))
    more_complete_lines = paddle_avg >= max(3.0, current_avg * 1.45)
    return grouped_enough and more_complete_lines


def _preserve_detection_metadata(vision_results: list[dict], detected_results: list[dict]) -> list[dict]:
    if not vision_results or not detected_results:
        return vision_results
    source_items = [
        item for item in detected_results
        if isinstance(item, dict) and isinstance(item.get("box"), dict)
    ]
    if not source_items:
        return vision_results
    preserved = []
    used_source_indexes: set[int] = set()
    for index, result in enumerate(vision_results):
        result_box = _box_from_payload(result.get("box") or {})
        best_idx = -1
        best_overlap = 0.0
        for candidate_idx, source in enumerate(source_items):
            if candidate_idx in used_source_indexes:
                continue
            overlap = _boxes_overlap(result_box, _box_from_payload(source["box"]))
            if overlap > best_overlap:
                best_idx = candidate_idx
                best_overlap = overlap
        merged = dict(result)
        if best_idx >= 0 and best_overlap >= 0.20:
            used_source_indexes.add(best_idx)
            source = source_items[best_idx]
            for key in (
                "box",
                "green_box",
                "green_polygon",
                "route",
                "bubble_idx",
                "mask_mode",
                "overlap_collision",
                "force_bubble_cleanup",
            ):
                if key in source:
                    merged[key] = source[key]
            if "erase_boxes" in source:
                merged["erase_boxes"] = source["erase_boxes"]
        merged["id"] = index
        preserved.append(merged)
    return preserved


def _vision_rescue_cjk_ocr(
    image_path: Path,
    image: np.ndarray,
    language: str | None,
    text_boxes: list[Box],
    seg_mask: np.ndarray | None,
    detected_results: list[dict],
) -> list[dict]:
    if language not in {"ko", "ch_tra", "ch_sim", "ja"}:
        return []
    if not text_boxes:
        return []
    # Ordered strongest-currently-healthy-first; a provider with revoked/
    # locked credentials just gets skipped for free at reserve_key() and
    # resumes automatically once keys are rotated (see api_manager.py).
    vision_providers = [
        item.strip().lower()
        for item in os.environ.get("VISION_OCR_PROVIDER_ORDER", "nvidia,groq,openrouter,gemini,github").split(",")
        if item.strip()
    ]
    vision_handlers = {
        "gemini": _gemini_vision_region_ocr,
        "github": _github_vision_region_ocr,
        "groq": _groq_vision_region_ocr,
        "nvidia": _nvidia_vision_region_ocr,
        "nvidia-nim": _nvidia_vision_region_ocr,
        "nim": _nvidia_vision_region_ocr,
        "openrouter": _openrouter_vision_region_ocr,
    }
    for provider in vision_providers:
        handler = vision_handlers.get(provider)
        if handler is None:
            continue
        rescued = handler(image_path, image, text_boxes, language, seg_mask)
        rescued = _preserve_detection_metadata(rescued, detected_results)
        for item in rescued:
            print(f"  [vision] {item['text'][:30]}...")
        if rescued:
            return rescued
    return []


# Process-lifetime hourly attempt budget for the full-page rescue specifically
# -- it's the most expensive vision call (whole-page image, more output
# tokens than a handful of boxed regions) and, since Task 3's blank-result
# cache fix means a page that still fails no longer gets served from cache,
# a chapter of genuinely undetectable pages re-requested repeatedly needs a
# hard ceiling independent of any single provider's own quota bookkeeping.
_VISION_FULL_PAGE_ATTEMPTS: list = []
_VISION_FULL_PAGE_ATTEMPTS_LOCK = threading.Lock()


def _vision_full_page_rescue_budget_ok() -> bool:
    limit = int(os.environ.get("VISION_RESCUE_MAX_PER_HOUR", "20"))
    if limit <= 0:
        return False
    now = time.monotonic()
    with _VISION_FULL_PAGE_ATTEMPTS_LOCK:
        while _VISION_FULL_PAGE_ATTEMPTS and now - _VISION_FULL_PAGE_ATTEMPTS[0] > 3600:
            _VISION_FULL_PAGE_ATTEMPTS.pop(0)
        if len(_VISION_FULL_PAGE_ATTEMPTS) >= limit:
            return False
        _VISION_FULL_PAGE_ATTEMPTS.append(now)
    return True


def _vision_full_page_image_payload(image: np.ndarray, max_dimension: int = 1344) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(h, w) or 1))
    resized = (
        cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _vision_full_page_rescue_request(
    provider: str, model: str, image_b64: str, prompt: str, img_w: int, img_h: int
) -> tuple[list[dict], str]:
    """Returns (parsed JSON array, error string). Never raises -- callers
    treat any exception-equivalent as a clean failure (empty list + reason)."""
    max_tokens = int(os.environ.get("VISION_OCR_MAX_TOKENS", "2048"))
    timeout = max(20, int(os.environ.get("VISION_OCR_TIMEOUT_SECONDS", "60")))
    estimated_tokens = API_MANAGER.estimate_tokens(prompt, output_tokens=max_tokens)
    try:
        lease = API_MANAGER.reserve_key(provider, estimated_tokens, capability="vision_ocr")
    except (ApiProviderUnavailable, ApiProviderAuthLocked, ApiQuotaExhausted, ApiRateLimited) as error:
        return [], str(error)

    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(lease.key)}"
        )
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"},
        }
        headers = {"Content-Type": "application/json"}
    else:
        endpoints = {
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "nvidia": os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/") + "/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "github": "https://models.github.ai/inference/chat/completions",
        }
        url = endpoints.get(provider, "")
        if not url:
            API_MANAGER.mark_failure(lease, 0, f"no full-page-rescue endpoint for provider {provider}")
            return [], f"unsupported provider {provider}"
        headers = {"Authorization": f"Bearer {lease.key}", "Content-Type": "application/json"}
        if provider == "github":
            headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        elif provider == "openrouter":
            headers.update({"HTTP-Referer": "http://127.0.0.1", "X-Title": "Free Manga Translator Step 5 Rescue"})
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful manga OCR assistant. You only return strict JSON and never invent unreadable text."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
        }

    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        last_error = f"HTTP {error.code}: {body or error.reason}"
        API_MANAGER.mark_failure(lease, error.code, last_error)
        return [], last_error
    except Exception as error:
        last_error = str(error)[:180]
        API_MANAGER.mark_failure(lease, 0, last_error)
        return [], last_error
    API_MANAGER.mark_success(lease, raw_payload)

    content = ""
    if provider == "gemini":
        for candidate in raw_payload.get("candidates", []):
            parts = (candidate.get("content") or {}).get("parts", [])
            content += "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    else:
        for choice in raw_payload.get("choices", []):
            message = choice.get("message") if isinstance(choice, dict) else None
            piece = message.get("content") if isinstance(message, dict) else ""
            if isinstance(piece, str):
                content += piece
            elif isinstance(piece, list):
                content += "\n".join(str(part.get("text", "")) for part in piece if isinstance(part, dict))
    return _extract_json_array(content), ""


def _vision_full_page_rescue(image_path: Path, image: np.ndarray, language: str) -> list[dict]:
    """Last-resort rescue for a CJK page where every existing OCR/consolidation/
    region-rescue path produced zero usable text -- asks a vision model to find
    its OWN text regions across the whole page, not just OCR boxes we already
    (and evidently unsuccessfully) detected. Any failure at any point returns
    [] so the caller is left with today's blank-page behavior; this can never
    make a page worse than it already is."""
    if os.environ.get("USE_API_VISION_OCR", "auto").strip().lower() in {"0", "false", "no", "off"}:
        return []
    if os.environ.get("VISION_FULL_PAGE_RESCUE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return []
    _load_env_file()
    if not _vision_full_page_rescue_budget_ok():
        print("  [vision-full-page] skipped: hourly rescue budget exhausted")
        return []

    img_h, img_w = image.shape[:2]
    image_b64 = _vision_full_page_image_payload(image)
    if not image_b64:
        return []

    language_name = {"ko": "Korean", "ja": "Japanese", "ch_tra": "Chinese", "ch_sim": "Chinese"}.get(language, "Chinese")
    prompt = (
        f"This is a {img_w}x{img_h} pixel manga/comic page. Find every distinct {language_name} "
        "dialogue or narration text block (speech bubbles, captions, hand-written dialogue). "
        "Exclude sound effects drawn as artwork, page numbers, watermarks, and signatures. "
        "Return STRICT JSON only, this exact shape: "
        "[{\"bbox\":[x1,y1,x2,y2],\"source_text\":\"...\",\"english\":\"...\"}], "
        "with bbox integers normalized to a 0-1000 scale (0,0 = top-left, 1000,1000 = bottom-right), "
        "x1<x2 and y1<y2. Return at most 12 regions, ordered by natural reading order. "
        "Return [] if no readable text is present."
    )

    vision_providers = [
        item.strip().lower()
        for item in os.environ.get("VISION_OCR_PROVIDER_ORDER", "nvidia,groq,openrouter,gemini,github").split(",")
        if item.strip()
    ]
    max_providers = max(1, int(os.environ.get("VISION_FULL_PAGE_MAX_PROVIDERS", "2")))
    model_env_by_provider = {
        "gemini": ("GEMINI_VISION_OCR_MODELS", "gemini-flash-latest"),
        "groq": ("GROQ_VISION_OCR_MODELS", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "nvidia": ("NVIDIA_NIM_VISION_OCR_MODELS", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"),
        "openrouter": ("OPENROUTER_VISION_OCR_MODELS", "qwen/qwen2.5-vl-72b-instruct"),
        "github": ("GITHUB_VISION_OCR_MODELS", "openai/gpt-4o-mini"),
    }

    attempts = 0
    for provider in vision_providers:
        if provider not in model_env_by_provider:
            continue
        if not API_MANAGER.provider_keys(provider):
            continue
        if attempts >= max_providers:
            break
        attempts += 1
        env_name, default_model = model_env_by_provider[provider]
        model = next(
            (m.strip() for m in os.environ.get(env_name, "").split(",") if m.strip()),
            default_model,
        )
        parsed, error = _vision_full_page_rescue_request(provider, model, image_b64, prompt, img_w, img_h)
        if error:
            print(f"  [vision-full-page warn] {provider}: {error}")
            continue

        final_results: list[dict] = []
        seen_boxes: list[Box] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            bbox = entry.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                nx1, ny1, nx2, ny2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            x1 = max(0, min(img_w, int(round(nx1 / 1000.0 * img_w))))
            y1 = max(0, min(img_h, int(round(ny1 / 1000.0 * img_h))))
            x2 = max(0, min(img_w, int(round(nx2 / 1000.0 * img_w))))
            y2 = max(0, min(img_h, int(round(ny2 / 1000.0 * img_h))))
            width, height = x2 - x1, y2 - y1
            if width < 12 or height < 12:
                continue
            if width * height > 0.35 * img_w * img_h:
                continue
            if width > 0.90 * img_w and height > 0.90 * img_h:
                continue
            source_text = str(entry.get("source_text") or entry.get("text") or "").strip()
            english_text = str(entry.get("english") or entry.get("en_text") or "").strip()
            if not _vision_source_script_ok(source_text, language):
                continue
            if not english_text or not any(ch.isalpha() for ch in english_text):
                continue
            candidate_box = Box(x1, y1, x2, y2)
            if any(_boxes_overlap(candidate_box, existing) > 0.65 for existing in seen_boxes):
                continue
            seen_boxes.append(candidate_box)
            if len(seen_boxes) > 12:
                break
            green_box = candidate_box.expanded(max(8, min(24, int(min(width, height) * 0.12))), img_w, img_h)
            final_results.append({
                "id": len(final_results),
                "text": source_text,
                "pretranslated_text": english_text,
                "ocr_provider": f"{provider}_vision_full_page:{model}",
                "ocr_confidence": None,
                "box": {k: int(v) for k, v in candidate_box.to_dict().items()},
                "erase_boxes": [{k: int(v) for k, v in candidate_box.to_dict().items()}],
                "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                "green_polygon": [
                    [green_box.x1, green_box.y1], [green_box.x2, green_box.y1],
                    [green_box.x2, green_box.y2], [green_box.x1, green_box.y2],
                ],
                "route": "floating_dialogue",
                "bubble_idx": -1,
                "mask_mode": "stroke",
                "fallback_source": f"{provider}_vision_full_page",
                "force_bubble_cleanup": False,
                "vision_rescue": True,
            })
        if final_results:
            print(f"  [vision-full-page] {provider} recovered {len(final_results)} regions with {model}")
            return final_results
    return []


def _expanded_xy(box: Box, pad_x: int, pad_y: int, img_w: int, img_h: int) -> Box:
    return Box(
        max(0, box.x1 - pad_x),
        max(0, box.y1 - pad_y),
        min(img_w, box.x2 + pad_x),
        min(img_h, box.y2 + pad_y),
    )


def _union_boxes(boxes: list[Box], img_w: int, img_h: int, pad: int = 0) -> Box:
    return Box(
        max(0, min(box.x1 for box in boxes) - pad),
        max(0, min(box.y1 for box in boxes) - pad),
        min(img_w, max(box.x2 for box in boxes) + pad),
        min(img_h, max(box.y2 for box in boxes) + pad),
    )


def _vertical_group_connected(left: Box, right: Box, img_w: int, img_h: int) -> bool:
    prospective = _union_boxes([left, right], img_w, img_h)
    if prospective.height > int(img_h * 0.72) and prospective.width > int(img_w * 0.24):
        return False

    center_gap_y = abs(((left.y1 + left.y2) / 2) - ((right.y1 + right.y2) / 2))
    if center_gap_y > img_h * 0.30:
        return False

    left_expanded = _expanded_xy(left, 8, 14, img_w, img_h)
    right_expanded = _expanded_xy(right, 8, 14, img_w, img_h)
    if _boxes_overlap(left_expanded, right_expanded) > 0:
        return True

    y_overlap = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    y_ratio = y_overlap / max(1, min(left.height, right.height))
    x_gap = max(0, max(left.x1, right.x1) - min(left.x2, right.x2))
    max_column_gap = max(14, int(img_w * 0.018))
    if prospective.width > int(img_w * 0.30):
        max_column_gap = min(max_column_gap, 16)
    return y_ratio >= 0.45 and x_gap <= max_column_gap


def _is_vertical_page_column(box: Box, img_h: int) -> bool:
    return box.height >= max(80, int(img_h * 0.18)) and box.height >= box.width * 4.0


def _split_overwide_vertical_group(
    members: list[dict],
    member_boxes: list[Box],
    img_w: int,
    img_h: int,
) -> list[list[tuple[dict, Box]]]:
    if len(members) < 4:
        return [list(zip(members, member_boxes))]

    vertical_boxes = [box for box in member_boxes if _is_vertical_page_column(box, img_h)]
    if len(vertical_boxes) < 4:
        return [list(zip(members, member_boxes))]

    union_box = _union_boxes(member_boxes, img_w, img_h)
    median_width = sorted(box.width for box in vertical_boxes)[len(vertical_boxes) // 2]
    if union_box.width <= max(90, int(img_w * 0.11), int(median_width * 2.9)):
        return [list(zip(members, member_boxes))]

    ordered = sorted(zip(members, member_boxes), key=lambda pair: pair[1].x1)
    chunk_size = 2 if len(ordered) <= 5 else 3
    chunks: list[list[tuple[dict, Box]]] = []
    for index in range(0, len(ordered), chunk_size):
        chunk = ordered[index:index + chunk_size]
        if chunks and len(chunk) == 1:
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)
    return chunks


def _group_paddle_page_items(items: list[dict], img_w: int, img_h: int) -> list[dict]:
    if not items:
        return []

    boxes = [
        Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])
        for item in items
    ]
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            if _vertical_group_connected(boxes[left_index], boxes[right_index], img_w, img_h):
                union(left_index, right_index)

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(items)):
        grouped_indices.setdefault(find(index), []).append(index)

    grouped_items = []
    for indices in grouped_indices.values():
        members = [items[index] for index in indices]
        member_boxes = [boxes[index] for index in indices]
        for chunk in _split_overwide_vertical_group(members, member_boxes, img_w, img_h):
            chunk_members = [member for member, _ in chunk]
            chunk_boxes = [box for _, box in chunk]
            if len(chunk_members) == 1:
                grouped_items.append(chunk_members[0])
                continue

            red_box = _union_boxes(chunk_boxes, img_w, img_h, pad=6)
            green_box = red_box.expanded(max(12, int(min(red_box.width, red_box.height) * 0.08)), img_w, img_h)
            ordered = sorted(chunk_members, key=lambda item: (item["box"]["x1"], item["box"]["y1"]))
            combined_text = " ".join(item["text"] for item in ordered).strip()
            confidence_values = [float(item.get("ocr_confidence") or 0.0) for item in chunk_members]
            confidence = sum(confidence_values) / max(1, len(confidence_values))
            erase_boxes = []
            for member in chunk_members:
                erase_boxes.extend(member.get("erase_boxes") or [member["box"]])
            providers = sorted({str(member.get("ocr_provider", "paddleocr")).removesuffix("_grouped") for member in chunk_members})
            grouped_provider = (
                f"{providers[0]}_grouped"
                if len(providers) == 1
                else "paddleocr_mixed_grouped"
            )
            grouped_items.append({
                "id": len(grouped_items),
                "text": combined_text,
                "ocr_provider": grouped_provider,
                "ocr_confidence": round(confidence, 4),
                "box": {k: int(v) for k, v in red_box.to_dict().items()},
                "erase_boxes": [
                    {k: int(v) for k, v in Box(box["x1"], box["y1"], box["x2"], box["y2"]).to_dict().items()}
                    for box in erase_boxes
                ],
                "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                "green_polygon": [
                    [green_box.x1, green_box.y1],
                    [green_box.x2, green_box.y1],
                    [green_box.x2, green_box.y2],
                    [green_box.x1, green_box.y2],
                ],
                "route": "floating_dialogue",
                "bubble_idx": -1,
                "mask_mode": "stroke",
                "fallback_source": "paddleocr_korean_grouped_page",
                "force_bubble_cleanup": False,
            })

    grouped_items.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(grouped_items):
        item["id"] = idx
    return grouped_items


def _fallback_korean_from_paddle(image_path: Path, image) -> list[dict]:
    img_h, img_w = image.shape[:2]
    final_results = []

    ocr_passes = [
        ("ko", "paddleocr_ko", "hangul", 2, 0.50),
        ("ch", "paddleocr_ch_mixed", "han", 2, 0.65),
    ]
    for language, provider, script, min_script_count, min_score in ocr_passes:
        pass_results = []
        pass_seen = set()
        reader = _paddleocr_reader(language)
        if reader is None:
            continue
        for payload, offset_x, offset_y, source_w, source_h in _paddle_page_payloads(reader, image_path, image, language):
            angle = (payload.get("doc_preprocessor_res") or {}).get("angle", 0)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            boxes = payload.get("rec_boxes") or []

            for text, score, box in zip(texts, scores, boxes):
                clean_text = str(text or "").strip()
                if not clean_text:
                    continue
                score = float(score or 0.0)
                script_hits = _script_count(clean_text, script)
                korean_geometry_hint = (
                    language == "ko"
                    and script_hits == 1
                    and score >= 0.08
                )
                if script_hits < min_script_count and not korean_geometry_hint:
                    continue
                if score < min_score and not korean_geometry_hint:
                    continue

                local_box = _map_paddle_box_to_original(box, angle, source_w, source_h)
                red_box = Box(
                    local_box.x1 + offset_x,
                    local_box.y1 + offset_y,
                    local_box.x2 + offset_x,
                    local_box.y2 + offset_y,
                ).expanded(4, img_w, img_h)
                if red_box.width < 8 or red_box.height < 8:
                    continue
                if any(_boxes_overlap(red_box, Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])) > 0.72 for item in pass_results):
                    continue
                key = (clean_text, red_box.x1 // 6, red_box.y1 // 6, red_box.x2 // 6, red_box.y2 // 6)
                if key in pass_seen:
                    continue
                pass_seen.add(key)

                green_box = _expanded_vertical_layout_box(red_box, img_w, img_h)
                pass_results.append({
                    "id": len(pass_results),
                    "text": clean_text,
                    "ocr_provider": provider,
                    "ocr_confidence": round(score, 4),
                    "box": {k: int(v) for k, v in red_box.to_dict().items()},
                    "erase_boxes": [{k: int(v) for k, v in red_box.to_dict().items()}],
                    "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                    "green_polygon": [
                        [green_box.x1, green_box.y1],
                        [green_box.x2, green_box.y1],
                        [green_box.x2, green_box.y2],
                        [green_box.x1, green_box.y2],
                    ],
                    "route": "floating_dialogue",
                    "bubble_idx": -1,
                    "mask_mode": "stroke",
                    "fallback_source": "paddleocr_korean_page",
                    "force_bubble_cleanup": False,
                })

        for item in _group_paddle_page_items(pass_results, img_w, img_h):
            item_box = Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])
            if any(
                _boxes_overlap(
                    item_box,
                    Box(existing["box"]["x1"], existing["box"]["y1"], existing["box"]["x2"], existing["box"]["y2"]),
                )
                > 0.72
                for existing in final_results
            ):
                continue
            final_results.append(item)

    final_results.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(final_results):
        item["id"] = idx
    return final_results


def _bubble_overlap_ratio(box: Box, bubble_mask) -> float:
    if bubble_mask is None or box.area <= 0:
        return 0.0
    roi = bubble_mask[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi > 0)) / float(max(1, box.area))


def _best_bubble_for_line(box: Box, bubble_masks) -> int:
    best_idx = -1
    best_ratio = 0.0
    for idx, bubble_mask in enumerate(bubble_masks):
        ratio = _bubble_overlap_ratio(box, bubble_mask)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx if best_ratio >= 0.18 else -1


def _weighted_confidence(lines: list[dict]) -> float:
    weights = [max(1, len(line["text"])) for line in lines]
    return sum(float(line["score"]) * weight for line, weight in zip(lines, weights)) / max(1, sum(weights))


def _sort_horizontal_lines(lines: list[dict]) -> list[dict]:
    if not lines:
        return []
    median_height = float(np.median([line["box"].height for line in lines])) if lines else 16.0
    row_unit = max(10.0, median_height * 0.75)
    return sorted(lines, key=lambda line: (round(line["box"].y1 / row_unit), line["box"].x1))


def _horizontal_lines_belong_together(previous: Box, current: Box) -> bool:
    max_height = max(previous.height, current.height)
    gap_y = current.y1 - previous.y2
    # Same-utterance line wrapping is packed tight (measured across every
    # ZH sample in the suite, both floating and bubble-assigned clusters:
    # every legitimate multi-line gap ratio is <= 0.143, most negative/
    # touching). A real speech-bubble/turn boundary between two lines that
    # merely happen to sit close together reads far looser -- new_sample_11's
    # "思賢早-" -> "昨天說的那個" gap ratio is 1.298, ~9x the highest
    # legitimate ratio observed -- so 1.35 was letting genuine bubble
    # boundaries bridge as if they were paragraph line-wraps. 0.5 keeps
    # comfortable margin above every observed legitimate case while clearly
    # excluding the boundary case.
    if gap_y > max(32, int(max_height * 0.5)):
        return False

    overlap_x = max(0, min(previous.x2, current.x2) - max(previous.x1, current.x1))
    min_width = max(1, min(previous.width, current.width))
    max_width = max(previous.width, current.width)
    left_aligned = abs(previous.x1 - current.x1) <= max(22, int(max_width * 0.18))
    center_aligned = abs(((previous.x1 + previous.x2) / 2) - ((current.x1 + current.x2) / 2)) <= max(
        34,
        int(max_width * 0.32),
    )
    strong_overlap = overlap_x >= int(min_width * 0.35)

    if gap_y <= max(8, int(max_height * 0.40)) and (strong_overlap or left_aligned):
        return True
    return strong_overlap or left_aligned or center_aligned


def _cluster_horizontal_cjk_lines(lines: list[dict]) -> list[list[dict]]:
    ordered = _sort_horizontal_lines(lines)
    clusters: list[list[dict]] = []
    for line in ordered:
        if not clusters:
            clusters.append([line])
            continue
        previous = clusters[-1][-1]["box"]
        current = line["box"]
        if _horizontal_lines_belong_together(previous, current):
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return clusters


def _should_merge_chinese_bubble_columns(cluster_boxes: list[Box], img_w: int, img_h: int) -> bool:
    if len(cluster_boxes) < 2:
        return False
    merged = _union_boxes(cluster_boxes, img_w, img_h, pad=0)
    if merged.width <= 0 or merged.height <= 0:
        return False
    vertical_columns = [box for box in cluster_boxes if box.height >= max(24, box.width * 1.6)]
    if len(vertical_columns) < 2:
        return False
    if merged.height < merged.width * 1.20:
        return False
    if merged.width > img_w * 0.28:
        return False
    ordered = sorted(cluster_boxes, key=lambda box: box.x1)
    gaps = [max(0, right.x1 - left.x2) for left, right in zip(ordered, ordered[1:])]
    median_width = float(np.median([box.width for box in cluster_boxes]))
    max_gap = max(gaps) if gaps else 0
    return max_gap <= max(72, int(median_width * 1.8))


def _paddle_chinese_page_lines(image_path: Path, image, bubble_masks) -> list[dict]:
    reader = _paddleocr_reader("ch")
    if reader is None:
        return []
    img_h, img_w = image.shape[:2]
    lines = []
    seen = set()
    for payload, offset_x, offset_y, source_w, source_h in _paddle_page_payloads(reader, image_path, image, "ch"):
        angle = (payload.get("doc_preprocessor_res") or {}).get("angle", 0)
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        boxes = payload.get("rec_boxes") or []
        for text, score, box in zip(texts, scores, boxes):
            clean_text = str(text or "").strip()
            if not clean_text:
                continue
            score = float(score or 0.0)
            if score < 0.42:
                continue
            han_count = _script_count(clean_text, "han")
            if han_count == 0:
                continue
            local_box = _map_paddle_box_to_original(box, angle, source_w, source_h)
            red_box = Box(
                local_box.x1 + offset_x,
                local_box.y1 + offset_y,
                local_box.x2 + offset_x,
                local_box.y2 + offset_y,
            ).expanded(3, img_w, img_h)
            if red_box.width < 6 or red_box.height < 6 or red_box.area < 45:
                continue
            key = (clean_text, red_box.x1 // 5, red_box.y1 // 5, red_box.x2 // 5, red_box.y2 // 5)
            if key in seen:
                continue
            seen.add(key)
            lines.append({
                "text": clean_text,
                "score": score,
                "box": red_box,
                "bubble_idx": _best_bubble_for_line(red_box, bubble_masks),
            })
    return lines


def _item_from_chinese_cluster(
    item_id: int,
    lines: list[dict],
    img_w: int,
    img_h: int,
    bubble_idx: int,
    zone: dict | None,
) -> dict:
    ordered = _sort_horizontal_lines(lines)
    line_boxes = [line["box"] for line in ordered]
    red_box = _union_boxes(line_boxes, img_w, img_h, pad=4)
    text = " ".join(line["text"] for line in ordered).strip()
    confidence = _weighted_confidence(ordered)
    if bubble_idx != -1 and zone is not None:
        green_box = zone["green_box"]
        green_polygon = zone["green_polygon"]
        route = "bubble_dialogue"
        mask_mode = "bubble_interior"
    else:
        green_box = red_box.expanded(max(10, int(min(red_box.width, red_box.height) * 0.18)), img_w, img_h)
        green_polygon = [
            [green_box.x1, green_box.y1],
            [green_box.x2, green_box.y1],
            [green_box.x2, green_box.y2],
            [green_box.x1, green_box.y2],
        ]
        route = "floating_dialogue"
        mask_mode = "stroke"
    return {
        "id": item_id,
        "text": text,
        "ocr_provider": "paddleocr_ch_page_clustered",
        "ocr_confidence": round(float(confidence), 4),
        "box": {k: int(v) for k, v in red_box.to_dict().items()},
        "erase_boxes": [{k: int(v) for k, v in box.to_dict().items()} for box in line_boxes],
        "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
        "green_polygon": green_polygon,
        "route": route,
        "bubble_idx": int(bubble_idx),
        "mask_mode": mask_mode,
        "fallback_source": "paddleocr_ch_page_clustered",
        "force_bubble_cleanup": False,
    }


def _fallback_chinese_from_paddle(image_path: Path, image, bubble_masks) -> list[dict]:
    lines = _paddle_chinese_page_lines(image_path, image, bubble_masks)
    if not lines:
        return []

    img_h, img_w = image.shape[:2]
    grouped: dict[int, list[dict]] = {}
    for line in lines:
        grouped.setdefault(int(line["bubble_idx"]), []).append(line)

    items: list[dict] = []
    next_id = 0
    for bubble_idx in sorted(idx for idx in grouped if idx != -1):
        clusters = _cluster_horizontal_cjk_lines(grouped[bubble_idx])
        cluster_boxes = [_union_boxes([line["box"] for line in cluster], img_w, img_h, pad=4) for cluster in clusters]
        if _should_merge_chinese_bubble_columns(cluster_boxes, img_w, img_h):
            merged_lines = [line for cluster in clusters for line in cluster]
            merged_box = _union_boxes(cluster_boxes, img_w, img_h, pad=4)
            zone = _bubble_cluster_zones([merged_box], bubble_masks[bubble_idx])[0]
            items.append(_item_from_chinese_cluster(next_id, merged_lines, img_w, img_h, bubble_idx, zone))
            next_id += 1
            continue
        zones = _bubble_cluster_zones(cluster_boxes, bubble_masks[bubble_idx])
        for cluster, zone in zip(clusters, zones):
            items.append(_item_from_chinese_cluster(next_id, cluster, img_w, img_h, bubble_idx, zone))
            next_id += 1

    floating_clusters = _cluster_horizontal_cjk_lines(grouped.get(-1, []))
    for cluster in floating_clusters:
        items.append(_item_from_chinese_cluster(next_id, cluster, img_w, img_h, -1, None))
        next_id += 1

    items.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(items):
        item["id"] = idx
    return items


def _glyph_count(text: str) -> int:
    return len("".join(str(text or "").split()))


def _merge_row_run(
    row_items: list[dict],
    image: np.ndarray,
    seg_mask: np.ndarray | None,
    ocr_runtime,
    img_w: int,
    img_h: int,
) -> list[dict]:
    # row_items all mutually satisfy the same-line y-overlap test already;
    # this chains left-to-right within the row on adjacency + scale, so a run
    # only spans items that are ALSO horizontally contiguous and similarly
    # sized (an unrelated same-height item far to the right of a real gap
    # must not be swept in).
    ordered = sorted(row_items, key=lambda it: it["box"]["x1"])
    out: list[dict] = []
    i = 0
    while i < len(ordered):
        run = [ordered[i]]
        j = i + 1
        while j < len(ordered):
            a = run[-1]["box"]
            b = ordered[j]["box"]
            a_h = a["y2"] - a["y1"]
            b_h = b["y2"] - b["y1"]
            shorter_h = min(a_h, b_h)
            if shorter_h <= 0:
                break
            if max(a_h, b_h) / shorter_h > 1.6:
                break
            run_heights = [r["box"]["y2"] - r["box"]["y1"] for r in run] + [b_h]
            median_h = float(np.median(run_heights))
            gap = max(0, b["x1"] - a["x2"])
            if gap > 0.8 * max(1, median_h):
                break
            run.append(ordered[j])
            j += 1
        i = j

        if len(run) < 2:
            out.append(run[0])
            continue

        texts = [str(it.get("text") or "") for it in run]
        glyph_counts = [_glyph_count(t) for t in texts]
        # A near-empty member (an OCR miss on a bridging glyph, e.g. the
        # missing "브" in a shattered "몰디브") is the actual fragmentation
        # signature. "3+ adjacent items all containing CJK text" was tried as
        # a second trigger but is NOT a reliable signal on its own: a normal
        # multi-column vertical-script passage (verified: external_ko_2, an
        # archaic Korean page) satisfies it too even though every fragment is
        # already complete, legitimate, separately-translatable text --
        # merging those consolidates them into one box that then gets
        # rejected whole by step 6's size gate, destroying content that was
        # fine before. Every fire this pass is meant to catch already carries
        # a genuine near-empty member (confirmed against both this session's
        # test cases and the master plan's own worked examples), so that
        # signal alone is sufficient and does not need this broader one.
        has_near_empty_fragment = any(gc <= 1 for gc in glyph_counts)
        if not has_near_empty_fragment:
            out.extend(run)
            continue

        boxes = [_box_from_payload(it["box"]) for it in run]
        merged_box = _union_boxes(boxes, img_w, img_h, pad=4)
        if merged_box.width > img_w * 0.97:
            out.extend(run)
            continue

        crop = image[merged_box.y1:merged_box.y2, merged_box.x1:merged_box.x2]
        seg_crop = (
            seg_mask[merged_box.y1:merged_box.y2, merged_box.x1:merged_box.x2]
            if seg_mask is not None
            else None
        )
        concatenated = " ".join(t.strip() for t in texts if t.strip())
        concat_glyphs = _glyph_count(concatenated)
        merged_text = concatenated
        merged_provider = "same_line_merge_concat"
        if crop.size > 0:
            reread = _read_ocr_crop(ocr_runtime, crop, seg_crop)
            reread_text = str(reread.get("text") or "")
            if reread_text.strip() and _glyph_count(reread_text) >= concat_glyphs * 0.6:
                merged_text = reread_text
                merged_provider = str(reread.get("provider") or "same_line_merge_reocr")

        green_box = merged_box.expanded(
            max(10, int(min(merged_box.width, merged_box.height) * 0.18)), img_w, img_h
        )
        merged_item = dict(run[0])
        merged_item["text"] = merged_text
        merged_item["ocr_provider"] = merged_provider
        merged_item["box"] = {k: int(v) for k, v in merged_box.to_dict().items()}
        merged_item["erase_boxes"] = [
            {k: int(v) for k, v in box.to_dict().items()} for box in boxes
        ]
        merged_item["green_box"] = {k: int(v) for k, v in green_box.to_dict().items()}
        merged_item["green_polygon"] = [
            [green_box.x1, green_box.y1],
            [green_box.x2, green_box.y1],
            [green_box.x2, green_box.y2],
            [green_box.x1, green_box.y2],
        ]
        existing_fallback = str(run[0].get("fallback_source") or "")
        merged_item["fallback_source"] = (
            f"{existing_fallback}+same_line_merge" if existing_fallback else "same_line_merge"
        )
        out.append(merged_item)
    return out


def _semantic_orphan_dialogue_rescue(
    semantic_result,
    final_results: list[dict],
    image: np.ndarray,
    seg_mask: np.ndarray | None,
    ocr_runtime,
    img_w: int,
    img_h: int,
) -> list[dict]:
    # step 1's semantic detector (magi) is a separate, independent pass from
    # the primary text detector whose seg_mask/detections.json drives the
    # main consolidation loop above -- it can flag a genuine dialogue region
    # that the primary detector missed entirely (zero seg_mask coverage), in
    # which case that region never becomes a candidate for OCR or rejection,
    # it simply never exists downstream (confirmed: new_sample_8_(ja)'s
    # "...でも" bubble, 88.8% bright interior, 6% dark ink -- a genuine
    # bubble signature -- had zero seg_mask pixels and an empty
    # rejected_layout_items.json). The two vision-rescue paths above exist
    # for this class of gap but are network-dependent and disabled offline
    # (USE_API_VISION_OCR=0); this is the offline, deterministic fallback.
    #
    # Gate (confidence>=0.55 AND bright_ratio>=0.70) measured against all 16
    # semantic-dialogue regions suite-wide that don't overlap an existing OCR
    # box: exactly 1 passes (the target, conf=0.64/bright=0.888); the
    # next-highest reject is conf=0.45. The false-positive class this must
    # exclude is the SAME one a prior session reverted for wiring this
    # detector into a broader gate -- SFX/decorative lettering on artwork
    # (e.g. new_sample_7_(ko)'s "ぷっ" at conf=0.09) -- which this gate
    # rejects on confidence alone. compact_text_len alone does NOT separate
    # these (SFX text can be 2+ chars); confidence+brightness together is the
    # actual discriminator here, not a redundant belt-and-suspenders.
    #
    # NOTE: the 0.55 cut is one-sample-supported (margin to next-highest
    # reject is 0.45, only one item sits above it). Do not loosen without a
    # fresh suite-wide sweep.
    existing_boxes = [
        _box_from_payload(item["box"])
        for item in final_results
        if isinstance(item, dict) and isinstance(item.get("box"), dict)
    ]
    next_id = max((int(item.get("id", -1)) for item in final_results), default=-1) + 1
    new_items: list[dict] = []
    for region in getattr(semantic_result, "regions", []):
        if str(getattr(region, "semantic_class", "")) != "dialogue":
            continue
        rbox = region.box
        rarea = max(1, rbox.width * rbox.height)
        overlap_px = 0
        for eb in existing_boxes:
            ix1, iy1 = max(rbox.x1, eb.x1), max(rbox.y1, eb.y1)
            ix2, iy2 = min(rbox.x2, eb.x2), min(rbox.y2, eb.y2)
            if ix2 > ix1 and iy2 > iy1:
                overlap_px += (ix2 - ix1) * (iy2 - iy1)
        if overlap_px / rarea >= 0.10:
            continue

        crop = image[rbox.y1:rbox.y2, rbox.x1:rbox.x2]
        if crop.size == 0:
            continue
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright_ratio = float(np.count_nonzero(gray_crop > 200)) / gray_crop.size
        confidence = float(getattr(region, "confidence", 0.0) or 0.0)
        if not (confidence >= 0.55 and bright_ratio >= 0.70):
            continue

        seg_crop = seg_mask[rbox.y1:rbox.y2, rbox.x1:rbox.x2] if seg_mask is not None else None
        ocr_meta = _read_ocr_crop(ocr_runtime, crop, seg_crop)
        ocr_text = str(ocr_meta.get("text") or "").strip()
        if not ocr_text:
            continue

        green_box = rbox.expanded(max(10, int(min(rbox.width, rbox.height) * 0.18)), img_w, img_h)
        new_items.append({
            "id": next_id,
            "text": ocr_text,
            "ocr_provider": ocr_meta.get("provider", "unknown"),
            "ocr_confidence": ocr_meta.get("confidence"),
            "box": {k: int(v) for k, v in rbox.to_dict().items()},
            "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
            "green_polygon": [
                [green_box.x1, green_box.y1],
                [green_box.x2, green_box.y1],
                [green_box.x2, green_box.y2],
                [green_box.x1, green_box.y2],
            ],
            "route": "floating_dialogue",
            "bubble_idx": -1,
            "mask_mode": "stroke",
            "overlap_collision": False,
            "fallback_source": "semantic_orphan_rescue",
            "force_bubble_cleanup": False,
        })
        print(
            f"  [semantic-orphan-rescue] {ocr_text[:30]}... "
            f"conf={confidence:.2f} bright={bright_ratio:.2f}"
        )
        next_id += 1
    return new_items


def _merge_same_line_ocr_fragments(
    final_results: list[dict],
    image: np.ndarray,
    seg_mask: np.ndarray | None,
    ocr_runtime,
    img_w: int,
    img_h: int,
) -> list[dict]:
    # A shattered CJK line (one empty/near-empty glyph-sized fragment bridging
    # two real fragments, or several tiny fragments of what is visually one
    # line) reads unreliably per-fragment but far better as one full-line crop.
    # Heal this at the source: merge same-line, same-surface runs and re-OCR
    # the union once. Conservative by design -- every gate below must hold, and
    # only runs carrying the fragmentation signature (a near-empty member, or
    # 3+ members forming a shattered CJK line) are merged; two healthy
    # neighboring text items are left for step 6's own merge pass.
    if len(final_results) < 2:
        return final_results

    groups: dict[int, list[dict]] = {}
    for item in final_results:
        groups.setdefault(int(item.get("bubble_idx", -1)), []).append(item)

    merged_out: list[dict] = []
    for bubble_idx, items in groups.items():
        if len(items) < 2:
            merged_out.extend(items)
            continue

        # Cluster into approximate rows by mutual (transitive) y-overlap
        # BEFORE any x1-based adjacency reasoning -- sorting the whole
        # bubble_idx group by x1 in one flat pass would interleave unrelated
        # items from other lines/panels that merely happen to share an x1
        # range with a real line, breaking adjacency detection for that line.
        n = len(items)
        heights = [it["box"]["y2"] - it["box"]["y1"] for it in items]
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a in range(n):
            for b in range(a + 1, n):
                ay1, ay2 = items[a]["box"]["y1"], items[a]["box"]["y2"]
                by1, by2 = items[b]["box"]["y1"], items[b]["box"]["y2"]
                shorter_h = min(heights[a], heights[b])
                if shorter_h <= 0:
                    continue
                # The height-ratio cap must hold here, not just in the later
                # chaining step: without it, one tall vertical text column
                # trivially clears 60% shorter-height overlap against every
                # small mark that falls within its span (a much smaller box
                # fully contained in a much taller one overlaps it at ~100%
                # of ITS OWN height almost by construction), transitively
                # chaining together unrelated small fragments -- verified
                # concretely on a page with a vertical JA dialogue column
                # bridging two unrelated single-kana boxes this way.
                if max(heights[a], heights[b]) / shorter_h > 1.6:
                    continue
                overlap = max(0, min(ay2, by2) - max(ay1, by1))
                if overlap / shorter_h >= 0.60:
                    union(a, b)

        rows: dict[int, list[int]] = {}
        for idx in range(n):
            rows.setdefault(find(idx), []).append(idx)

        for row_indices in rows.values():
            row_items = [items[k] for k in row_indices]
            merged_out.extend(
                _merge_row_run(row_items, image, seg_mask, ocr_runtime, img_w, img_h)
            )

    merged_out.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(merged_out):
        item["id"] = idx
    return merged_out


def _save_ocr_outputs(sample_path: Path, image: np.ndarray, final_results: list[dict]) -> None:
    out_dir = sample_path / "step_5_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    debug_img = image.copy()
    for res in final_results:
        rb, gb = res["box"], res["green_box"]
        poly = res.get("green_polygon", [])
        cv2.rectangle(debug_img, (rb["x1"], rb["y1"]), (rb["x2"], rb["y2"]), (0, 0, 255), 2)

        color = (0, 165, 255) if res.get("overlap_collision") else (0, 255, 0)
        if poly and len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(debug_img, (gb["x1"], gb["y1"]), (gb["x2"], gb["y2"]), color, 2)
    cv2.imwrite(str(out_dir / "debug_ocr_boxes.jpg"), debug_img)


def _best_bubble_for_box(box: Box, bubble_masks) -> int:
    best_idx = -1
    best_overlap = 0
    for idx, bubble_mask in enumerate(bubble_masks):
        roi = bubble_mask[box.y1:box.y2, box.x1:box.x2]
        overlap = int(np.count_nonzero(roi > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fallback_ocr_from_raw_detections(image, text_boxes, bubble_masks, ocr_model):
    if not text_boxes:
        return None

    img_h, img_w = image.shape[:2]
    usable_boxes = [
        box for box in text_boxes
        if box.width * box.height >= 80 and box.width >= 5 and box.height >= 5
    ]
    if not usable_boxes:
        return None

    x1 = max(0, min(box.x1 for box in usable_boxes) - 14)
    y1 = max(0, min(box.y1 for box in usable_boxes) - 14)
    x2 = min(img_w, max(box.x2 for box in usable_boxes) + 14)
    y2 = min(img_h, max(box.y2 for box in usable_boxes) + 14)
    union_box = Box(x1, y1, x2, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ocr_meta = _read_ocr_crop(ocr_model, crop)
    ocr_text = ocr_meta["text"]
    if classify_text_by_content(ocr_text) != "dialogue":
        return None

    bubble_idx = _best_bubble_for_box(union_box, bubble_masks)
    erase_box = union_box
    green_box = union_box.expanded(18, img_w, img_h)
    if bubble_idx != -1:
        bounds = _mask_bounds(bubble_masks[bubble_idx])
        if bounds:
            bx1, by1, bx2, by2 = bounds
            inset_x = max(2, int((bx2 - bx1) * 0.04))
            inset_y = max(2, int((by2 - by1) * 0.04))
            green_box = Box(
                max(0, bx1 + inset_x),
                max(0, by1 + inset_y),
                min(img_w, bx2 - inset_x),
                min(img_h, by2 - inset_y),
            )
            erase_box = green_box
    else:
        margin_x = max(4, int(img_w * 0.03))
        margin_y = max(4, int(img_h * 0.03))
        green_box = Box(margin_x, margin_y, img_w - margin_x, img_h - margin_y)
        erase_box = green_box

    return {
        "text": ocr_text,
        "box": erase_box,
        "green_box": green_box,
        "green_polygon": [
            [green_box.x1, green_box.y1],
            [green_box.x2, green_box.y1],
            [green_box.x2, green_box.y2],
            [green_box.x1, green_box.y2],
        ],
        "route": "bubble_dialogue" if bubble_idx != -1 else "floating_dialogue",
        "bubble_idx": bubble_idx,
        "mask_mode": "bubble_interior" if bubble_idx != -1 else "stroke",
        "force_bubble_cleanup": True,
        "ocr_provider": ocr_meta.get("provider", "unknown"),
        "ocr_confidence": ocr_meta.get("confidence"),
    }

def _run_step5_ocr_unlocked(sample_map: dict[str, str] | None = None, samples_dir: Path | None = None):
    global _MANGA_OCR_MODEL, _TEXT_HANDLE, _BUBBLE_MODEL, _BUBBLE_DEVICE, _SEMANTIC_HANDLE

    print("=" * 60)
    print("  Step 5 — OCR & Consolidation (v4)")
    print("=" * 60)

    cfg = MLConfig()
    samples_dir = Path(samples_dir) if samples_dir is not None else sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP
    resume_existing = os.environ.get("PIPELINE_RESUME_EXISTING_OCR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    def get_ocr_runtime(language: str | None):
        global _MANGA_OCR_MODEL
        if language:
            return LocalCjkOcr(None, language)
        if _MANGA_OCR_MODEL is None:
            _MANGA_OCR_MODEL = load_ocr_model(force_cpu=False)
        return LocalCjkOcr(_MANGA_OCR_MODEL, None)
    
    # Lazy load detection models only if needed
    text_handle = _TEXT_HANDLE
    bubble_model = _BUBBLE_MODEL
    bubble_device = _BUBBLE_DEVICE
    semantic_handle = _SEMANTIC_HANDLE

    for sample_name, img_file in sample_map.items():
        sample_path = samples_dir / sample_name
        img_path = sample_path / img_file
        if not img_path.exists(): continue
            
        print(f"\nProcessing {sample_name}")
        existing_ocr = sample_path / "step_5_ocr" / "ocr_results.json"
        if resume_existing and existing_ocr.exists():
            print(f"  [resume] existing Step 5 OCR kept: {existing_ocr}")
            continue
        sample_ocr_language = _sample_cjk_ocr_language(sample_name) if _local_cjk_mode() else None
        ocr_runtime = get_ocr_runtime(sample_ocr_language)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(
                f"cv2.imread() could not read {img_path} -- the file is missing, truncated, or "
                "not a valid image. This surfaces here as a clear error instead of the cryptic "
                "'NoneType' object has no attribute 'shape'."
            )
        h, w = image.shape[:2]
        
        # Check for Step 1-3 results
        detect_dir = sample_path / "step_1_detect"
        step1_res_path = detect_dir / "detections.json"
        seg_mask_path = detect_dir / "seg_mask.png"
        semantic_path = detect_dir / "semantic_detections.json"
        needs_step1_detection = (
            not step1_res_path.exists()
            or not seg_mask_path.exists()
            or not semantic_path.exists()
            or _step1_detection_is_stale(img_path, detect_dir)
        )
        
        if needs_step1_detection:
            if detect_dir.exists() and _step1_detection_is_stale(img_path, detect_dir):
                print("  Step 1 results stale for current input. Rerunning detection models...")
                _clear_stale_step1_outputs(detect_dir)
            else:
                print(f"  Step 1 results missing. Running detection models...")
            # Reload if ANY of the three is missing, not just text_handle -- a caller (e.g.
            # the runtime server's startup warmup) can leave a partial trio if it loaded text
            # detection/bubble segmentation successfully but semantic detection failed or
            # hadn't committed yet. Keying the reload off one handle as a proxy for "all
            # loaded" left that partial state permanently un-repaired: every request after it
            # would reuse the same stale text_handle (non-None) and crash on the still-None
            # semantic_handle instead of ever reloading it. This whole block already runs
            # inside _STEP5_RUN_LOCK (acquired by the public run_step5_ocr() wrapper before
            # calling this function), so this reload is already safely serialized against
            # concurrent requests and against the runtime server's own atomic warmup commit.
            if text_handle is None or bubble_model is None or semantic_handle is None:
                text_handle = load_text_model(cfg.text_model_path)
                bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path)
                semantic_handle = load_semantic_model("magi")
                _TEXT_HANDLE = text_handle
                _BUBBLE_MODEL = bubble_model
                _BUBBLE_DEVICE = bubble_device
                _SEMANTIC_HANDLE = semantic_handle
            
            detect_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Detect
            text_result = detect_text(text_handle, image, cfg)
            cv2.imwrite(str(seg_mask_path), text_result.seg_mask)
            with open(step1_res_path, 'w') as f:
                json.dump({"boxes": [{k: int(v) for k, v in b.to_dict().items()} for b in text_result.boxes]}, f)
                
            # Step 2: Bubble & Semantic
            bubble_masks = detect_bubbles(bubble_model, bubble_device, image, cfg)
            bubble_count_raw = len(bubble_masks)
            bubble_masks = _filter_and_save_bubble_masks(detect_dir, bubble_masks, text_result.boxes)
            # Was silent -- a pasted backend.log had text-region and classification detail
            # (steps 5/6 already print plenty) but nothing at all for step 1/2 detection, so a
            # detection MISS (as opposed to a classification/OCR miss) was undiagnosable from
            # logs alone. text_result.boxes is step 1's independent text-region detector, not
            # part of this bubble count, but printing both together is what actually answers
            # "did detection see anything on this page at all."
            print(f"  [detect] text_boxes={len(text_result.boxes)} bubbles_raw={bubble_count_raw} bubbles_kept={len(bubble_masks)}")

            semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
            with open(semantic_path, 'w') as f:
                json.dump({"regions": [{"box": {k: int(v) for k, v in r.box.to_dict().items()}, "class_id": int(r.class_id), 
                                     "raw_class_name": str(r.raw_class_name), "semantic_class": str(r.semantic_class),
                                     "action": str(r.action), "confidence": float(r.confidence)} for r in semantic_result.regions]}, f)
        else:
            # Load existing Step 1 results
            from ml_region_lib import TextDetectionResult, SemanticDetectionResult, SemanticTextRegion
            seg_mask = cv2.imread(str(seg_mask_path), cv2.IMREAD_GRAYSCALE)
            with open(step1_res_path, 'r') as f:
                d1 = json.load(f)
                text_result = TextDetectionResult(boxes=[Box(b["x1"], b["y1"], b["x2"], b["y2"]) for b in d1["boxes"]], seg_mask=seg_mask)
            with open(semantic_path, 'r') as f:
                d2 = json.load(f)
                semantic_result = SemanticDetectionResult(regions=[SemanticTextRegion(box=Box(r["box"]["x1"], r["box"]["y1"], r["box"]["x2"], r["box"]["y2"]), **{k:v for k,v in r.items() if k!="box"}) for r in d2["regions"]])
            bubble_masks = []
            for i in range(100):
                bm_p = detect_dir / f"bubble_{i}.png"
                if bm_p.exists(): bubble_masks.append(cv2.imread(str(bm_p), cv2.IMREAD_GRAYSCALE))
                else: break
            bubble_masks = _filter_and_save_bubble_masks(detect_dir, bubble_masks, text_result.boxes)

        if sample_ocr_language == "ch_tra":
            final_results = _fallback_chinese_from_paddle(img_path, image, bubble_masks)
            if final_results:
                print(f"  [paddle-ch] using {len(final_results)} clustered page OCR regions")
                _save_ocr_outputs(sample_path, image, final_results)
                continue
        
        # 1. Routing
        routed = build_step2_routing_state(text_result, semantic_result, bubble_masks, cfg, w, h, ocr_runtime, image)
        
        # 2. CONSOLIDATION (Group-by-Bubble)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        consolidated = consolidate_by_bubble(routed, text_result.seg_mask, bubble_masks, cfg, gray_img)
        
        # 3. OCR on Consolidated Red Boxes
        final_results = []
        for idx, ct in enumerate(consolidated):
            if ct.route_state == "onomatopoeia": continue
            
            crop = image[ct.box.y1:ct.box.y2, ct.box.x1:ct.box.x2]
            if crop.size == 0: continue

            seg_crop = text_result.seg_mask[ct.box.y1:ct.box.y2, ct.box.x1:ct.box.x2]
            ocr_meta = _read_ocr_crop(ocr_runtime, crop, seg_crop)
            ocr_text = ocr_meta["text"]
            
            final_results.append({
                "id": idx,
                "text": ocr_text,
                "ocr_provider": ocr_meta.get("provider", "unknown"),
                "ocr_confidence": ocr_meta.get("confidence"),
                "box": {k: int(v) for k, v in ct.box.to_dict().items()},
                "green_box": {k: int(v) for k, v in ct.expanded_box.to_dict().items()},
                "green_polygon": getattr(ct, "green_polygon", []),
                "route": ct.route_state,
                "bubble_idx": int(ct.bubble_idx),
                "mask_mode": ct.mask_mode,
                "overlap_collision": getattr(ct, "overlap_collision", False)
            })

            print(f"  [{idx}] {ocr_text[:30]}...")

        consolidation_produced_zero = not final_results
        if not final_results:
            fallback = _fallback_ocr_from_raw_detections(
                image, text_result.boxes, bubble_masks, ocr_runtime
            )
            if fallback:
                rb = fallback["box"]
                gb = fallback["green_box"]
                final_results.append({
                    "id": 0,
                    "text": fallback["text"],
                    "box": {k: int(v) for k, v in rb.to_dict().items()},
                    "green_box": {k: int(v) for k, v in gb.to_dict().items()},
                    "green_polygon": fallback["green_polygon"],
                    "route": fallback["route"],
                    "bubble_idx": int(fallback["bubble_idx"]),
                    "mask_mode": fallback["mask_mode"],
                    "overlap_collision": False,
                    "fallback_source": "raw_detection_union",
                    "force_bubble_cleanup": fallback.get("force_bubble_cleanup", False),
                    "ocr_provider": fallback.get("ocr_provider", "unknown"),
                    "ocr_confidence": fallback.get("ocr_confidence"),
                })
                print(f"  [fallback] {fallback['text'][:30]}...")

        if not final_results and sample_ocr_language == "ko":
            final_results = _fallback_korean_from_paddle(img_path, image)
            for item in final_results:
                print(f"  [paddle] {item['text'][:30]}...")

        if sample_ocr_language == "ko" and _korean_ocr_is_fragmented(final_results):
            current_score = _korean_ocr_quality_score(final_results)
            paddle_results = _fallback_korean_from_paddle(img_path, image)
            paddle_score = _korean_ocr_quality_score(paddle_results)
            if paddle_results and (
                paddle_score > current_score + 2.0
                or _korean_paddle_grouping_is_better(final_results, paddle_results)
            ):
                print(
                    f"  [paddle-ko] replacing fragmented OCR "
                    f"items={len(final_results)}->{len(paddle_results)} "
                    f"score={current_score:.1f}->{paddle_score:.1f}"
                )
                final_results = paddle_results
                for item in final_results:
                    print(f"  [paddle-ko] {item['text'][:30]}...")

        # A sample with no explicit zh/ko marker in its name defaults to
        # Japanese -- the pipeline's implicit primary target language (see
        # _sample_cjk_ocr_language's own docstring-equivalent comment above;
        # it only ever distinguishes ko/zh from "everything else"). This
        # covers runtime_ja_* live-server samples the same way ko/zh runtime
        # samples are already covered, without touching ko/zh resolution at
        # all. _local_cjk_mode() also being true for the offline validation
        # suite is harmless: USE_API_VISION_OCR=0 there blocks every vision
        # handler's own network call regardless of what rescue_language is.
        rescue_language = sample_ocr_language or ("ja" if _local_cjk_mode() else None)
        # Manga-ocr garble on the raw-detection-union fallback above can
        # coincidentally contain >=2 kana characters and pass the plain
        # usable-count check below even though it's junk, not real dialogue
        # -- so for ja specifically also trust the upstream signal directly:
        # no bubble detected AND consolidation itself produced nothing is
        # the established zero-bubble-detector failure mode (see
        # DEVELOPMENT_NOTES), and should always get a rescue attempt.
        ja_degenerate = (
            rescue_language == "ja" and not bubble_masks and consolidation_produced_zero
        )
        if rescue_language and (
            _usable_cjk_text_count(final_results, rescue_language) == 0 or ja_degenerate
        ):
            detected_boxes = [
                _box_from_payload(item["box"])
                for item in final_results
                if isinstance(item, dict) and isinstance(item.get("box"), dict)
            ]
            rescue_boxes = text_result.boxes if ja_degenerate else (detected_boxes or text_result.boxes)
            # In the degenerate case the only "detected" item is the raw
            # union fallback's own single garbled entry, whose
            # force_bubble_cleanup/route metadata would corrupt the first
            # rescued region if _preserve_detection_metadata matched it --
            # pass no source items to preserve-from in that case.
            rescue_sources = [] if ja_degenerate else final_results
            rescued = _vision_rescue_cjk_ocr(
                img_path,
                image,
                rescue_language,
                rescue_boxes,
                text_result.seg_mask,
                rescue_sources,
            )
            if rescued:
                final_results = rescued

        # Last resort: every existing path (local OCR, the region-boxed
        # vision rescue above) still produced zero usable text. Ask a vision
        # model to find its own text regions across the whole page instead
        # of relying on boxes we already (and evidently unsuccessfully)
        # detected -- this is the only path that can rescue a page where
        # detection itself found nothing at all to hand a box-based rescue.
        vision_rescue_meta: dict[str, object] = {"attempted": False}
        if rescue_language and _usable_cjk_text_count(final_results, rescue_language) == 0:
            full_page_rescued = _vision_full_page_rescue(img_path, image, rescue_language)
            vision_rescue_meta = {
                "attempted": True,
                "mode": "full_page",
                "language": rescue_language,
                "succeeded": bool(full_page_rescued),
                "regions": len(full_page_rescued),
            }
            if full_page_rescued:
                final_results = full_page_rescued
                vision_rescue_meta["provider"] = full_page_rescued[0].get("ocr_provider", "")
        try:
            step5_dir = sample_path / "step_5_ocr"
            step5_dir.mkdir(parents=True, exist_ok=True)
            (step5_dir / "vision_rescue_meta.json").write_text(
                json.dumps(vision_rescue_meta, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

        orphan_rescued = _semantic_orphan_dialogue_rescue(
            semantic_result, final_results, image, text_result.seg_mask, ocr_runtime, w, h
        )
        if orphan_rescued:
            final_results = final_results + orphan_rescued

        final_results = _merge_same_line_ocr_fragments(
            final_results, image, text_result.seg_mask, ocr_runtime, w, h
        )

        _save_ocr_outputs(sample_path, image, final_results)


def run_step5_ocr(sample_map: dict[str, str] | None = None, samples_dir: Path | None = None):
    acquired = _STEP5_RUN_LOCK.acquire(blocking=False)
    if not acquired:
        print("  [step5-lock] waiting for active OCR/detection pass to finish")
        _STEP5_RUN_LOCK.acquire()
    try:
        return _run_step5_ocr_unlocked(sample_map=sample_map, samples_dir=samples_dir)
    finally:
        _STEP5_RUN_LOCK.release()

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    run_step5_ocr()
