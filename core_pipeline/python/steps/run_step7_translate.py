"""
Step 7 — Contextual LLM Translation
===================================
Takes OCR results from Step 5 and translates Japanese, Korean, or Chinese
text fragments to concise natural English.

Translation approach:
  - Provider LLM translation first, using configured API keys and model fallback.
  - Optional local NLLB fallback only when `LOCAL_NLLB_TRANSLATION=1`.
  - No curated per-sample translation database in the production path.
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
import html
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from api_manager import (
    API_MANAGER,
    DAILY_LIMIT_MESSAGE,
    ApiProviderAuthLocked,
    ApiProviderUnavailable,
    ApiQuotaExhausted,
    ApiRateLimited,
)
from ml_region_lib import SAMPLE_MAP
from pipeline_paths import DEFAULT_SAMPLES_ROOT, PROJECT_ROOT, sample_root_from_env

_LOCAL_TRANSLATOR = None
_LOCAL_TOKENIZERS: dict[str, object] = {}
_LOCAL_TRANSLATOR_MODEL = os.environ.get("LOCAL_TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M").strip() or "facebook/nllb-200-distilled-600M"
# Guards the check-then-act singleton loads below against concurrent cold-start requests (the GPU
# scheduler allows 2-4 concurrent pipeline runs; see the matching lock in run_step4_inpaint.py).
_LOCAL_TRANSLATOR_LOAD_LOCK = threading.Lock()
ENV_FILE = PROJECT_ROOT / ".env"
API_PROVIDER_FAILURES: list[dict[str, object]] = []
API_PROVIDER_USES: list[dict[str, object]] = []
API_PROVIDER_DISABLED: dict[str, str] = {}
API_PROVIDER_KEY_CURSOR: dict[str, int] = {}
API_PROVIDER_KEY_DISABLED: dict[str, dict[int, str]] = {}


def _load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

API_TIMEOUT_SECONDS = max(5, int(os.environ.get("API_TRANSLATION_TIMEOUT_SECONDS", "45")))
API_RETRIES_PER_PROVIDER = max(1, int(os.environ.get("API_TRANSLATION_RETRIES_PER_PROVIDER", "2")))


def _configured(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped and "YOUR_" not in stripped and not re.fullmatch(r"[_\-\s]*", stripped))


def _csv_env(name: str) -> list[str]:
    values: list[str] = []
    candidate_names = [name]
    candidate_names.extend(f"{name}_{index}" for index in range(1, 51))
    for candidate_name in candidate_names:
        for part in os.environ.get(candidate_name, "").split(","):
            value = part.strip()
            if _configured(value) and value not in values:
                values.append(value)
    return values


def _csv_values_from_env(*names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        for part in os.environ.get(name, "").split(","):
            value = part.strip()
            if value and "YOUR_" not in value and value not in values:
                values.append(value)
    return values


def _model_candidates(env_names: tuple[str, ...], defaults: list[str]) -> list[str]:
    configured_models = _csv_values_from_env(*env_names)
    return configured_models or defaults


def _provider_keys(*names: str) -> list[str]:
    keys: list[str] = []
    for name in names:
        for key in _csv_env(name):
            if key not in keys:
                keys.append(key)
    return keys


SECRET_ENV_NAMES = {
    "GEMINI_API_KEYS",
    "GITHUB_API_KEYS",
    "GITHUB_API_KEY",
    "GROQ_API_KEYS",
    "GROQ_API_KEY",
    "MISTRAL_API_KEYS",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEYS",
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEYS",
    "CEREBRAS_API_KEY",
    "CEREBERAS_API_KEYS",
    "CEREBERAS_API_KEY",
    "NVIDIA_API_KEYS",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEYS",
    "NVIDIA_NIM_API_KEY",
    "FIREWORKS_API_KEYS",
    "FIREWORKS_API_KEY",
    "CLOUDFLARE_WORKERS_API_KEYS",
    "CLOUDFLARE_WORKERS_API_KEY",
    "CLOUDFLARE_API_KEYS",
    "CLOUDFLARE_API_KEY",
}
SECRETS_TO_SCRUB: list[str] = []
for secret_name in SECRET_ENV_NAMES:
    raw_secret = os.environ.get(secret_name, "")
    if _configured(raw_secret):
        SECRETS_TO_SCRUB.append(raw_secret)
        SECRETS_TO_SCRUB.extend(part.strip() for part in raw_secret.split(",") if part.strip())


def _scrub_secret(text: Any) -> str:
    rendered = str(text)
    for secret in sorted(set(SECRETS_TO_SCRUB), key=len, reverse=True):
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
    rendered = re.sub(r"Bearer\s+[A-Za-z0-9_\.\-]+", "Bearer [REDACTED]", rendered)
    rendered = re.sub(r"key=([^&\s]+)", "key=[REDACTED]", rendered)
    return rendered[:1000]


def _has_api_keys() -> bool:
    return any(
        [
            API_MANAGER.provider_keys("gemini"),
            API_MANAGER.provider_keys("github"),
            API_MANAGER.provider_keys("groq"),
            API_MANAGER.provider_keys("mistral"),
            API_MANAGER.provider_keys("openrouter"),
            API_MANAGER.provider_keys("cerebras"),
            API_MANAGER.provider_keys("nvidia"),
            API_MANAGER.provider_keys("fireworks"),
            API_MANAGER.provider_keys("cloudflare"),
        ]
    )


def _api_translation_enabled() -> bool:
    value = os.environ.get("USE_API_TRANSLATION", "auto").strip().lower()
    if value in {"0", "false", "no", "off", "local"}:
        return False
    if value in {"1", "true", "yes", "on", "api", "auto"}:
        return _has_api_keys()
    return _has_api_keys()


def _translation_items_with_layout_merges(ocr_results: list[dict], sample_dir: Path) -> list[dict]:
    layout_path = sample_dir / "step_6_layout" / "layout_constraints.json"
    if not layout_path.exists():
        return ocr_results
    try:
        layout_data = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ocr_results
    if not isinstance(layout_data, list):
        return ocr_results

    # Step 6 records a semantic_role ("dialogue"/"sfx"/...) per constraint;
    # carry it onto every OCR item so translation/typesetting downstream can
    # give SFX text a different (punchier, non-sentence) treatment instead
    # of translating and rendering it identically to ordinary dialogue.
    # short_fragment similarly flags a rescued tiny particle box (see
    # run_step6_layout.py) so its translation stays proportionally short.
    role_by_id: dict[int, str] = {}
    short_fragment_ids: set[int] = set()
    for layout in layout_data:
        if not isinstance(layout, dict):
            continue
        try:
            layout_id = int(layout.get("id"))
        except (TypeError, ValueError):
            continue
        role_by_id[layout_id] = str(layout.get("semantic_role") or "")
        if layout.get("short_fragment"):
            short_fragment_ids.add(layout_id)
    for item in ocr_results:
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        role = role_by_id.get(item_id)
        if role:
            item["semantic_role"] = role
        if item_id in short_fragment_ids:
            item["short_fragment"] = True

    original_by_id: dict[int, dict] = {}
    for item in ocr_results:
        try:
            original_by_id[int(item.get("id"))] = item
        except (TypeError, ValueError):
            continue

    replacements: dict[int, dict] = {}
    skip_ids: set[int] = set()
    for layout in layout_data:
        if not isinstance(layout, dict):
            continue
        fragment_ids = layout.get("line_fragment_ids")
        if not isinstance(fragment_ids, list) or len(fragment_ids) <= 1:
            continue
        try:
            primary_id = int(layout.get("id"))
            normalized_fragment_ids = [int(item_id) for item_id in fragment_ids]
        except (TypeError, ValueError):
            continue
        base = dict(original_by_id.get(primary_id) or original_by_id.get(normalized_fragment_ids[0]) or {})
        if not base:
            continue
        text = str(layout.get("text") or "").strip()
        red_box = layout.get("red_box")
        if not text or not isinstance(red_box, list) or len(red_box) < 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in red_box[:4]]
        base["id"] = primary_id
        base["text"] = text
        base["box"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}
        base["line_fragment_ids"] = normalized_fragment_ids
        fallback_source = str(base.get("fallback_source") or "")
        base["fallback_source"] = f"{fallback_source}+layout_line_merge" if fallback_source else "layout_line_merge"
        replacements[primary_id] = base
        skip_ids.update(item_id for item_id in normalized_fragment_ids if item_id != primary_id)

    if not replacements:
        return ocr_results

    merged_results: list[dict] = []
    for item in ocr_results:
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            merged_results.append(item)
            continue
        if item_id in skip_ids:
            continue
        merged_results.append(replacements.pop(item_id, item))
    merged_results.extend(replacements.values())
    return merged_results


def _provider_order() -> list[str]:
    preferred = os.environ.get("PREFERRED_PROVIDER", "").strip().lower()
    explicit = [
        provider.strip().lower()
        for provider in os.environ.get("TRANSLATION_PROVIDER_ORDER", "").split(",")
        if provider.strip()
    ]
    # Ordered by translation quality among the models actually configured per
    # provider below, strongest-healthy-first: mistral-large and the
    # non-reasoning 70B-class chat models outperform small/reasoning models
    # (qwen3-32b, gpt-oss) on grammar and register for short manga dialogue
    # lines. Providers with revoked/locked credentials are kept at the tail
    # rather than removed -- reserve_key() skips them for free and they
    # resume automatically the moment keys are rotated.
    defaults = [
        "mistral",
        "cerebras",
        "fireworks",
        "groq",
        "nvidia",
        "github",
        "openrouter",
        "gemini",
        "cloudflare",
    ]
    ordered = []
    for provider in [preferred, *explicit, *defaults]:
        normalized = {
            "google": "gemini",
            "google-gemini": "gemini",
            "gh": "github",
            "github-models": "github",
            "qwen": "openrouter",
            "qwen-mt": "openrouter",
            "alibaba": "openrouter",
            "alibaba-cloud": "openrouter",
            "open-router": "openrouter",
            "cerebras-ai": "cerebras",
            "nvidia-nim": "nvidia",
            "nim": "nvidia",
            "workers": "cloudflare",
            "workers-ai": "cloudflare",
            "cloudflare-workers": "cloudflare",
        }.get(provider, provider)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _page_context_enabled() -> bool:
    return os.environ.get("TRANSLATION_PAGE_CONTEXT", "1").strip().lower() not in {"0", "false", "off"}


def _load_page_context(sample_dir: Path, max_chars: int = 600) -> str:
    # Written by the live runtime server (run_extension_pipeline_server.py)
    # before this step runs, from a rolling per-site ring of the last couple
    # pages' accepted translations -- purely so names/tone stay consistent
    # across pages of the same manga read in sequence. Never present for the
    # offline validation suite (nothing writes this file there), so this is
    # a no-op / dead code path when running that suite.
    if not _page_context_enabled():
        return ""
    context_path = sample_dir / "page_context.json"
    if not context_path.exists():
        return ""
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list) or not lines:
        return ""
    text = " / ".join(str(line).strip() for line in lines if str(line).strip())
    return text[:max_chars]


def _rotated_provider_keys(provider: str, keys: list[str]) -> list[tuple[int, str]]:
    if not keys:
        return []
    disabled = API_PROVIDER_KEY_DISABLED.get(provider, {})
    active_indexes = [index for index in range(len(keys)) if index + 1 not in disabled]
    if not active_indexes:
        return []
    start = API_PROVIDER_KEY_CURSOR.get(provider, 0) % len(keys)
    ordered_indexes = [index for index in range(start, len(keys))]
    ordered_indexes.extend(index for index in range(0, start))
    rotated = [(index + 1, keys[index]) for index in ordered_indexes if index in active_indexes]
    API_PROVIDER_KEY_CURSOR[provider] = (start + 1) % len(keys)
    return rotated


def _auth_or_network_block(statuses: list[int]) -> bool:
    return bool(statuses) and all(status in {401, 403} for status in statuses)


def _disable_provider_key(provider: str, key_index: int, reason: object) -> None:
    API_PROVIDER_KEY_DISABLED.setdefault(provider, {})[key_index] = _scrub_secret(reason)


def _all_provider_keys_disabled(provider: str, keys: list[str]) -> bool:
    disabled = API_PROVIDER_KEY_DISABLED.get(provider, {})
    return bool(keys) and len(disabled) >= len(keys)


def _disabled_key_error(provider: str) -> str:
    disabled = API_PROVIDER_KEY_DISABLED.get(provider, {})
    if not disabled:
        return f"{provider} has no active API keys"
    rendered = ", ".join(f"key {index}: {reason}" for index, reason in sorted(disabled.items()))
    return f"{provider} has no active API keys ({rendered})"


def _http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
) -> tuple[int, object, int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "free-manga-translator-step7/1.0",
            **(headers or {}),
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8", errors="replace")
            elapsed = int((time.perf_counter() - started) * 1000)
            try:
                return response.status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return response.status, raw, elapsed
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        elapsed = int((time.perf_counter() - started) * 1000)
        try:
            return error.code, json.loads(raw), elapsed
        except json.JSONDecodeError:
            return error.code, raw, elapsed
    except Exception as error:
        elapsed = int((time.perf_counter() - started) * 1000)
        return 0, {"error": repr(error)}, elapsed

def _cjk_alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _local_fallback_enabled() -> bool:
    return os.environ.get("LOCAL_NLLB_TRANSLATION", "").strip().lower() in {"1", "true", "yes", "on"}


def _detect_source_lang(text: str) -> str:
    if re.search(r"[\uac00-\ud7af]", text):
        return "kor_Hang"
    if re.search(r"[\u3040-\u30ff]", text):
        return "jpn_Jpan"
    if re.search(r"[\u4e00-\u9fff]", text):
        traditional_markers = set("臺灣繁體國與學會還過後發個萬億醫藥龍鳳門風雲廣東話語")
        simplified_markers = set("台湾简体国与学会还过后发个万亿医药龙凤门风云广东话语")
        traditional_score = sum(1 for ch in text if ch in traditional_markers)
        simplified_score = sum(1 for ch in text if ch in simplified_markers)
        return "zho_Hant" if traditional_score > simplified_score else "zho_Hans"
    return "jpn_Jpan"


def _load_local_translator():
    global _LOCAL_TRANSLATOR
    if _LOCAL_TRANSLATOR is not None:
        return _LOCAL_TRANSLATOR

    with _LOCAL_TRANSLATOR_LOAD_LOCK:
        if _LOCAL_TRANSLATOR is not None:
            return _LOCAL_TRANSLATOR

        import torch
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(_LOCAL_TRANSLATOR_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(
                "  [LocalTranslator] CUDA unavailable; falling back to CPU. This is a real "
                "slowdown for the local NLLB fallback translator, not a silent no-op.",
                flush=True,
            )
        model.to(device)
        model.eval()
        _LOCAL_TRANSLATOR = (model, device)
        return _LOCAL_TRANSLATOR


def _local_tokenizer(source_lang: str):
    if source_lang not in _LOCAL_TOKENIZERS:
        with _LOCAL_TRANSLATOR_LOAD_LOCK:
            if source_lang not in _LOCAL_TOKENIZERS:
                from transformers import AutoTokenizer

                _LOCAL_TOKENIZERS[source_lang] = AutoTokenizer.from_pretrained(
                    _LOCAL_TRANSLATOR_MODEL,
                    src_lang=source_lang,
                )
    return _LOCAL_TOKENIZERS[source_lang]


def _clean_machine_translation(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ・")


def _local_translate(text: str) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) <= 1 or not re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", compact):
        return ""

    try:
        import torch

        source_lang = _detect_source_lang(compact)
        tokenizer = _local_tokenizer(source_lang)
        model, device = _load_local_translator()
        inputs = tokenizer(compact, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=96,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        translated = _clean_machine_translation(
            tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        )
    except Exception as error:
        print(f"  [local-translate-warn] {str(error)[:120]}", file=sys.stderr)
        return ""

    if not re.search(r"[A-Za-z]", translated):
        return ""
    if translated.count("?") / max(1, len(translated)) > 0.25:
        return ""
    if len(translated) > 240:
        translated = translated[:240].rsplit(" ", 1)[0].strip() or translated[:240]
    return translated


def _fallback_translate(japanese_text: str) -> str:
    """Translate with optional local model fallback only."""
    if not japanese_text or not japanese_text.strip():
        return ""

    if _local_fallback_enabled():
        fallback = _local_translate(japanese_text)
        if fallback:
            return fallback

    return f"[TL: {japanese_text[:20]}...]"


def translate(japanese_text: str) -> str:
    """Compatibility wrapper for single-string fallback translation."""
    return _fallback_translate(japanese_text)


def _clean_api_translation(text: object) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'` ")
    if cleaned.lower().startswith("translation:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned[:360]


def _repair_translation_perspective(source: str, translated: str) -> str:
    source_text = str(source or "")
    text = _clean_api_translation(translated)
    if not text:
        return text

    imperative_go_do = re.search(r"(してこい|してこ|してきな|してきて)", source_text)
    if imperative_go_do:
        pattern = re.compile(
            r"\b(?:I\s+should|I\s+need\s+to|I\s+have\s+to|I'll|I\s+will)\s+go\s+([^.!?]+)",
            flags=re.IGNORECASE,
        )

        def repl(match: re.Match[str]) -> str:
            action = match.group(1).strip()
            action = re.sub(r"\bmy\b", "your", action, flags=re.IGNORECASE)
            return f"Go {action}"

        repaired = pattern.sub(repl, text, count=1)
        repaired = re.sub(r"\bbrush\s+my\s+teeth\b", "brush your teeth", repaired, flags=re.IGNORECASE)
        return repaired

    return text


def _translation_box_budget(item: dict[str, object]) -> int:
    box = item.get("box") if isinstance(item, dict) else None
    if not isinstance(box, dict):
        return 110
    try:
        width = int(box.get("width") or int(box.get("x2", 0)) - int(box.get("x1", 0)))
        height = int(box.get("height") or int(box.get("y2", 0)) - int(box.get("y1", 0)))
    except Exception:
        return 110
    if width <= 0 or height <= 0:
        return 110
    area_budget = int((width * height) / 70)
    long_side_budget = int(max(width, height) * 0.95)
    return max(16, min(120, max(area_budget, long_side_budget)))


def _shorten_for_small_box(text: str, item: dict[str, object]) -> str:
    budget = _translation_box_budget(item)
    cleaned = _clean_api_translation(text)
    if len(cleaned) <= budget:
        return cleaned

    replacements = [
        (r"\bThere are lots of\b", "Many"),
        (r"\beveryone knows but doesn't know the name of\b", "everyone knows, not by name"),
        (r"\bdoes not know\b", "doesn't know"),
        (r"\bdo not know\b", "don't know"),
        (r"\bdo not\b", "don't"),
        (r"\bdoes not\b", "doesn't"),
        (r"\bI am\b", "I'm"),
        (r"\byou are\b", "you're"),
        (r"\bwe are\b", "we're"),
        (r"\bthey are\b", "they're"),
        (r"\bthat is\b", "that's"),
        (r"\bIt is\b", "It's"),
        (r"\bused to\b", "for"),
        (r"\bat the harbor\b", "at harbor"),
        (r"\bmushroom-shaped things\b", "mushroom things"),
        (r"\biron mushroom-shaped things\b", "iron mushrooms"),
        (r"\bkind of\b", "kinda"),
        (r"\bsort of\b", "sorta"),
    ]
    compacted = cleaned
    for pattern, replacement in replacements:
        compacted = re.sub(pattern, replacement, compacted, flags=re.IGNORECASE)
        compacted = re.sub(r"\s+", " ", compacted).strip()
        if len(compacted) <= budget:
            return compacted

    compacted = re.sub(r"\([^)]*\)", "", compacted)
    compacted = re.sub(r"\b(just|really|actually|basically|probably|maybe|perhaps|those|these)\b", "", compacted, flags=re.IGNORECASE)
    compacted = re.sub(r"\s+", " ", compacted).strip()
    if len(compacted) <= budget:
        return compacted

    first_sentence = re.split(r"(?<=[.!?])\s+", compacted, maxsplit=1)[0].strip()
    if 8 <= len(first_sentence) <= budget:
        return first_sentence

    words = compacted.split()
    if len(words) > 3:
        filtered = [
            word for word in words
            if word.lower().strip(".,!?;:") not in {"a", "an", "the", "of", "to", "that"}
        ]
        candidate = " ".join(filtered)
        if 8 <= len(candidate) <= budget:
            return candidate
        compacted = candidate or compacted

    if len(compacted) <= budget:
        return compacted
    clipped = compacted[: max(12, budget)].rsplit(" ", 1)[0].strip()
    return clipped or compacted[:budget].strip()


def _valid_api_translation(source: str, translated: object) -> bool:
    text = _clean_api_translation(translated)
    if not text or text.startswith("[TL:") or len(text) > 360:
        return False
    source_cjk = set(
        ch for ch in source
        if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"
    )
    output_cjk = set(
        ch for ch in text
        if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"
    )
    if source_cjk and len(source_cjk & output_cjk) / max(1, len(source_cjk)) > 0.35:
        return False
    if source_cjk and not re.search(r"[A-Za-z0-9]", text):
        meaningful_source_len = len(
            [
                ch for ch in source
                if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"
            ]
        )
        return meaningful_source_len <= 2 and len(text) <= 8
    return True


def _source_requires_translation(source: object) -> bool:
    text = str(source or "").strip()
    if not text:
        return False
    cjk_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    if cjk_count == 0:
        return False
    alnum_count = len(re.findall(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    if alnum_count == 0:
        return False
    if cjk_count <= 1 and len(text) <= 3:
        return False
    return True


def _pretranslated_text(item: dict[str, object]) -> str:
    for key in ("pretranslated_text", "vision_english", "english"):
        text = _clean_api_translation(item.get(key))
        if text and _valid_api_translation(str(item.get("text", "")), text):
            return text
    return ""


def _translation_prompt(items: list[dict[str, object]], sample_name: str, page_context: str = "") -> str:
    compact_items = []
    for item in items:
        source_text = str(item.get("text", "")).strip()
        if not _source_requires_translation(source_text):
            continue
        payload = {"id": int(item["id"]), "source_text": source_text}
        if str(item.get("semantic_role") or "") == "sfx":
            payload["role"] = "sfx"
        elif item.get("short_fragment"):
            payload["role"] = "brief"
        compact_items.append(payload)
    return (
        "You are a professional manga, manhwa, and manhua translator/typesetter assistant.\n"
        "Translate OCR text fragments from Japanese, Korean, or Chinese into accurate, natural English.\n"
        "Items tagged \"role\": \"sfx\" are sound effects, not dialogue: render them as a short punchy "
        "English sound-effect word (e.g. BAM, THUD, WHOOSH, CRASH) or onomatopoeia rather than a full "
        "grammatical sentence, and keep them in the same emphatic/exclamatory register as the source.\n"
        "Items tagged \"role\": \"brief\" are a tiny 1-4 character spoken fragment (a trailing particle, "
        "interjection, or reaction sound) that occupies a very small area on the page -- translate it as a "
        "similarly short interjection or trailing word (1-3 English words, e.g. \"...!\", \"Huh?\", \"Right...\", "
        "\"I know.\"), never as a full explanatory sentence, even if the source is ambiguous out of context.\n"
        "Use the full list of items as shared page context: before translating, identify any token that "
        "repeats across multiple items or looks like a personal name, honorific-attached name, or title "
        "(rather than a common noun/adjective) and translate it the same way — as a proper noun/name — in "
        "every item where it appears. Do not translate a name as an unrelated common word (e.g. a name must "
        "never become an object, animal, place, or furniture word) even if a literal character-by-character "
        "reading would suggest one.\n"
        "Romanize ONLY personal names, place names, and titles. Every common noun and ordinary object or "
        "concept word must ALWAYS be translated to its English meaning, never left as a romanized reading, "
        "even if it sounds name-like out of context or you are unsure — when in doubt, translate the meaning "
        "rather than romanize.\n"
        "Still translate each item's own sentence independently for grammar and meaning — only reuse the "
        "cross-item context for proper-noun and terminology consistency, not for inventing plot relationships.\n"
        "Understand manga/comic reading direction, connected speech bubbles, character tone, social hierarchy, slang, honorifics, and implied subjects.\n"
        "Preserve names, SFX tone when translated as dialogue, stutters, ellipses, shouting, short reactions, and punctuation rhythm.\n"
        "Preserve speaker/addressee perspective. Do not swap my/your/his/her/their.\n"
        "Japanese often omits subjects and objects: infer pronouns from local page context only when clear; otherwise use neutral wording instead of inventing ownership.\n"
        "For commands addressed to another person, use second person when the grammar/context implies it.\n"
        "If OCR is slightly noisy, infer the most plausible intended line.\n"
        "Do not include source-language characters in the English output unless they are personal/place/title "
        "names intentionally romanized.\n"
        "Keep each translation concise enough for manga typesetting while preserving meaning.\n"
        "Return JSON only, exactly this shape: [{\"id\": 0, \"en_text\": \"...\"}].\n"
        + (
            f"Prior page context, for name/tone consistency ONLY -- do not re-translate these, "
            f"they are already rendered: {page_context}\n"
            if page_context else ""
        )
        + f"Sample: {sample_name}\n"
        f"Items: {json.dumps(compact_items, ensure_ascii=False)}"
    )


def _extract_json_array(text: str) -> list[object]:
    stripped = text.strip()
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


def _parse_translation_response(payload: object) -> dict[int, str]:
    content = ""
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = str(message.get("content", ""))
        result = payload.get("result")
        if not content and isinstance(result, dict):
            content = str(result.get("response", ""))
        candidates = payload.get("candidates")
        if not content and isinstance(candidates, list) and candidates:
            candidate = candidates[0] if isinstance(candidates[0], dict) else {}
            parts = candidate.get("content", {}).get("parts", []) if isinstance(candidate.get("content"), dict) else []
            content = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    elif isinstance(payload, str):
        content = payload
    translations: dict[int, str] = {}
    for entry in _extract_json_array(content):
        if not isinstance(entry, dict):
            continue
        try:
            item_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        en_text = _clean_api_translation(entry.get("en_text") or entry.get("translation") or entry.get("text"))
        if en_text:
            translations[item_id] = en_text
    return translations


def _openai_payload(model: str, prompt: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior manga/manhwa/manhua translator. Preserve character voice, "
                    "speaker perspective, implied subjects, tone, punctuation rhythm, honorific nuance, "
                    "and layout-safe concision. Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("API_TRANSLATION_MAX_TOKENS", "1400")),
    }



def _api_max_tokens() -> int:
    return int(os.environ.get("API_TRANSLATION_MAX_TOKENS", "1400"))


def _estimated_translation_tokens(prompt: str) -> int:
    return API_MANAGER.estimate_tokens(prompt, output_tokens=_api_max_tokens())


def _call_with_quota(
    provider: str,
    prompt: str,
    models: list[str],
    request_builder,
    parser=_parse_translation_response,
) -> tuple[dict[int, str], dict[str, object]]:
    if not API_MANAGER.provider_keys(provider):
        raise RuntimeError(f"{provider} has no configured API keys")
    estimated_tokens = _estimated_translation_tokens(prompt)
    last_error = None
    for model_index, model in enumerate(models, start=1):
        attempted_hashes: set[str] = set()
        while True:
            try:
                lease = API_MANAGER.reserve_key(provider, estimated_tokens, capability="translation")
            except ApiProviderUnavailable as error:
                raise RuntimeError(str(error)) from error
            except ApiProviderAuthLocked:
                raise
            except ApiRateLimited:
                raise
            except ApiQuotaExhausted:
                raise
            if lease.key_hash in attempted_hashes:
                break
            attempted_hashes.add(lease.key_hash)
            status, payload, elapsed = request_builder(lease.key, model)
            if status == 200:
                API_MANAGER.mark_success(lease, payload)
                return parser(payload), {
                    "provider": provider,
                    "model": model,
                    "model_index": model_index,
                    "model_count": len(models),
                    "key_index": lease.key_index,
                    "key_count": len(API_MANAGER.provider_keys(provider)),
                    "key_fingerprint": lease.key_hash[:8],
                    "http_status": status,
                    "elapsed_ms": elapsed,
                    "estimated_tokens": estimated_tokens,
                }
            last_error = f"{provider} key {lease.key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
            API_MANAGER.mark_failure(lease, status, _scrub_secret(payload))
            if API_MANAGER.is_terminal_quota_error(status, payload) or status in {401, 402, 403, 429}:
                continue
            break
    raise RuntimeError(last_error or f"{provider} request failed")


def _call_openai_compatible(
    provider: str,
    prompt: str,
    endpoint: str,
    headers_factory,
    models: list[str],
) -> tuple[dict[int, str], dict[str, object]]:
    def request_builder(key: str, model: str):
        return _http_json(
            "POST",
            endpoint,
            headers=headers_factory(key),
            payload=_openai_payload(model, prompt),
        )

    return _call_with_quota(provider, prompt, models, request_builder)


def _call_mistral(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("MISTRAL_TRANSLATION_MODELS", "MISTRAL_TRANSLATION_MODEL"),
        ["mistral-large-latest", "mistral-small-latest", "mistral-medium-latest"],
    )
    return _call_openai_compatible(
        "mistral",
        prompt,
        "https://api.mistral.ai/v1/chat/completions",
        lambda key: {"Authorization": f"Bearer {key}"},
        models,
    )


def _call_github(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("GITHUB_TRANSLATION_MODELS", "GITHUB_TRANSLATION_MODEL"),
        ["openai/gpt-4o-mini", "openai/gpt-4.1-mini", "mistral-ai/mistral-small-2503"],
    )
    return _call_openai_compatible(
        "github",
        prompt,
        "https://models.github.ai/inference/chat/completions",
        lambda key: {
            "Authorization": f"Bearer {key}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        models,
    )


def _call_openrouter(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("OPENROUTER_TRANSLATION_MODELS", "OPENROUTER_TRANSLATION_MODEL"),
        [
            "qwen/qwen3-235b-a22b:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
    )
    return _call_openai_compatible(
        "openrouter",
        prompt,
        "https://openrouter.ai/api/v1/chat/completions",
        lambda key: {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "http://127.0.0.1",
            "X-Title": "Free Manga Translator Step 7",
        },
        models,
    )


def _call_groq(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("GROQ_TRANSLATION_MODELS", "GROQ_TRANSLATION_MODEL"),
        # llama-3.3-70b-versatile first: a plain chat model beats qwen3-32b's
        # reasoning-model output (think-tag risk, worse register match) on
        # short manga dialogue lines.
        ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"],
    )
    return _call_openai_compatible(
        "groq",
        prompt,
        "https://api.groq.com/openai/v1/chat/completions",
        lambda key: {"Authorization": f"Bearer {key}"},
        models,
    )


def _call_cerebras(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    base_url = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
    models = _model_candidates(
        ("CEREBRAS_TRANSLATION_MODELS", "CEREBRAS_TRANSLATION_MODEL"),
        # Non-reasoning chat models first; qwen-3-32b (reasoning) demoted to last.
        ["gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b"],
    )
    return _call_openai_compatible(
        "cerebras",
        prompt,
        f"{base_url}/chat/completions",
        lambda key: {"Authorization": f"Bearer {key}"},
        models,
    )


def _call_nvidia(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    base_url = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    models = _model_candidates(
        ("NVIDIA_NIM_TRANSLATION_MODELS", "NVIDIA_NIM_TRANSLATION_MODEL"),
        # A verified, well-established 70B chat model first; the qwen3.5 IDs
        # below don't match any published Qwen release naming and are kept
        # only as unverified fallbacks.
        ["meta/llama-3.1-70b-instruct", "qwen/qwen3-5-122b-a10b", "qwen/qwen3.5-397b-a17b", "mistralai/mistral-7b-instruct-v0.3", "meta/llama-3.1-8b-instruct"],
    )
    return _call_openai_compatible(
        "nvidia",
        prompt,
        f"{base_url}/chat/completions",
        lambda key: {"Authorization": f"Bearer {key}"},
        models,
    )


def _call_fireworks(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("FIREWORKS_TRANSLATION_MODELS", "FIREWORKS_TRANSLATION_MODEL"),
        [
            # A plain 72B chat model first; the 235B qwen3 entry is a
            # hybrid-thinking model (think-tag risk) demoted below it.
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/qwen3p235b-a22b",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        ],
    )
    return _call_openai_compatible(
        "fireworks",
        prompt,
        "https://api.fireworks.ai/inference/v1/chat/completions",
        lambda key: {"Authorization": f"Bearer {key}"},
        models,
    )


def _call_cloudflare(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("CLOUDFLARE_TRANSLATION_MODELS", "CLOUDFLARE_TRANSLATION_MODEL"),
        ["@cf/qwen/qwen3-30b-a3b-fp8", "@cf/qwen/qwq-32b", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "@cf/qwen/qwen1.5-14b-chat-awq", "@cf/meta/llama-3.1-8b-instruct"],
    )

    def request_builder(key: str, model: str):
        keys = API_MANAGER.provider_keys("cloudflare")
        try:
            key_index = keys.index(key) + 1
        except ValueError:
            key_index = 1
        account_id = API_MANAGER.cloudflare_account_id_for_key_index(key_index)
        return _http_json(
            "POST",
            f"https://api.cloudflare.com/client/v4/accounts/{urllib.parse.quote(account_id)}/ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload=_openai_payload(model, prompt),
        )

    return _call_with_quota("cloudflare", prompt, models, request_builder)


def _call_gemini(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    models = _model_candidates(
        ("GEMINI_TRANSLATION_MODELS", "GEMINI_TRANSLATION_MODEL"),
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
    )

    def request_builder(key: str, model: str):
        return _http_json(
            "POST",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(key)}",
            payload={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": _api_max_tokens(),
                    "responseMimeType": "application/json",
                },
            },
        )

    return _call_with_quota("gemini", prompt, models, request_builder)


PROVIDER_CALLS = {
    "mistral": _call_mistral,
    "github": _call_github,
    "gemini": _call_gemini,
    "cerebras": _call_cerebras,
    "fireworks": _call_fireworks,
    "openrouter": _call_openrouter,
    "groq": _call_groq,
    "nvidia": _call_nvidia,
    "cloudflare": _call_cloudflare,
}


def _api_translate_items(
    items: list[dict[str, object]], sample_name: str, page_context: str = ""
) -> tuple[dict[int, str], dict[str, object]]:
    if not _api_translation_enabled():
        return {}, {"enabled": False, "reason": "API translation disabled or no provider keys configured"}
    skipped_ids = [
        int(item["id"])
        for item in items
        if "id" in item and not _source_requires_translation(item.get("text", ""))
    ]
    pretranslated_ids = [
        int(item["id"])
        for item in items
        if "id" in item and _source_requires_translation(item.get("text", "")) and _pretranslated_text(item)
    ]
    translatable_items = [
        item
        for item in items
        if "id" in item
        and _source_requires_translation(item.get("text", ""))
        and int(item["id"]) not in pretranslated_ids
    ]
    source_by_id = {int(item["id"]): str(item.get("text", "")) for item in translatable_items}
    pending_ids = set(source_by_id)
    translations: dict[int, str] = {}
    provider_attempts: list[dict[str, object]] = []
    provider_order = _provider_order()
    if not pending_ids:
        return translations, {
            "enabled": True,
            "attempts": provider_attempts,
            "translated": 0,
            "missing": [],
            "skipped_nontranslatable": sorted(skipped_ids),
            "pretranslated": sorted(pretranslated_ids),
        }
    if API_MANAGER.all_configured_providers_exhausted(provider_order):
        if _local_fallback_enabled():
            return translations, {
                "enabled": True,
                "status": "api_quota_exhausted_local_fallback",
                "reason": DAILY_LIMIT_MESSAGE,
                "attempts": provider_attempts,
                "translated": 0,
                "missing": sorted(pending_ids),
                "skipped_nontranslatable": sorted(skipped_ids),
                "pretranslated": sorted(pretranslated_ids),
            }
        raise ApiQuotaExhausted(DAILY_LIMIT_MESSAGE)

    prompt = _translation_prompt(translatable_items, sample_name, page_context)

    for provider in provider_order:
        if not pending_ids:
            break
        if provider in API_PROVIDER_DISABLED:
            provider_attempts.append(
                {
                    "provider": provider,
                    "status": "skipped_disabled",
                    "reason": API_PROVIDER_DISABLED[provider],
                    "remaining": len(pending_ids),
                }
            )
            continue
        call = PROVIDER_CALLS.get(provider)
        if call is None:
            continue
        for attempt in range(1, API_RETRIES_PER_PROVIDER + 1):
            try:
                parsed, meta = call(prompt)
                accepted = 0
                for item_id, en_text in parsed.items():
                    if item_id in pending_ids and _valid_api_translation(source_by_id[item_id], en_text):
                        translations[item_id] = _clean_api_translation(en_text)
                        pending_ids.remove(item_id)
                        accepted += 1
                attempt_meta = {
                    **meta,
                    "attempt": attempt,
                    "accepted": accepted,
                    "remaining": len(pending_ids),
                    "status": "pass" if accepted else "empty_or_invalid",
                }
                provider_attempts.append(attempt_meta)
                API_PROVIDER_USES.append({"sample": sample_name, **attempt_meta})
                print(f"  [api-translate] {provider} attempt {attempt}: accepted {accepted}, remaining {len(pending_ids)}")
                if accepted:
                    break
            except ApiProviderAuthLocked as error:
                failure = {
                    "sample": sample_name,
                    "provider": provider,
                    "attempt": attempt,
                    "error": _scrub_secret(error),
                    "status": "auth_locked",
                    "remaining": len(pending_ids),
                }
                API_PROVIDER_FAILURES.append(failure)
                provider_attempts.append(failure)
                API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
                print(f"  [api-translate-access-blocked] {provider}: {_scrub_secret(error)}", file=sys.stderr)
                break
            except ApiQuotaExhausted as error:
                failure = {
                    "sample": sample_name,
                    "provider": provider,
                    "attempt": attempt,
                    "error": _scrub_secret(error),
                    "status": "quota_exhausted",
                    "remaining": len(pending_ids),
                }
                API_PROVIDER_FAILURES.append(failure)
                provider_attempts.append(failure)
                API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
                print(f"  [api-translate-critical] {provider}: {_scrub_secret(error)}", file=sys.stderr)
                break
            except ApiRateLimited as error:
                failure = {
                    "sample": sample_name,
                    "provider": provider,
                    "attempt": attempt,
                    "error": _scrub_secret(error),
                    "status": "rate_limited",
                    "remaining": len(pending_ids),
                }
                API_PROVIDER_FAILURES.append(failure)
                provider_attempts.append(failure)
                API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
                print(f"  [api-translate-rate-limit] {provider}: {_scrub_secret(error)}", file=sys.stderr)
                break
            except Exception as error:
                failure = {
                    "sample": sample_name,
                    "provider": provider,
                    "attempt": attempt,
                    "error": _scrub_secret(error),
                }
                API_PROVIDER_FAILURES.append(failure)
                provider_attempts.append({**failure, "status": "fail"})
                print(f"  [api-translate-warn] {provider} attempt {attempt}: {_scrub_secret(error)}", file=sys.stderr)
                if "no active API keys" in str(error).lower() or "http 401" in str(error).lower() or "http 403" in str(error).lower():
                    API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
                    break
                if attempt < API_RETRIES_PER_PROVIDER:
                    time.sleep(min(4, attempt * 1.5))
                else:
                    API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
    if pending_ids and API_MANAGER.all_configured_providers_exhausted(provider_order):
        if _local_fallback_enabled():
            provider_attempts.append({
                "status": "api_quota_exhausted_local_fallback",
                "reason": DAILY_LIMIT_MESSAGE,
                "remaining": len(pending_ids),
            })
        else:
            raise ApiQuotaExhausted(DAILY_LIMIT_MESSAGE)
    return translations, {
        "enabled": True,
        "attempts": provider_attempts,
        "translated": len(translations),
        "missing": sorted(pending_ids),
        "skipped_nontranslatable": sorted(skipped_ids),
        "pretranslated": sorted(pretranslated_ids),
    }


def run_step7_translate(sample_map: dict[str, str] | None = None, samples_dir: Path | None = None):
    RUN_NAME = "step_7_translate"
    samples_dir = Path(samples_dir) if samples_dir is not None else sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP
    
    print("=" * 60)
    print("  Step 7 — Contextual Translation")
    print("=" * 60)
    
    for sample_name, img_file in sample_map.items():
        ocr_json_path = samples_dir / sample_name / "step_5_ocr" / "ocr_results.json"
        
        if not ocr_json_path.exists():
            continue
            
        print(f"\nProcessing {sample_name}")
        ocr_results = json.loads(ocr_json_path.read_text(encoding="utf-8"))
        ocr_results = _translation_items_with_layout_merges(ocr_results, samples_dir / sample_name)

        # Reuse translations from a prior run for ids whose source text is
        # byte-identical to what it was then, so a rerun only pays API cost
        # for genuinely new/changed ids (Step 6 relayout, retried OCR, etc.).
        # "fallback" entries are deliberately excluded from the reusable set:
        # they mean the API was unavailable/rate-limited last time, not that
        # the text was untranslatable, so a later run with quota available
        # should get a real shot at them instead of being locked into the
        # degraded dictionary fallback forever.
        prior_out_dir = samples_dir / sample_name / RUN_NAME
        prior_results_path = prior_out_dir / "translation_results.json"
        reusable_by_id: dict[int, dict[str, object]] = {}
        if prior_results_path.exists():
            try:
                prior_results = json.loads(prior_results_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prior_results = []
            for entry in prior_results:
                if entry.get("translation_source") == "fallback":
                    continue
                if "id" not in entry:
                    continue
                reusable_by_id[int(entry["id"])] = entry

        reused_ids: set[int] = set()
        for item in ocr_results:
            if "id" not in item:
                continue
            prior_entry = reusable_by_id.get(int(item["id"]))
            if prior_entry is not None and prior_entry.get("jp_text") == item.get("text", ""):
                reused_ids.add(int(item["id"]))

        items_for_api = [item for item in ocr_results if int(item.get("id", -1)) not in reused_ids]
        page_context = _load_page_context(samples_dir / sample_name)
        api_translations, api_report = _api_translate_items(items_for_api, sample_name, page_context)
        api_report["reused_from_prior_run"] = sorted(reused_ids)

        translated_texts = []
        for item in ocr_results:
            item_id = int(item["id"])
            if item_id in reused_ids:
                prior_entry = reusable_by_id[item_id]
                translated_texts.append({
                    "id": item["id"],
                    "box": item["box"],
                    "jp_text": item["text"],
                    "en_text": prior_entry.get("en_text", ""),
                    "translation_source": prior_entry.get("translation_source", "api"),
                    "semantic_role": item.get("semantic_role", ""),
                })
                print(f"  [{item['id']}] {item['text'][:30]}  →  {prior_entry.get('en_text', '')[:50]} [reused]")
                continue
            vision_text = _pretranslated_text(item)
            api_text = api_translations.get(item_id)
            if vision_text:
                en_text = _shorten_for_small_box(
                    _repair_translation_perspective(item["text"], vision_text),
                    item,
                )
                source = "vision"
            elif not _source_requires_translation(item.get("text", "")):
                en_text = ""
                source = "skipped"
            elif api_text and _valid_api_translation(item["text"], api_text):
                en_text = _shorten_for_small_box(
                    _repair_translation_perspective(item["text"], api_text),
                    item,
                )
                source = "api"
            else:
                en_text = _repair_translation_perspective(
                    item["text"],
                    _fallback_translate(item["text"]),
                )
                source = "fallback"
            translated_texts.append({
                "id": item["id"],
                "box": item["box"],
                "jp_text": item["text"],
                "en_text": en_text,
                "translation_source": source,
                "semantic_role": item.get("semantic_role", ""),
            })
            print(f"  [{item['id']}] {item['text'][:30]}  →  {en_text[:50]} [{source}]")

        # === Output ===
        out_dir = samples_dir / sample_name / RUN_NAME
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        (out_dir / "translation_results.json").write_text(
            json.dumps(translated_texts, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )

        (out_dir / "translation_provider_report.json").write_text(
            json.dumps(
                {
                    "sample_name": sample_name,
                    "api_enabled": api_report.get("enabled", False),
                    "provider_order": _provider_order(),
                    "api_report": api_report,
                    "api_failures": [
                        failure for failure in API_PROVIDER_FAILURES
                        if failure.get("sample") == sample_name
                    ],
                    "disabled_provider_keys": {
                        provider: {
                            str(key_index): reason
                            for key_index, reason in disabled.items()
                        }
                        for provider, disabled in API_PROVIDER_KEY_DISABLED.items()
                    },
                    "translation_source_counts": {
                        source: sum(1 for item in translated_texts if item["translation_source"] == source)
                        for source in sorted({item["translation_source"] for item in translated_texts})
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"  Saved to: {out_dir}")

    print("\nDone!")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_step7_translate()
