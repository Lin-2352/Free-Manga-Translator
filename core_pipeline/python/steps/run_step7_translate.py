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

import html
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ml_region_lib import SAMPLE_MAP
from pipeline_paths import DEFAULT_SAMPLES_ROOT, PROJECT_ROOT, sample_root_from_env

_LOCAL_TRANSLATOR = None
_LOCAL_TOKENIZERS: dict[str, object] = {}
_LOCAL_TRANSLATOR_MODEL = "facebook/nllb-200-distilled-600M"
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
    return [part.strip() for part in os.environ.get(name, "").split(",") if _configured(part.strip())]


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
    "NVIDIA_API_KEYS",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEYS",
    "NVIDIA_NIM_API_KEY",
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
            _csv_env("GEMINI_API_KEYS"),
            _provider_keys("GITHUB_API_KEYS", "GITHUB_API_KEY"),
            _provider_keys("GROQ_API_KEYS", "GROQ_API_KEY"),
            _provider_keys("MISTRAL_API_KEYS", "MISTRAL_API_KEY"),
            _provider_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
            _provider_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY"),
        ]
    )


def _api_translation_enabled() -> bool:
    value = os.environ.get("USE_API_TRANSLATION", "auto").strip().lower()
    if value in {"0", "false", "no", "off", "local"}:
        return False
    if value in {"1", "true", "yes", "on", "api", "auto"}:
        return _has_api_keys()
    return _has_api_keys()


def _provider_order() -> list[str]:
    preferred = os.environ.get("PREFERRED_PROVIDER", "").strip().lower()
    explicit = [
        provider.strip().lower()
        for provider in os.environ.get("TRANSLATION_PROVIDER_ORDER", "").split(",")
        if provider.strip()
    ]
    defaults = [
        "mistral",
        "github",
        "gemini",
        "openrouter",
        "groq",
        "nvidia",
    ]
    ordered = []
    for provider in [preferred, *explicit, *defaults]:
        normalized = {
            "google": "gemini",
            "google-gemini": "gemini",
            "gh": "github",
            "github-models": "github",
            "open-router": "openrouter",
            "nvidia-nim": "nvidia",
            "nim": "nvidia",
        }.get(provider, provider)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


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
        return "zho_Hant"
    return "jpn_Jpan"


def _load_local_translator():
    global _LOCAL_TRANSLATOR
    if _LOCAL_TRANSLATOR is not None:
        return _LOCAL_TRANSLATOR

    import torch
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(_LOCAL_TRANSLATOR_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    _LOCAL_TRANSLATOR = (model, device)
    return _LOCAL_TRANSLATOR


def _local_tokenizer(source_lang: str):
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
                max_new_tokens=96,
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


def _translation_prompt(items: list[dict[str, object]], sample_name: str) -> str:
    compact_items = []
    for item in items:
        source_text = str(item.get("text", "")).strip()
        if not source_text:
            continue
        payload = {"id": int(item["id"]), "source_text": source_text}
        compact_items.append(payload)
    return (
        "You are a professional manga, manhwa, and manhua translator/typesetter assistant.\n"
        "Translate OCR text fragments from Japanese, Korean, or Chinese into natural English.\n"
        "Use the surrounding list as page context, but translate each id independently.\n"
        "Preserve names, SFX tone, stutters, ellipses, shouting, and short reactions.\n"
        "Preserve speaker/addressee perspective. Do not swap my/your/his/her/their.\n"
        "Japanese often omits subjects and objects: infer pronouns from local page context only when clear; otherwise use neutral wording instead of inventing ownership.\n"
        "For commands addressed to another person, use second person when the grammar/context implies it.\n"
        "If OCR is slightly noisy, infer the most plausible intended line.\n"
        "Do not include source-language characters in the English output unless they are proper names intentionally romanized.\n"
        "Return JSON only, exactly this shape: [{\"id\": 0, \"en_text\": \"...\"}].\n"
        f"Sample: {sample_name}\n"
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
                "content": "You translate CJK manga OCR fragments to concise, natural English and return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("API_TRANSLATION_MAX_TOKENS", "1400")),
    }


def _call_mistral(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _provider_keys("MISTRAL_API_KEYS", "MISTRAL_API_KEY")
    if not keys:
        raise RuntimeError("MISTRAL_API_KEY/MISTRAL_API_KEYS is not configured")
    models = _model_candidates(
        ("MISTRAL_TRANSLATION_MODELS", "MISTRAL_TRANSLATION_MODEL"),
        ["mistral-small-latest", "mistral-medium-latest", "ministral-8b-latest"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("mistral", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                payload=_openai_payload(model, prompt),
            )
            key_statuses.append(status)
            if status == 200:
                return _parse_translation_response(payload), {"provider": "mistral", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"Mistral key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("mistral", key_index, last_error)
    if _all_provider_keys_disabled("mistral", keys):
        raise RuntimeError(_disabled_key_error("mistral"))
    raise RuntimeError(last_error or "Mistral request failed")


def _call_github(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _provider_keys("GITHUB_API_KEYS", "GITHUB_API_KEY")
    if not keys:
        raise RuntimeError("GITHUB_API_KEY/GITHUB_API_KEYS is not configured")
    models = _model_candidates(
        ("GITHUB_TRANSLATION_MODELS", "GITHUB_TRANSLATION_MODEL"),
        ["openai/gpt-4.1-mini", "openai/gpt-4o-mini", "mistral-ai/mistral-small-2503"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("github", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                "https://models.github.ai/inference/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                payload=_openai_payload(model, prompt),
            )
            key_statuses.append(status)
            if status == 200:
                return _parse_translation_response(payload), {"provider": "github", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"GitHub Models key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("github", key_index, last_error)
    if _all_provider_keys_disabled("github", keys):
        raise RuntimeError(_disabled_key_error("github"))
    raise RuntimeError(last_error or "GitHub Models request failed")


def _call_openrouter(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _provider_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
    if not keys:
        raise RuntimeError("OPENROUTER_API_KEY/OPENROUTER_API_KEYS is not configured")
    models = _model_candidates(
        ("OPENROUTER_TRANSLATION_MODELS", "OPENROUTER_TRANSLATION_MODEL"),
        ["openrouter/free", "meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-exp:free", "qwen/qwen-2.5-72b-instruct:free"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("openrouter", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "http://127.0.0.1",
                    "X-Title": "Free Manga Translator Step 7",
                },
                payload=_openai_payload(model, prompt),
            )
            key_statuses.append(status)
            if status == 200:
                return _parse_translation_response(payload), {"provider": "openrouter", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"OpenRouter key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("openrouter", key_index, last_error)
    if _all_provider_keys_disabled("openrouter", keys):
        raise RuntimeError(_disabled_key_error("openrouter"))
    raise RuntimeError(last_error or "OpenRouter request failed")


def _call_groq(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _provider_keys("GROQ_API_KEYS", "GROQ_API_KEY")
    if not keys:
        raise RuntimeError("GROQ_API_KEY/GROQ_API_KEYS is not configured")
    models = _model_candidates(
        ("GROQ_TRANSLATION_MODELS", "GROQ_TRANSLATION_MODEL"),
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "gemma2-9b-it"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("groq", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                payload=_openai_payload(model, prompt),
            )
            key_statuses.append(status)
            if status == 200:
                return _parse_translation_response(payload), {"provider": "groq", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"Groq key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("groq", key_index, last_error)
    if _all_provider_keys_disabled("groq", keys):
        raise RuntimeError(_disabled_key_error("groq"))
    raise RuntimeError(last_error or "Groq request failed")


def _call_nvidia(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _provider_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY")
    if not keys:
        raise RuntimeError("NVIDIA_API_KEY/NVIDIA_NIM_API_KEY is not configured")
    base_url = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    models = _model_candidates(
        ("NVIDIA_NIM_TRANSLATION_MODELS", "NVIDIA_NIM_TRANSLATION_MODEL"),
        ["meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct", "mistralai/mistral-7b-instruct-v0.3"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("nvidia", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                payload=_openai_payload(model, prompt),
            )
            key_statuses.append(status)
            if status == 200:
                return _parse_translation_response(payload), {"provider": "nvidia", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"NVIDIA NIM key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("nvidia", key_index, last_error)
    if _all_provider_keys_disabled("nvidia", keys):
        raise RuntimeError(_disabled_key_error("nvidia"))
    raise RuntimeError(last_error or "NVIDIA NIM request failed")


def _call_gemini(prompt: str) -> tuple[dict[int, str], dict[str, object]]:
    keys = _csv_env("GEMINI_API_KEYS")
    if not keys:
        raise RuntimeError("GEMINI_API_KEYS is not configured")
    models = _model_candidates(
        ("GEMINI_TRANSLATION_MODELS", "GEMINI_TRANSLATION_MODEL"),
        ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"],
    )
    last_error = None
    for key_index, key in _rotated_provider_keys("gemini", keys):
        key_statuses: list[int] = []
        for model_index, model in enumerate(models, start=1):
            status, payload, elapsed = _http_json(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(key)}",
                payload={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": int(os.environ.get("API_TRANSLATION_MAX_TOKENS", "1400")),
                        "responseMimeType": "application/json",
                    },
                },
            )
            key_statuses.append(status)
            if status == 200:
                parsed = _parse_translation_response(payload)
                return parsed, {"provider": "gemini", "model": model, "model_index": model_index, "model_count": len(models), "key_index": key_index, "key_count": len(keys), "http_status": status, "elapsed_ms": elapsed}
            last_error = f"Gemini key {key_index} model {model} HTTP {status}: {_scrub_secret(payload)}"
        if _auth_or_network_block(key_statuses):
            _disable_provider_key("gemini", key_index, last_error)
    if _all_provider_keys_disabled("gemini", keys):
        raise RuntimeError(_disabled_key_error("gemini"))
    raise RuntimeError(last_error or "Gemini request failed")


PROVIDER_CALLS = {
    "mistral": _call_mistral,
    "github": _call_github,
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
    "groq": _call_groq,
    "nvidia": _call_nvidia,
}


def _api_translate_items(items: list[dict[str, object]], sample_name: str) -> tuple[dict[int, str], dict[str, object]]:
    if not _api_translation_enabled():
        return {}, {"enabled": False, "reason": "API translation disabled or no provider keys configured"}
    prompt = _translation_prompt(items, sample_name)
    source_by_id = {int(item["id"]): str(item.get("text", "")) for item in items if "id" in item}
    pending_ids = set(source_by_id)
    translations: dict[int, str] = {}
    provider_attempts: list[dict[str, object]] = []

    for provider in _provider_order():
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
                if attempt < API_RETRIES_PER_PROVIDER:
                    time.sleep(min(4, attempt * 1.5))
                else:
                    API_PROVIDER_DISABLED[provider] = _scrub_secret(error)
    return translations, {
        "enabled": True,
        "attempts": provider_attempts,
        "translated": len(translations),
        "missing": sorted(pending_ids),
    }


def run_step7_translate():
    RUN_NAME = "step_7_translate"
    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    
    print("=" * 60)
    print("  Step 7 — Contextual Translation")
    print("=" * 60)
    
    for sample_name, img_file in SAMPLE_MAP.items():
        ocr_json_path = samples_dir / sample_name / "step_5_ocr" / "ocr_results.json"
        
        if not ocr_json_path.exists():
            continue
            
        print(f"\nProcessing {sample_name}")
        ocr_results = json.loads(ocr_json_path.read_text(encoding="utf-8"))

        api_translations, api_report = _api_translate_items(ocr_results, sample_name)

        translated_texts = []
        for item in ocr_results:
            item_id = int(item["id"])
            api_text = api_translations.get(item_id)
            if api_text and _valid_api_translation(item["text"], api_text):
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
            })
            print(f"  [{item['id']}] {item['text'][:30]}  →  {en_text[:50]} [{source}]")


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
                        "api": sum(1 for item in translated_texts if item["translation_source"] == "api"),
                        "fallback": sum(1 for item in translated_texts if item["translation_source"] == "fallback"),
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
