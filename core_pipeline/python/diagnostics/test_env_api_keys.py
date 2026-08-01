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

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from pipeline_paths import PROJECT_ROOT
ENV_PATH = PROJECT_ROOT / ".env"
REPORT_PATH = PROJECT_ROOT / "quality_reports" / "api_key_report.md"
TIMEOUT_SECONDS = 45

TRANSLATION_PROMPT = "Translate this Japanese text to English. Answer with only the translation: \u3053\u3093\u306b\u3061\u306f"
VISION_PROMPT = "What is the main color of this image? Answer with one word."
def make_red_png_base64() -> str:
    try:
        from io import BytesIO

        from PIL import Image

        image = Image.new("RGB", (8, 8), (255, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


RED_DOT_PNG_BASE64 = make_red_png_base64()
RED_DOT_DATA_URL = f"data:image/png;base64,{RED_DOT_PNG_BASE64}"


@dataclass
class CheckResult:
    provider: str
    check: str
    status: str
    detail: str
    model: str = ""
    http_status: int | None = None
    elapsed_ms: int = 0


@dataclass
class ProviderSummary:
    provider: str
    configured: bool
    uses_current_pipeline: bool
    required_for_current_pipeline: bool
    results: list[CheckResult] = field(default_factory=list)


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


ENV_VALUES = load_dotenv(ENV_PATH)
SECRET_KEYS = {
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
    "CEREBRAS_API_KEYS",
    "CEREBRAS_API_KEY",
    "CEREBERAS_API_KEYS",
    "CEREBERAS_API_KEY",
    "FIREWORKS_API_KEYS",
    "FIREWORKS_API_KEY",
    "CLOUDFLARE_WORKERS_API_KEYS",
    "CLOUDFLARE_WORKERS_API_KEY",
    "CLOUDFLARE_API_KEYS",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_ACCOUNT_IDS",
    "CLOUDFLARE_ACCOUNT_ID",
    "HF_TOKEN",
}
SECRETS = [
    value
    for key, value in ENV_VALUES.items()
    if key in SECRET_KEYS and value and "YOUR_" not in value
]
for maybe_csv in list(SECRETS):
    if "," in maybe_csv:
        SECRETS.extend(part.strip() for part in maybe_csv.split(",") if part.strip())


def is_configured(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped and "YOUR_" not in stripped and not re.fullmatch(r"[_\-\s]*", stripped))


def csv_values(name: str) -> list[str]:
    raw = ENV_VALUES.get(name, "")
    return [part.strip() for part in raw.split(",") if is_configured(part.strip())]


def provider_keys(*names: str) -> list[str]:
    keys: list[str] = []
    for name in names:
        for key in csv_values(name):
            if key not in keys:
                keys.append(key)
    return keys


def model_candidates(names: tuple[str, ...], defaults: list[str]) -> list[str]:
    values: list[str] = []
    for name in names:
        for value in csv_values(name):
            if value not in values:
                values.append(value)
    return values or defaults


def api_translation_enabled() -> bool:
    value = ENV_VALUES.get("USE_API_TRANSLATION", "auto").strip().lower()
    if value in {"0", "false", "no", "off", "local"}:
        return False
    return any(
        [
            provider_keys("GEMINI_API_KEYS"),
            provider_keys("GITHUB_API_KEYS", "GITHUB_API_KEY"),
            provider_keys("GROQ_API_KEYS", "GROQ_API_KEY"),
            provider_keys("MISTRAL_API_KEYS", "MISTRAL_API_KEY"),
            provider_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
            provider_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY"),
            provider_keys("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY", "CEREBERAS_API_KEYS", "CEREBERAS_API_KEY"),
            provider_keys("FIREWORKS_API_KEYS", "FIREWORKS_API_KEY"),
            provider_keys("CLOUDFLARE_WORKERS_API_KEYS", "CLOUDFLARE_WORKERS_API_KEY", "CLOUDFLARE_API_KEYS", "CLOUDFLARE_API_KEY"),
        ]
    )


def provider_order() -> list[str]:
    preferred = ENV_VALUES.get("PREFERRED_PROVIDER", "").strip().lower()
    explicit = [
        provider.strip().lower()
        for provider in ENV_VALUES.get("TRANSLATION_PROVIDER_ORDER", "").split(",")
        if provider.strip()
    ]
    defaults = ["gemini", "mistral", "github", "openrouter", "groq", "nvidia", "cerebras", "fireworks", "cloudflare"]
    ordered: list[str] = []
    for provider in [preferred, *explicit, *defaults]:
        if provider and provider not in ordered:
            ordered.append(provider)
    return ordered


def current_pipeline_uses(provider: str) -> bool:
    return api_translation_enabled() and provider in provider_order()


def make_summary(display_name: str, provider: str, keys: list[str]) -> ProviderSummary:
    return ProviderSummary(display_name, bool(keys), current_pipeline_uses(provider), False)


def scrub(text: Any) -> str:
    rendered = str(text)
    for secret in sorted(set(SECRETS), key=len, reverse=True):
        if secret:
            rendered = rendered.replace(secret, "[REDACTED]")
    rendered = re.sub(r"key=([^&\\s]+)", "key=[REDACTED]", rendered)
    rendered = re.sub(r"Bearer\\s+[A-Za-z0-9_\\.\\-]+", "Bearer [REDACTED]", rendered)
    return rendered[:600]


def http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any, int]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "free-manga-translator-api-key-test/1.0",
            **(headers or {}),
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
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


def ok_from_chat(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, scrub(payload)
    if "error" in payload:
        return False, scrub(payload["error"])
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        return bool(content), scrub(content)
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = " ".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return bool(text), scrub(text)
    return False, scrub(payload)


def add_result(
    summary: ProviderSummary,
    check: str,
    status: str,
    detail: str,
    model: str = "",
    http_status: int | None = None,
    elapsed_ms: int = 0,
) -> None:
    summary.results.append(
        CheckResult(
            provider=summary.provider,
            check=check,
            status=status,
            detail=scrub(detail),
            model=model,
            http_status=http_status,
            elapsed_ms=elapsed_ms,
        )
    )


def select_model(ids: list[str], preferred: list[str], contains: list[str] | None = None) -> str:
    id_set = set(ids)
    for model in preferred:
        if model in id_set:
            return model
        prefixed = f"models/{model}"
        if prefixed in id_set:
            return prefixed
    if contains:
        lowered = [(model, model.lower()) for model in ids]
        for needle in contains:
            for model, lowered_model in lowered:
                if needle.lower() in lowered_model:
                    return model
    return ids[0] if ids else ""


def test_gemini() -> ProviderSummary:
    keys = provider_keys("GEMINI_API_KEYS")
    summary = make_summary("Gemini", "gemini", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable GEMINI_API_KEYS value found.")
        return summary

    for index, key in enumerate(keys, start=1):
        label = f"GEMINI_API_KEYS[{index}]"
        status, models_payload, elapsed = http_json(
            "GET",
            f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(key)}",
        )
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            continue
        models = models_payload.get("models", []) if isinstance(models_payload, dict) else []
        model_names = [
            item.get("name", "")
            for item in models
            if isinstance(item, dict)
            and "generateContent" in item.get("supportedGenerationMethods", [])
            # Exclude audio/image/embedding-only variants that happen to
            # contain "flash" in their id (e.g. gemini-2.5-flash-preview-tts)
            # -- they list generateContent but reject TEXT response modality.
            and not any(bad in item.get("name", "").lower() for bad in ("-tts", "-image", "embedding"))
        ]
        preferred_models = model_candidates(
            ("GEMINI_TRANSLATION_MODELS", "GEMINI_TRANSLATION_MODEL"),
            ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"],
        )
        candidate_models: list[str] = []
        model_set = set(model_names)
        for candidate in preferred_models:
            if candidate in model_set and candidate not in candidate_models:
                candidate_models.append(candidate)
            prefixed = f"models/{candidate}"
            if prefixed in model_set and candidate not in candidate_models:
                candidate_models.append(candidate)
        for candidate in model_names:
            model_id = candidate.replace("models/", "")
            if "flash" in model_id and model_id not in candidate_models:
                candidate_models.append(model_id)
        add_result(summary, f"{label} auth/models", "pass", f"{len(model_names)} generateContent-capable models listed; {len(candidate_models)} fallback candidates.", model=candidate_models[0] if candidate_models else "", http_status=status, elapsed_ms=elapsed)
        if not candidate_models:
            add_result(summary, f"{label} text", "skip", "No generateContent model available.")
            add_result(summary, f"{label} vision", "skip", "No generateContent model available.")
            continue
        text_last = ("No Gemini text model candidate succeeded.", "", None, 0)
        for model_id in candidate_models[:8]:
            base = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={urllib.parse.quote(key)}"
            text_payload = {
                "contents": [{"parts": [{"text": TRANSLATION_PROMPT}]}],
                "generationConfig": {"maxOutputTokens": 64, "temperature": 0},
            }
            status, payload, elapsed = http_json("POST", base, payload=text_payload)
            success, detail = ok_from_chat(payload)
            if status == 200 and success:
                add_result(summary, f"{label} text/translation", "pass", detail, model=model_id, http_status=status, elapsed_ms=elapsed)
                break
            text_last = (detail, model_id, status, elapsed)
        else:
            detail, model_id, status, elapsed = text_last
            add_result(summary, f"{label} text/translation", "fail", detail, model=model_id, http_status=status, elapsed_ms=elapsed)

        vision_last = ("No Gemini vision model candidate succeeded.", "", None, 0)
        for model_id in candidate_models[:8]:
            base = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={urllib.parse.quote(key)}"
            vision_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": VISION_PROMPT},
                            {"inline_data": {"mime_type": "image/png", "data": RED_DOT_PNG_BASE64}},
                        ]
                    }
                ],
                "generationConfig": {"maxOutputTokens": 64, "temperature": 0},
            }
            status, payload, elapsed = http_json("POST", base, payload=vision_payload)
            success, detail = ok_from_chat(payload)
            if status == 200 and success:
                add_result(summary, f"{label} vision", "pass", detail, model=model_id, http_status=status, elapsed_ms=elapsed)
                break
            vision_last = (detail, model_id, status, elapsed)
        else:
            detail, model_id, status, elapsed = vision_last
            add_result(summary, f"{label} vision", "fail", detail, model=model_id, http_status=status, elapsed_ms=elapsed)
    return summary


def openai_chat_payload(content: Any, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0,
    }


def test_groq() -> ProviderSummary:
    keys = provider_keys("GROQ_API_KEYS", "GROQ_API_KEY")
    summary = make_summary("Groq", "groq", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable GROQ_API_KEY/GROQ_API_KEYS value found.")
        return summary
    for index, key in enumerate(keys, start=1):
        label = f"GROQ_API_KEYS[{index}]"
        headers = {"Authorization": f"Bearer {key}"}
        status, models_payload, elapsed = http_json("GET", "https://api.groq.com/openai/v1/models", headers=headers)
        configured_text_models = model_candidates(
            ("GROQ_TRANSLATION_MODELS", "GROQ_TRANSLATION_MODEL"),
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "gemma2-9b-it"],
        )
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            last_detail = "Direct Groq chat probe did not succeed."
            last_model = configured_text_models[0] if configured_text_models else ""
            last_status: int | None = None
            last_elapsed = 0
            for model in configured_text_models:
                status, payload, elapsed = http_json(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    payload=openai_chat_payload(TRANSLATION_PROMPT, model),
                )
                success, detail = ok_from_chat(payload)
                if status == 200 and success:
                    add_result(summary, f"{label} direct text/translation", "pass", detail, model=model, http_status=status, elapsed_ms=elapsed)
                    break
                last_detail = detail
                last_model = model
                last_status = status
                last_elapsed = elapsed
            else:
                add_result(summary, f"{label} direct text/translation", "fail", last_detail, model=last_model, http_status=last_status, elapsed_ms=last_elapsed)
            continue
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_model = select_model(
            ids,
            configured_text_models,
            ["llama", "gemma", "mixtral"],
        )
        vision_model = select_model(ids, [], ["vision", "llama-4", "scout", "maverick", "vl"])
        add_result(summary, f"{label} auth/models", "pass", f"{len(ids)} models listed.", model=text_model, http_status=status, elapsed_ms=elapsed)
        if text_model:
            status, payload, elapsed = http_json(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} text/translation", "pass" if status == 200 and success else "fail", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
        if vision_model:
            content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": RED_DOT_DATA_URL}}]
            status, payload, elapsed = http_json(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                payload=openai_chat_payload(content, vision_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} vision", "pass" if status == 200 and success else "fail", detail, model=vision_model, http_status=status, elapsed_ms=elapsed)
        else:
            add_result(summary, f"{label} vision", "skip", "No obvious vision model found in model list.")
    return summary


def test_mistral() -> ProviderSummary:
    keys = provider_keys("MISTRAL_API_KEYS", "MISTRAL_API_KEY")
    summary = make_summary("Mistral", "mistral", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable MISTRAL_API_KEY/MISTRAL_API_KEYS value found.")
        return summary
    for index, key in enumerate(keys, start=1):
        label = f"MISTRAL_API_KEYS[{index}]"
        headers = {"Authorization": f"Bearer {key}"}
        status, models_payload, elapsed = http_json("GET", "https://api.mistral.ai/v1/models", headers=headers)
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            continue
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_model = select_model(
            ids,
            model_candidates(
                ("MISTRAL_TRANSLATION_MODELS", "MISTRAL_TRANSLATION_MODEL"),
                ["mistral-small-latest", "mistral-medium-latest", "ministral-8b-latest"],
            ),
            ["mistral", "ministral"],
        )
        vision_candidates = []
        for model in ids:
            lower = model.lower()
            if any(needle in lower for needle in ["pixtral", "mistral-small", "mistral-medium", "mistral-large", "ministral"]):
                vision_candidates.append(model)
        add_result(summary, f"{label} auth/models", "pass", f"{len(ids)} models listed.", model=text_model, http_status=status, elapsed_ms=elapsed)
        if text_model:
            status, payload, elapsed = http_json(
                "POST",
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} text/translation", "pass" if status == 200 and success else "fail", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
        tested = False
        last_detail = "Vision request failed."
        last_status: int | None = None
        last_elapsed = 0
        for model in vision_candidates[:6]:
            content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": RED_DOT_DATA_URL}]
            status, payload, elapsed = http_json(
                "POST",
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                payload=openai_chat_payload(content, model),
            )
            success, detail = ok_from_chat(payload)
            if status == 200 and success:
                add_result(summary, f"{label} vision", "pass", detail, model=model, http_status=status, elapsed_ms=elapsed)
                tested = True
                break
            last_detail = detail
            last_status = status
            last_elapsed = elapsed
        if not tested:
            if vision_candidates:
                add_result(summary, f"{label} vision", "fail", last_detail, model=vision_candidates[0], http_status=last_status, elapsed_ms=last_elapsed)
            else:
                add_result(summary, f"{label} vision", "skip", "No obvious vision-capable model candidate found.")
    return summary


def test_openrouter() -> ProviderSummary:
    keys = provider_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
    summary = make_summary("OpenRouter", "openrouter", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable OPENROUTER_API_KEY/OPENROUTER_API_KEYS value found.")
        return summary
    for index, key in enumerate(keys, start=1):
        label = f"OPENROUTER_API_KEYS[{index}]"
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "http://127.0.0.1",
            "X-Title": "Free Manga Translator API Key Test",
        }
        status, models_payload, elapsed = http_json("GET", "https://openrouter.ai/api/v1/models", headers=headers)
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            continue
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_ids = []
        vision_ids = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id", "")
            architecture = item.get("architecture", {}) or {}
            input_modalities = item.get("input_modalities") or architecture.get("input_modalities") or []
            if not input_modalities:
                input_modalities = ["text"]
            if "text" in input_modalities:
                text_ids.append(model_id)
            if "image" in input_modalities:
                vision_ids.append(model_id)
        preferred_text_models = model_candidates(
            ("OPENROUTER_TRANSLATION_MODELS", "OPENROUTER_TRANSLATION_MODEL"),
            [
                "openrouter/free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "qwen/qwen-2.5-72b-instruct:free",
            ],
        )
        text_candidates = []
        available_text_ids = text_ids or ids
        for model in preferred_text_models:
            if (model == "openrouter/free" or model in available_text_ids) and model not in text_candidates:
                text_candidates.append(model)
        for model in available_text_ids:
            lowered = model.lower()
            if (":free" in lowered or "llama" in lowered or "qwen" in lowered or "gemini" in lowered) and model not in text_candidates:
                text_candidates.append(model)
        preferred_vision_models = [
            "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "qwen/qwen2.5-vl-7b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.0-flash-001",
        ]
        vision_candidates = []
        for model in preferred_vision_models:
            if model in vision_ids and model not in vision_candidates:
                vision_candidates.append(model)
        for model in vision_ids:
            lowered = model.lower()
            if (":free" in lowered or "qwen" in lowered or "vision" in lowered) and model not in vision_candidates:
                vision_candidates.append(model)
        add_result(summary, f"{label} auth/models", "pass", f"{len(ids)} models listed; {len(vision_ids)} advertise image input.", model=text_candidates[0] if text_candidates else "", http_status=status, elapsed_ms=elapsed)
        if text_candidates:
            last_detail = "No OpenRouter text model candidate succeeded."
            last_model = text_candidates[0]
            last_status: int | None = None
            last_elapsed = 0
            for text_model in text_candidates[:12]:
                status, payload, elapsed = http_json(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
                )
                success, detail = ok_from_chat(payload)
                if status == 200 and success:
                    add_result(summary, f"{label} text/translation", "pass", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
                    break
                last_detail = detail
                last_model = text_model
                last_status = status
                last_elapsed = elapsed
            else:
                add_result(summary, f"{label} text/translation", "fail", last_detail, model=last_model, http_status=last_status, elapsed_ms=last_elapsed)
        if vision_candidates:
            content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": RED_DOT_DATA_URL}}]
            last_detail = "No OpenRouter vision model candidate succeeded."
            last_model = vision_candidates[0]
            last_status: int | None = None
            last_elapsed = 0
            for vision_model in vision_candidates[:12]:
                status, payload, elapsed = http_json(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    payload=openai_chat_payload(content, vision_model),
                )
                success, detail = ok_from_chat(payload)
                if status == 200 and success:
                    add_result(summary, f"{label} vision", "pass", detail, model=vision_model, http_status=status, elapsed_ms=elapsed)
                    break
                last_detail = detail
                last_model = vision_model
                last_status = status
                last_elapsed = elapsed
            else:
                add_result(summary, f"{label} vision", "fail", last_detail, model=last_model, http_status=last_status, elapsed_ms=last_elapsed)
        else:
            add_result(summary, f"{label} vision", "skip", "No image-input model advertised by model list.")
    return summary


def test_github() -> ProviderSummary:
    keys = provider_keys("GITHUB_API_KEYS", "GITHUB_API_KEY")
    summary = make_summary("GitHub", "github", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable GITHUB_API_KEY/GITHUB_API_KEYS value found.")
        return summary
    for index, key in enumerate(keys, start=1):
        label = f"GITHUB_API_KEYS[{index}]"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        status, user_payload, elapsed = http_json("GET", "https://api.github.com/user", headers=headers)
        add_result(summary, f"{label} github-rest-auth", "pass" if status == 200 else "fail", "GitHub REST token accepted." if status == 200 else user_payload, http_status=status, elapsed_ms=elapsed)
        status, models_payload, elapsed = http_json("GET", "https://models.github.ai/catalog/models", headers=headers)
        if status != 200:
            add_result(summary, f"{label} models/catalog", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            add_result(summary, f"{label} text/translation", "skip", "GitHub Models catalog did not authorize/list models.")
            add_result(summary, f"{label} vision", "skip", "GitHub Models catalog did not authorize/list models.")
            continue
        if isinstance(models_payload, list):
            data = models_payload
        elif isinstance(models_payload, dict):
            data = models_payload.get("models", models_payload.get("data", []))
        else:
            data = []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_model = select_model(
            ids,
            model_candidates(
                ("GITHUB_TRANSLATION_MODELS", "GITHUB_TRANSLATION_MODEL"),
                ["openai/gpt-4.1-mini", "openai/gpt-4o-mini", "mistral-ai/mistral-small-2503"],
            ),
            ["mini", "gpt", "mistral"],
        )
        vision_ids = []
        for item in data:
            if not isinstance(item, dict):
                continue
            modalities = item.get("supported_input_modalities") or item.get("input_modalities") or []
            modalities_text = json.dumps(modalities).lower()
            if "image" in modalities_text or "vision" in modalities_text:
                vision_ids.append(item.get("id", ""))
        vision_model = select_model(vision_ids, ["openai/gpt-4.1", "openai/gpt-4o"], ["vision", "gpt"])
        add_result(summary, f"{label} models/catalog", "pass", f"{len(ids)} models listed; {len(vision_ids)} mention image/vision.", model=text_model, http_status=status, elapsed_ms=elapsed)
        if text_model:
            status, payload, elapsed = http_json(
                "POST",
                "https://models.github.ai/inference/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} text/translation", "pass" if status == 200 and success else "fail", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
        if vision_model:
            content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": RED_DOT_DATA_URL}}]
            status, payload, elapsed = http_json(
                "POST",
                "https://models.github.ai/inference/chat/completions",
                headers=headers,
                payload=openai_chat_payload(content, vision_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} vision", "pass" if status == 200 and success else "fail", detail, model=vision_model, http_status=status, elapsed_ms=elapsed)
        else:
            add_result(summary, f"{label} vision", "skip", "No vision-capable GitHub Models candidate found in catalog.")
    return summary


def test_nvidia() -> ProviderSummary:
    keys = provider_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY")
    summary = make_summary("NVIDIA NIM", "nvidia", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable NVIDIA_API_KEY/NVIDIA_NIM_API_KEY value found.")
        return summary
    base_url = ENV_VALUES.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    text_model_candidates = model_candidates(
        ("NVIDIA_NIM_TRANSLATION_MODELS", "NVIDIA_NIM_TRANSLATION_MODEL"),
        ["meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct", "mistralai/mistral-7b-instruct-v0.3"],
    )
    for index, key in enumerate(keys, start=1):
        label = f"NVIDIA_NIM_API_KEYS[{index}]"
        headers = {"Authorization": f"Bearer {key}"}
        status, models_payload, elapsed = http_json("GET", f"{base_url}/models", headers=headers)
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            continue
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_model = select_model(
            ids,
            text_model_candidates,
            ["llama", "nemotron", "mistral"],
        )
        vision_model = select_model(
            ids,
            ["meta/llama-3.2-11b-vision-instruct", "nvidia/vila", "microsoft/phi-3-vision-128k-instruct"],
            ["vision", "vila", "vl", "qwen", "maverick", "scout"],
        )
        add_result(summary, f"{label} auth/models", "pass", f"{len(ids)} models listed.", model=text_model, http_status=status, elapsed_ms=elapsed)
        if text_model:
            status, payload, elapsed = http_json(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} text/translation", "pass" if status == 200 and success else "fail", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
        if vision_model:
            content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": RED_DOT_DATA_URL}}]
            status, payload, elapsed = http_json(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                payload=openai_chat_payload(content, vision_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} vision", "pass" if status == 200 and success else "fail", detail, model=vision_model, http_status=status, elapsed_ms=elapsed)
        else:
            add_result(summary, f"{label} vision", "skip", "No obvious vision-capable NVIDIA model found in model list.")
    return summary


def test_cerebras() -> ProviderSummary:
    keys = provider_keys("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY", "CEREBERAS_API_KEYS", "CEREBERAS_API_KEY")
    summary = make_summary("Cerebras", "cerebras", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable CEREBRAS_API_KEY/CEREBRAS_API_KEYS value found.")
        return summary
    base_url = ENV_VALUES.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
    text_model_candidates = model_candidates(
        ("CEREBRAS_TRANSLATION_MODELS", "CEREBRAS_TRANSLATION_MODEL"),
        ["gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b"],
    )
    for index, key in enumerate(keys, start=1):
        label = f"CEREBRAS_API_KEYS[{index}]"
        headers = {"Authorization": f"Bearer {key}"}
        status, models_payload, elapsed = http_json("GET", f"{base_url}/models", headers=headers)
        if status != 200:
            add_result(summary, f"{label} auth/models", "fail", models_payload, http_status=status, elapsed_ms=elapsed)
            continue
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        ids = [item.get("id", "") for item in data if isinstance(item, dict)]
        text_model = select_model(ids, text_model_candidates, ["llama", "gpt-oss", "qwen"])
        add_result(summary, f"{label} auth/models", "pass", f"{len(ids)} models listed.", model=text_model, http_status=status, elapsed_ms=elapsed)
        if text_model:
            status, payload, elapsed = http_json(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, text_model),
            )
            success, detail = ok_from_chat(payload)
            add_result(summary, f"{label} text/translation", "pass" if status == 200 and success else "fail", detail, model=text_model, http_status=status, elapsed_ms=elapsed)
        add_result(summary, f"{label} vision", "skip", "Cerebras has no vision_ocr capability configured in api_manager.py.")
    return summary


def test_fireworks() -> ProviderSummary:
    keys = provider_keys("FIREWORKS_API_KEYS", "FIREWORKS_API_KEY")
    summary = make_summary("Fireworks", "fireworks", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable FIREWORKS_API_KEY/FIREWORKS_API_KEYS value found.")
        return summary
    text_model_candidates = model_candidates(
        ("FIREWORKS_TRANSLATION_MODELS", "FIREWORKS_TRANSLATION_MODEL"),
        [
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/qwen3p235b-a22b",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        ],
    )
    for index, key in enumerate(keys, start=1):
        label = f"FIREWORKS_API_KEYS[{index}]"
        headers = {"Authorization": f"Bearer {key}"}
        last_detail = "No Fireworks text model candidate succeeded."
        last_model = text_model_candidates[0] if text_model_candidates else ""
        last_status: int | None = None
        last_elapsed = 0
        for model in text_model_candidates:
            status, payload, elapsed = http_json(
                "POST",
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, model),
            )
            success, detail = ok_from_chat(payload)
            if status == 200 and success:
                add_result(summary, f"{label} text/translation", "pass", detail, model=model, http_status=status, elapsed_ms=elapsed)
                break
            last_detail, last_model, last_status, last_elapsed = detail, model, status, elapsed
        else:
            add_result(summary, f"{label} text/translation", "fail", last_detail, model=last_model, http_status=last_status, elapsed_ms=last_elapsed)
        add_result(summary, f"{label} vision", "skip", "Fireworks has no vision_ocr capability configured in api_manager.py.")
    return summary


def test_cloudflare() -> ProviderSummary:
    keys = provider_keys("CLOUDFLARE_WORKERS_API_KEYS", "CLOUDFLARE_WORKERS_API_KEY", "CLOUDFLARE_API_KEYS", "CLOUDFLARE_API_KEY")
    account_ids = csv_values("CLOUDFLARE_ACCOUNT_IDS") or csv_values("CLOUDFLARE_ACCOUNT_ID")
    summary = make_summary("Cloudflare Workers AI", "cloudflare", keys)
    if not keys:
        add_result(summary, "configured", "skip", "No usable CLOUDFLARE_WORKERS_API_KEY/CLOUDFLARE_WORKERS_API_KEYS value found.")
        return summary
    if not account_ids:
        add_result(summary, "configured", "fail", "Keys present but CLOUDFLARE_ACCOUNT_IDS/CLOUDFLARE_ACCOUNT_ID is missing.")
        return summary
    if len(account_ids) not in (1, len(keys)):
        add_result(summary, "configured", "fail", f"CLOUDFLARE_ACCOUNT_IDS count ({len(account_ids)}) must be 1 or match key count ({len(keys)}).")
        return summary
    text_model_candidates = model_candidates(
        ("CLOUDFLARE_TRANSLATION_MODELS", "CLOUDFLARE_TRANSLATION_MODEL"),
        ["@cf/qwen/qwen3-30b-a3b-fp8", "@cf/qwen/qwq-32b", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "@cf/qwen/qwen1.5-14b-chat-awq", "@cf/meta/llama-3.1-8b-instruct"],
    )
    for index, key in enumerate(keys, start=1):
        label = f"CLOUDFLARE_WORKERS_API_KEYS[{index}]"
        account_id = account_ids[0] if len(account_ids) == 1 else account_ids[index - 1]
        headers = {"Authorization": f"Bearer {key}"}
        url = f"https://api.cloudflare.com/client/v4/accounts/{urllib.parse.quote(account_id)}/ai/v1/chat/completions"
        last_detail = "No Cloudflare Workers AI model candidate succeeded."
        last_model = text_model_candidates[0] if text_model_candidates else ""
        last_status: int | None = None
        last_elapsed = 0
        for model in text_model_candidates:
            status, payload, elapsed = http_json(
                "POST",
                url,
                headers=headers,
                payload=openai_chat_payload(TRANSLATION_PROMPT, model),
            )
            success, detail = ok_from_chat(payload)
            if status == 200 and success:
                add_result(summary, f"{label} text/translation", "pass", detail, model=model, http_status=status, elapsed_ms=elapsed)
                break
            last_detail, last_model, last_status, last_elapsed = detail, model, status, elapsed
        else:
            add_result(summary, f"{label} text/translation", "fail", last_detail, model=last_model, http_status=last_status, elapsed_ms=last_elapsed)
        add_result(summary, f"{label} vision", "skip", "Cloudflare has no vision_ocr capability configured in api_manager.py.")
    return summary


def render_report(summaries: list[ProviderSummary]) -> str:
    lines = [
        "# API Key And Extension Usage Test Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Key Answer",
        "",
        "- The local 8-step pipeline can use provider API keys from `core_pipeline/.env` during Step 7 when `USE_API_TRANSLATION` is enabled or set to `auto` with keys present.",
        "- Step 7 no longer uses a curated per-sample translation database; configured provider LLMs are the production translation path.",
        "- Multiple keys per provider are supported through comma-separated `*_API_KEYS` values; the runtime rotates the starting key per provider call and falls through on failures.",
        "- Runtime provider calls try configured model lists before falling through, and auth-blocked keys are disabled for the rest of the run.",
        "- Provider keys must stay server-side only. They must never be pasted into, bundled with, or exposed by the browser extension.",
        "",
        "## Provider Test Results",
        "",
        "| Provider | Configured | Used By Current Pipeline | Required Now | Check | Status | HTTP | Model | Detail |",
        "|---|---:|---:|---:|---|---|---:|---|---|",
    ]
    for summary in summaries:
        for result in summary.results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary.provider,
                        "yes" if summary.configured else "no",
                        "yes" if summary.uses_current_pipeline else "no",
                        "yes" if summary.required_for_current_pipeline else "no",
                        result.check,
                        result.status,
                        "" if result.http_status is None else str(result.http_status),
                        result.model,
                        result.detail.replace("|", "\\|").replace("\n", " "),
                    ]
                )
                + " |"
            )
    translation_ready = sorted(
        {
            summary.provider
            for summary in summaries
            for result in summary.results
            if "text/translation" in result.check and result.status == "pass"
        }
    )
    vision_ready = sorted(
        {
            summary.provider
            for summary in summaries
            for result in summary.results
            if "vision" in result.check and result.status == "pass"
        }
    )
    failing_or_limited = sorted(
        {
            summary.provider
            for summary in summaries
            for result in summary.results
            if result.status == "fail"
        }
    )
    lines.extend(
        [
            "",
            "## Live Result Summary",
            "",
            f"- Translation-capable providers in this run: {', '.join(translation_ready) if translation_ready else 'none'}.",
            f"- Vision-capable providers in this run: {', '.join(vision_ready) if vision_ready else 'none'}.",
            f"- Providers with at least one failure/limit in this run: {', '.join(failing_or_limited) if failing_or_limited else 'none'}.",
            f"- Current recommended provider order: `{ENV_VALUES.get('TRANSLATION_PROVIDER_ORDER', 'mistral,github,nvidia,gemini,openrouter,groq')}` with `PREFERRED_PROVIDER={ENV_VALUES.get('PREFERRED_PROVIDER', 'mistral')}`.",
            "- Keep failing/quota-limited providers later in the order so the pipeline can fall through without blocking successful translation.",
            "- `401`/`403` completion failures are treated as key/account/network failures; the runtime disables that specific key and tries the next configured key.",
            "",
            "## How To Use The Extension Right Now",
            "",
            "1. Open PowerShell in `D:\\Desktop\\translator D\\app`.",
            "2. Start the legacy local server:",
            "",
            "```powershell",
            "$env:PYTHONIOENCODING='utf-8'",
            "$py='D:\\Desktop\\translator D\\free-manga-translator-codex\\free-manga-translator-codex\\.venv\\Scripts\\python.exe'",
            "& $py run_extension_pipeline_server.py --host 127.0.0.1 --port 8765",
            "```",
            "",
            "3. Load `D:\\Desktop\\translator D\\app\\extension` as an unpacked extension in Chrome or Brave.",
            "4. In the popup, set Local Pipeline URL to `http://127.0.0.1:8765/translate`.",
            "5. Select source language: Japanese `ja`, Korean `ko`, or Chinese `zh`.",
            "6. Use `Translate Page` for whole-page image translation or `Selection Panel` for a specific crop.",
            "",
            "## FastAPI Companion Alternative",
            "",
            "```powershell",
            "$env:PYTHONIOENCODING='utf-8'",
            "$py='D:\\Desktop\\translator D\\free-manga-translator-codex\\free-manga-translator-codex\\.venv\\Scripts\\python.exe'",
            "& $py -m uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8766",
            "```",
            "",
            "Use this URL in the extension popup:",
            "",
            "```text",
            "http://127.0.0.1:8766/v1/translate-image",
            "```",
            "",
            "## Important Notes",
            "",
            "- Do not paste provider keys into the extension popup; provider calls must stay in the local Python/server layer.",
            "- If direct cloud calls are added later, they must still be proxied through a trusted backend so secrets are never shipped to the browser.",
            "- Vision API availability is provider/model-specific and may require paid quota or special token permissions.",
            "- Translation API availability means the provider can generate text for Step 7; OCR, inpainting, layout, and typesetting still run through the local pipeline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    # This prints live API response text (result.detail below), which can carry non-ASCII
    # (e.g. the Japanese translation-probe prompt's echoed response) -- on a plain Windows
    # console (cp1252) that print() raises UnicodeEncodeError and fails this test for a
    # reason that has nothing to do with the API keys it checks. Every other diagnostics
    # entrypoint that prints CJK text either imports run_extension_pipeline_server (which
    # reconfigures stdout to UTF-8 on import) or does this itself; this one did neither.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Testing provider keys without printing secret values...")
    summaries = [
        test_gemini(),
        test_mistral(),
        test_github(),
        test_openrouter(),
        test_groq(),
        test_nvidia(),
        test_cerebras(),
        test_fireworks(),
        test_cloudflare(),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(summaries), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    for summary in summaries:
        configured = "configured" if summary.configured else "not configured"
        print(f"{summary.provider}: {configured}")
        for result in summary.results:
            model = f" model={result.model}" if result.model else ""
            http = f" http={result.http_status}" if result.http_status is not None else ""
            print(f"  - {result.check}: {result.status}{http}{model} :: {result.detail}")
    failures = [
        result
        for summary in summaries
        for result in summary.results
        if result.status == "fail"
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
