from __future__ import annotations

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

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from api_manager import API_MANAGER, ApiProviderUnavailable, ApiQuotaExhausted


TEXT_PROMPT = "Translate to concise natural English. Return only the translation.\nText: こんにちは"
VISION_PROMPT = "What is the main color of this image? Answer with one word."
RED_DOT_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
RED_DOT_BASE64 = RED_DOT_DATA_URL.split(",", 1)[1]
TIMEOUT_SECONDS = 30
PROBE_OUTPUT_TOKENS = 64

MODEL_ENVS = {
    "gemini": (("GEMINI_TRANSLATION_MODELS", "GEMINI_TRANSLATION_MODEL"), ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]),
    "mistral": (("MISTRAL_TRANSLATION_MODELS", "MISTRAL_TRANSLATION_MODEL"), ["mistral-large-latest", "mistral-small-latest"]),
    "dashscope": (("DASHSCOPE_TRANSLATION_MODELS", "DASHSCOPE_TRANSLATION_MODEL", "QWEN_TRANSLATION_MODELS", "QWEN_TRANSLATION_MODEL"), ["qwen-mt-plus", "qwen-mt-flash", "qwen-plus-latest"]),
    "github": (("GITHUB_TRANSLATION_MODELS", "GITHUB_TRANSLATION_MODEL"), ["openai/gpt-4o-mini"]),
    "fireworks": (("FIREWORKS_TRANSLATION_MODELS", "FIREWORKS_TRANSLATION_MODEL"), ["accounts/fireworks/models/qwen3p235b-a22b", "accounts/fireworks/models/qwen2p5-72b-instruct"]),
    "openrouter": (("OPENROUTER_TRANSLATION_MODELS", "OPENROUTER_TRANSLATION_MODEL"), ["qwen/qwen3-235b-a22b:free", "qwen/qwen-2.5-72b-instruct:free", "google/gemini-2.5-flash"]),
    "groq": (("GROQ_TRANSLATION_MODELS", "GROQ_TRANSLATION_MODEL"), ["qwen/qwen3-32b", "llama-3.3-70b-versatile"]),
    "cerebras": (("CEREBRAS_TRANSLATION_MODELS", "CEREBRAS_TRANSLATION_MODEL"), ["qwen-3-32b", "gpt-oss-120b", "llama-3.3-70b"]),
    "nvidia": (("NVIDIA_NIM_TRANSLATION_MODELS", "NVIDIA_NIM_TRANSLATION_MODEL"), ["qwen/qwen3-5-122b-a10b", "qwen/qwen3.5-397b-a17b", "meta/llama-3.1-8b-instruct"]),
    "cloudflare": (("CLOUDFLARE_TRANSLATION_MODELS", "CLOUDFLARE_TRANSLATION_MODEL"), ["@cf/qwen/qwen3-30b-a3b-fp8", "@cf/qwen/qwq-32b", "@cf/meta/llama-3.1-8b-instruct"]),
}

VISION_MODEL_ENVS = {
    "gemini": (("GEMINI_VISION_OCR_MODELS",), ["gemini-2.5-flash", "gemini-2.5-flash-lite"]),
    "github": (("GITHUB_VISION_OCR_MODELS",), ["openai/gpt-4o-mini"]),
    "groq": (("GROQ_VISION_OCR_MODELS",), ["meta-llama/llama-4-scout-17b-16e-instruct"]),
    "openrouter": (("OPENROUTER_VISION_OCR_MODELS",), ["qwen/qwen2.5-vl-72b-instruct:free", "google/gemini-2.5-flash"]),
}

OPENAI_ENDPOINTS = {
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "dashscope": "{dashscope_base_url}/chat/completions",
    "github": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "{cerebras_base_url}/chat/completions",
    "nvidia": "{base_url}/chat/completions",
    "fireworks": "https://api.fireworks.ai/inference/v1/chat/completions",
}


def configured(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped and "YOUR_" not in stripped and stripped not in {"-", "_"})


def csv_values(*names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        for raw in os.environ.get(name, "").split(","):
            value = raw.strip()
            if configured(value) and value not in values:
                values.append(value)
    return values


def model_candidates(provider: str, capability: str) -> list[str]:
    source = VISION_MODEL_ENVS if capability == "vision_ocr" else MODEL_ENVS
    env_names, defaults = source.get(provider, ((), []))
    configured_models = csv_values(*env_names)
    return configured_models or defaults


def secret_values() -> list[str]:
    values: list[str] = []
    for provider in API_MANAGER.providers:
        values.extend(API_MANAGER.provider_keys(provider))
    values.extend(API_MANAGER.cloudflare_account_ids())
    return [value for value in values if value]


def scrub(value: Any) -> str:
    rendered = str(value)
    for secret in sorted(set(secret_values()), key=len, reverse=True):
        rendered = rendered.replace(secret, "[REDACTED]")
    rendered = re.sub(r"Bearer\s+[A-Za-z0-9_.\-]+", "Bearer [REDACTED]", rendered)
    rendered = re.sub(r"key=([^&\s]+)", "key=[REDACTED]", rendered)
    return rendered[:700]


def http_json(method: str, url: str, headers: dict[str, str] | None = None, payload: dict[str, Any] | None = None) -> tuple[int, Any, int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "User-Agent": "free-manga-translator-key-validator/1.0", **(headers or {})},
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


def headers_for(provider: str, key: str) -> dict[str, str]:
    if provider == "github":
        return {"Authorization": f"Bearer {key}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if provider == "openrouter":
        return {"Authorization": f"Bearer {key}", "HTTP-Referer": "http://127.0.0.1", "X-Title": "Free Manga Translator Key Validator"}
    return {"Authorization": f"Bearer {key}"}


def openai_payload(model: str, prompt: str) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": PROBE_OUTPUT_TOKENS}


def openai_vision_payload(model: str) -> dict[str, Any]:
    content = [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": RED_DOT_DATA_URL}}]
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": PROBE_OUTPUT_TOKENS}


def response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first.get("message", {}), dict) else {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content", {}) if isinstance(candidate.get("content", {}), dict) else {}
        parts = content.get("parts", []) if isinstance(content.get("parts", []), list) else []
        return " ".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    return ""


def probe_gemini(key: str, model: str, capability: str) -> tuple[int, Any, int]:
    parts: list[dict[str, Any]] = [{"text": TEXT_PROMPT if capability == "translation" else VISION_PROMPT}]
    if capability == "vision_ocr":
        parts.append({"inline_data": {"mime_type": "image/png", "data": RED_DOT_BASE64}})
    return http_json(
        "POST",
        f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(key)}",
        payload={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0, "maxOutputTokens": PROBE_OUTPUT_TOKENS}},
    )


def probe_cloudflare(key: str, key_index: int, model: str) -> tuple[int, Any, int]:
    account_id = API_MANAGER.cloudflare_account_id_for_key_index(key_index)
    return http_json(
        "POST",
        f"https://api.cloudflare.com/client/v4/accounts/{urllib.parse.quote(account_id)}/ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload=openai_payload(model, TEXT_PROMPT),
    )


def probe_openai_compatible(provider: str, key: str, model: str, capability: str) -> tuple[int, Any, int]:
    endpoint = OPENAI_ENDPOINTS[provider]
    if provider == "nvidia":
        endpoint = endpoint.format(base_url=os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"))
    if provider == "dashscope":
        endpoint = endpoint.format(dashscope_base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/"))
    if provider == "cerebras":
        endpoint = endpoint.format(cerebras_base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/"))
    payload = openai_vision_payload(model) if capability == "vision_ocr" else openai_payload(model, TEXT_PROMPT)
    if provider == "dashscope" and model.startswith("qwen-mt"):
        payload["translation_options"] = {"source_lang": "auto", "target_lang": "English"}
    return http_json("POST", endpoint, headers=headers_for(provider, key), payload=payload)


def run_probe(provider: str, key: str, key_index: int, model: str, capability: str) -> tuple[int, Any, int]:
    if provider == "gemini":
        return probe_gemini(key, model, capability)
    if provider == "cloudflare":
        return probe_cloudflare(key, key_index, model)
    return probe_openai_compatible(provider, key, model, capability)


def dry_run_payload(providers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in providers:
        status = API_MANAGER.provider_status(provider)
        rows.append(
            {
                "provider": provider,
                "configured": status["configured"],
                "reason": status.get("reason", ""),
                "keyCount": status["keyCount"],
                "activeKeys": status["activeKeys"],
                "lockedKeys": status["lockedKeys"],
                "health": status["health"],
                "capabilities": status["capabilities"],
                "translationModels": model_candidates(provider, "translation"),
                "visionModels": model_candidates(provider, "vision_ocr") if "vision_ocr" in status["capabilities"] else [],
                "keyRoles": API_MANAGER._key_roles(provider, status["keyCount"]),
            }
        )
    return rows


def live_probe(providers: list[str], all_keys: bool, max_keys_per_provider: int, live_vision: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    capabilities = ["translation", "vision_ocr"] if live_vision else ["translation"]
    for provider in providers:
        status = API_MANAGER.provider_status(provider)
        if not status["configured"]:
            results.append({"provider": provider, "status": "skip", "detail": status.get("reason") or "not configured"})
            continue
        limit = status["keyCount"] if all_keys else min(status["keyCount"], max(1, max_keys_per_provider))
        for key_index in range(1, limit + 1):
            for capability in capabilities:
                if capability not in status["capabilities"]:
                    continue
                if capability == "vision_ocr" and provider not in VISION_MODEL_ENVS:
                    continue
                models = model_candidates(provider, capability)
                model = models[0] if models else ""
                if not model:
                    results.append({"provider": provider, "keyIndex": key_index, "capability": capability, "status": "skip", "detail": "no model configured"})
                    continue
                estimated_tokens = API_MANAGER.estimate_tokens(TEXT_PROMPT if capability == "translation" else VISION_PROMPT, output_tokens=PROBE_OUTPUT_TOKENS)
                try:
                    lease = API_MANAGER.reserve_key_index(provider, key_index, estimated_tokens, capability=capability)
                except (ApiQuotaExhausted, ApiProviderUnavailable) as error:
                    results.append({"provider": provider, "keyIndex": key_index, "capability": capability, "status": "skip", "detail": scrub(error)})
                    continue
                http_status, payload, elapsed_ms = run_probe(provider, lease.key, lease.key_index, model, capability)
                text = response_text(payload)
                ok = http_status == 200 and bool(text)
                if ok:
                    API_MANAGER.mark_success(lease, payload)
                else:
                    API_MANAGER.mark_failure(lease, http_status, scrub(payload))
                results.append(
                    {
                        "provider": provider,
                        "keyIndex": key_index,
                        "capability": capability,
                        "model": model,
                        "httpStatus": http_status,
                        "elapsedMs": elapsed_ms,
                        "status": "pass" if ok else "fail",
                        "detail": scrub(text or payload),
                    }
                )
    return results


def print_table(rows: list[dict[str, Any]], live: bool) -> None:
    if live:
        for row in rows:
            fields = [row.get("provider", ""), f"key={row.get('keyIndex', '-')}", row.get("capability", "-"), row.get("status", "-")]
            if row.get("httpStatus") is not None:
                fields.append(f"http={row['httpStatus']}")
            if row.get("model"):
                fields.append(f"model={row['model']}")
            print(" | ".join(str(item) for item in fields))
            if row.get("detail"):
                print(f"  {row['detail']}")
        return
    for row in rows:
        models = ", ".join(row["translationModels"][:3])
        caps = ", ".join(row["capabilities"])
        reason = f" reason={row['reason']}" if row.get("reason") else ""
        roles = row.get("keyRoles") or []
        roles_field = f" roles=[{', '.join(roles)}]" if roles and any(role != "*" for role in roles) else ""
        print(
            f"{row['provider']}: configured={row['configured']} health={row['health']} "
            f"keys={row['keyCount']} active={row['activeKeys']} locked={row['lockedKeys']} caps=[{caps}] models=[{models}]{roles_field}{reason}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate provider key visibility and optionally run quota-safe live probes.")
    parser.add_argument("--provider", action="append", choices=list(API_MANAGER.providers), help="Provider to inspect. Repeatable. Defaults to all providers.")
    parser.add_argument("--live", action="store_true", help="Run tiny live translation probes. This consumes provider requests/tokens.")
    parser.add_argument("--all-keys", action="store_true", help="With --live, probe every configured key instead of the first active key per provider.")
    parser.add_argument("--max-keys-per-provider", type=int, default=1, help="With --live, cap probed keys per provider unless --all-keys is used.")
    parser.add_argument("--live-vision", action="store_true", help="With --live, also test vision OCR-capable providers using a 1x1 PNG.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()

    providers = args.provider or list(API_MANAGER.providers)
    payload = live_probe(providers, args.all_keys, args.max_keys_per_provider, args.live_vision) if args.live else dry_run_payload(providers)
    if args.json:
        print(json.dumps({"live": args.live, "envFileLoaded": API_MANAGER.quota_status().get("envFileLoaded", ""), "results": payload}, indent=2, ensure_ascii=False))
    else:
        print(f"env={API_MANAGER.quota_status().get('envFileLoaded', '')}")
        print_table(payload, args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
