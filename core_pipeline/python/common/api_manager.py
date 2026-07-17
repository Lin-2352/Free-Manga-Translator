from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, BinaryIO

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from pipeline_paths import PROJECT_ROOT
from diagnostic_logger import write_diagnostic_event


def _acquire_cross_process_file_lock(lock_path: Path, timeout: float = 10.0) -> BinaryIO:
    """Blocks (briefly retrying) until an OS-level advisory lock on lock_path is acquired, or
    raises TimeoutError. Returns the open file handle; the caller must pass it to
    _release_cross_process_file_lock to release it. A plain in-process threading.Lock (which
    ApiManager also has) only serializes threads within ONE Python process -- it does nothing for
    two separate processes (e.g. the always-running backend plus a separate CLI/batch script)
    that each construct their own ApiManager pointed at the same state file.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    deadline = time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return fh
            except OSError:
                if time.monotonic() >= deadline:
                    fh.close()
                    raise TimeoutError(f"timed out waiting for cross-process lock: {lock_path}")
                time.sleep(0.02)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return fh


def _release_cross_process_file_lock(fh: BinaryIO) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


DAILY_LIMIT_MESSAGE = "Your daily translation limit has been reached to protect API quotas. Please try again tomorrow."
LOGGER = logging.getLogger("free_manga_translator.api_manager")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    env_names: tuple[str, ...]
    default_daily_tokens: int
    default_daily_requests: int
    default_minute_tokens: int
    default_minute_requests: int
    capabilities: tuple[str, ...] = ("translation",)
    required_env: tuple[str, ...] = ()
    default_limit_scope: str = "provider"


@dataclass(frozen=True)
class ApiKeyLease:
    provider: str
    key_index: int
    key_hash: str
    key: str
    reserved_tokens: int
    capability: str


class ApiQuotaExhausted(RuntimeError):
    pass


class ApiProviderAuthLocked(RuntimeError):
    pass


class ApiRateLimited(RuntimeError):
    pass


class ApiProviderUnavailable(RuntimeError):
    pass


class ApiManager:
    def __init__(self, state_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._state_path = state_path or PROJECT_ROOT / "runtime_samples" / "api_quota_state.json"
        self._lock_path = self._state_path.with_suffix(".lock")
        self._state: dict[str, Any] = self._read_state()
        self._env_path: Path | None = None
        self._missing_env_warning_emitted = False
        self._load_env()

    @contextlib.contextmanager
    def _synced(self):
        """The critical section every state-touching public method must use instead of
        `with self._lock:` directly. Acquires a cross-process file lock (serializing every
        ApiManager instance across every process pointed at this state file) and refreshes
        self._state from disk BEFORE yielding, so this operation's admission/rotation/lock
        decisions are made against the current on-disk truth, not a stale snapshot from whenever
        this instance was constructed or last wrote. The existing threading.Lock is kept nested
        inside as defense in depth for threads sharing one instance; it alone cannot protect
        against a second process, which is the actual gap this closes.
        """
        fh = _acquire_cross_process_file_lock(self._lock_path)
        try:
            with self._lock:
                self._state = self._read_state()
                yield
        finally:
            _release_cross_process_file_lock(fh)

    @property
    def providers(self) -> dict[str, ProviderConfig]:
        return {
            "gemini": ProviderConfig(
                "gemini",
                ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
                800_000,
                900,
                200_000,
                8,
                ("translation", "vision_ocr", "dialogue_classifier"),
            ),
            "mistral": ProviderConfig(
                "mistral",
                ("MISTRAL_API_KEYS", "MISTRAL_API_KEY"),
                250_000,
                300,
                40_000,
                4,
            ),
            "github": ProviderConfig(
                "github",
                ("GITHUB_API_KEYS", "GITHUB_API_KEY"),
                200_000,
                200,
                20_000,
                2,
                ("translation", "vision_ocr", "dialogue_classifier"),
            ),
            "openrouter": ProviderConfig(
                "openrouter",
                ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
                90_000,
                50,
                30_000,
                20,
                ("translation", "vision_ocr", "dialogue_classifier"),
            ),
            "groq": ProviderConfig(
                "groq",
                ("GROQ_API_KEYS", "GROQ_API_KEY"),
                500_000,
                1_000,
                120_000,
                12,
                ("translation", "vision_ocr", "dialogue_classifier"),
            ),
            "cerebras": ProviderConfig(
                "cerebras",
                ("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY", "CEREBERAS_API_KEYS", "CEREBERAS_API_KEY"),
                250_000,
                300,
                40_000,
                4,
            ),
            "nvidia": ProviderConfig(
                "nvidia",
                ("NVIDIA_API_KEYS", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY"),
                200_000,
                200,
                30_000,
                3,
                ("translation", "vision_ocr", "dialogue_classifier"),
            ),
            "fireworks": ProviderConfig(
                "fireworks",
                ("FIREWORKS_API_KEYS", "FIREWORKS_API_KEY"),
                250_000,
                300,
                40_000,
                4,
            ),
            "cloudflare": ProviderConfig(
                "cloudflare",
                ("CLOUDFLARE_WORKERS_API_KEYS", "CLOUDFLARE_WORKERS_API_KEY", "CLOUDFLARE_API_KEYS", "CLOUDFLARE_API_KEY"),
                250_000,
                300,
                40_000,
                4,
            ),
        }

    def _load_env(self) -> None:
        candidates: list[Path] = []
        explicit = os.environ.get("FMT_ENV_FILE", "").strip()
        if explicit:
            candidates.append(Path(explicit).expanduser())
        candidates.extend(
            [
                PROJECT_ROOT / ".env",
                PROJECT_ROOT.parent / ".env",
                PROJECT_ROOT.parents[1] / ".env",
                Path.cwd() / ".env",
            ]
        )

        env_path = next((path.resolve() for path in candidates if path.expanduser().exists()), None)
        if env_path is None:
            if not self._missing_env_warning_emitted:
                LOGGER.warning(
                    "No .env file found. Checked: %s",
                    ", ".join(str(path) for path in candidates),
                )
                self._missing_env_warning_emitted = True
            return
        self._env_path = env_path
        self._missing_env_warning_emitted = False
        if load_dotenv is not None:
            load_dotenv(env_path, override=True)
            return
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    def _read_state(self) -> dict[str, Any]:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
        return {}

    def _write_state_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._state_path)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _minute_bucket(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    def _next_minute_iso(self) -> str:
        now = datetime.now(timezone.utc)
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return next_minute.isoformat()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _locked_until_iso(self) -> str:
        return (datetime.now(timezone.utc) + timedelta(hours=self.lock_hours)).isoformat()

    def _provider_state_locked(self, provider: str) -> dict[str, Any]:
        self._state.setdefault("providers", {})
        provider_state = self._state["providers"].setdefault(provider, {"keys": {}, "date": self._today()})
        if provider_state.get("date") != self._today():
            for key_state in provider_state.get("keys", {}).values():
                key_state["tokensUsed"] = 0
                key_state["requestsUsed"] = 0
                key_state["softLocked"] = False
                key_state["softLockReason"] = ""
            provider_state["softLocked"] = False
            provider_state["softLockReason"] = ""
            if not provider_state.get("lockedUntil"):
                provider_state["status"] = "healthy"
            provider_state["date"] = self._today()
        return provider_state

    def _key_hash(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _configured(self, value: str | None) -> bool:
        if not value:
            return False
        stripped = value.strip()
        return bool(stripped and "YOUR_" not in stripped and stripped not in {"-", "_"})

    def _csv_env(self, *names: str) -> list[str]:
        self._load_env()
        keys: list[str] = []
        for name in names:
            candidate_names = [name]
            candidate_names.extend(f"{name}_{index}" for index in range(1, 51))
            for candidate_name in candidate_names:
                for raw in os.environ.get(candidate_name, "").split(","):
                    value = raw.strip()
                    if self._configured(value) and value not in keys:
                        keys.append(value)
        return keys

    def provider_keys(self, provider: str) -> list[str]:
        config = self.providers.get(provider)
        if not config:
            return []
        return self._csv_env(*config.env_names)

    def cloudflare_account_ids(self) -> list[str]:
        return self._csv_env("CLOUDFLARE_ACCOUNT_IDS", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_ACCOUNT_IDS")

    def cloudflare_account_id_for_key_index(self, key_index: int) -> str:
        account_ids = self.cloudflare_account_ids()
        keys = self.provider_keys("cloudflare")
        if not account_ids:
            raise ApiProviderUnavailable("cloudflare: missing required env: CLOUDFLARE_ACCOUNT_IDS or CLOUDFLARE_ACCOUNT_ID")
        if len(account_ids) == 1:
            return account_ids[0]
        if len(account_ids) != len(keys):
            raise ApiProviderUnavailable(
                f"cloudflare: account id count ({len(account_ids)}) must be 1 or match key count ({len(keys)})"
            )
        if key_index < 1 or key_index > len(account_ids):
            raise ApiProviderUnavailable(f"cloudflare: no account id for key index {key_index}")
        return account_ids[key_index - 1]

    @property
    def soft_cap_ratio(self) -> float:
        raw = os.environ.get("API_MANAGER_SOFT_CAP_RATIO", "0.80").strip()
        try:
            return max(0.1, min(0.95, float(raw)))
        except ValueError:
            return 0.80

    def _soft_cap_ratio(self, provider: str) -> float:
        raw = os.environ.get(f"{provider.upper()}_SOFT_CAP_RATIO", os.environ.get("API_MANAGER_SOFT_CAP_RATIO", "0.80")).strip()
        try:
            return max(0.1, min(0.95, float(raw)))
        except ValueError:
            return 0.80

    def _minute_soft_cap_ratio(self, provider: str) -> float:
        raw = os.environ.get(
            f"{provider.upper()}_MINUTE_SOFT_CAP_RATIO",
            os.environ.get("API_MANAGER_MINUTE_SOFT_CAP_RATIO", "0.90"),
        ).strip()
        try:
            return max(0.1, min(1.0, float(raw)))
        except ValueError:
            return 0.90

    def _limit_scope(self, provider: str) -> str:
        config = self.providers.get(provider)
        default = config.default_limit_scope if config else "provider"
        raw = os.environ.get(f"{provider.upper()}_LIMIT_SCOPE", os.environ.get("API_MANAGER_DEFAULT_LIMIT_SCOPE", default)).strip().lower()
        aliases = {
            "account": "provider",
            "org": "provider",
            "organization": "provider",
            "workspace": "provider",
            "project": "provider",
            "shared": "provider",
            "global": "provider",
            "provider": "provider",
            "key": "key",
            "per_key": "key",
            "per-key": "key",
        }
        return aliases.get(raw, default)

    @property
    def lock_hours(self) -> int:
        raw = os.environ.get("API_MANAGER_EXHAUSTED_LOCK_HOURS", "24").strip()
        try:
            return max(1, min(168, int(raw)))
        except ValueError:
            return 24

    def _token_limit(self, provider: str) -> int:
        config = self.providers.get(provider)
        default = config.default_daily_tokens if config else 250_000
        raw = os.environ.get(f"{provider.upper()}_DAILY_TOKEN_LIMIT", os.environ.get("API_MANAGER_DEFAULT_DAILY_TOKENS", ""))
        try:
            return max(1_000, int(raw)) if raw else default
        except ValueError:
            return default

    def _request_limit(self, provider: str) -> int:
        config = self.providers.get(provider)
        default = config.default_daily_requests if config else 250
        raw = os.environ.get(f"{provider.upper()}_DAILY_REQUEST_LIMIT", os.environ.get("API_MANAGER_DEFAULT_DAILY_REQUESTS", ""))
        try:
            return max(1, int(raw)) if raw else default
        except ValueError:
            return default

    def _minute_token_limit(self, provider: str) -> int:
        config = self.providers.get(provider)
        default = config.default_minute_tokens if config else 20_000
        raw = os.environ.get(f"{provider.upper()}_TOKENS_PER_MINUTE", os.environ.get("API_MANAGER_DEFAULT_TOKENS_PER_MINUTE", ""))
        try:
            return max(1_000, int(raw)) if raw else default
        except ValueError:
            return default

    def _minute_request_limit(self, provider: str) -> int:
        config = self.providers.get(provider)
        default = config.default_minute_requests if config else 2
        raw = os.environ.get(f"{provider.upper()}_REQUESTS_PER_MINUTE", os.environ.get("API_MANAGER_DEFAULT_REQUESTS_PER_MINUTE", ""))
        try:
            return max(1, int(raw)) if raw else default
        except ValueError:
            return default

    def estimate_tokens(self, *parts: object, output_tokens: int | None = None) -> int:
        text = "\n".join(str(part or "") for part in parts)
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("o200k_base")
            prompt_tokens = len(encoder.encode(text))
        except Exception:
            prompt_tokens = int(math.ceil(len(text) / 3.4)) + 12
        return max(1, prompt_tokens + int(output_tokens or 0))

    def _is_locked(self, key_state: dict[str, Any]) -> bool:
        locked_until = str(key_state.get("lockedUntil") or "")
        if not locked_until:
            return bool(key_state.get("softLocked"))
        try:
            return datetime.fromisoformat(locked_until) > datetime.now(timezone.utc)
        except ValueError:
            return True

    def _is_provider_locked(self, provider_state: dict[str, Any]) -> bool:
        locked_until = str(provider_state.get("lockedUntil") or "")
        if not locked_until:
            return bool(provider_state.get("softLocked"))
        try:
            return datetime.fromisoformat(locked_until) > datetime.now(timezone.utc)
        except ValueError:
            return True

    def _locked_provider_error(self, provider: str, provider_state: dict[str, Any]) -> RuntimeError:
        reason_text = (
            str(provider_state.get("softLockReason") or "")
            or str(provider_state.get("lastError") or "")
            or "provider is locked"
        )
        if str(provider_state.get("status") or "") == "auth_locked":
            return ApiProviderAuthLocked(f"{provider}: {reason_text}")
        return ApiQuotaExhausted(f"{provider}: {reason_text}")

    def _locked_key_error(self, provider: str, key_state: dict[str, Any], key_index: int) -> RuntimeError:
        reason_text = (
            str(key_state.get("softLockReason") or "")
            or str(key_state.get("lastError") or "")
            or f"key {key_index} is exhausted, locked, or unavailable"
        )
        if str(key_state.get("status") or "") == "auth_locked":
            return ApiProviderAuthLocked(f"{provider}: {reason_text}")
        return ApiQuotaExhausted(f"{provider}: {reason_text}")

    def _provider_usage_locked(self, provider_state: dict[str, Any]) -> tuple[int, int]:
        used_tokens = 0
        used_requests = 0
        for key_state in provider_state.get("keys", {}).values():
            used_tokens += int(key_state.get("tokensUsed", 0))
            used_requests += int(key_state.get("requestsUsed", 0))
        return used_tokens, used_requests

    def _soft_lock_provider_locked(self, provider_state: dict[str, Any], reason: str) -> None:
        provider_state["softLocked"] = True
        provider_state["status"] = "soft_cap"
        provider_state["softLockReason"] = reason

    def _minute_state_locked(self, provider_state: dict[str, Any]) -> dict[str, Any]:
        bucket = self._minute_bucket()
        minute_state = provider_state.setdefault("minuteWindow", {})
        if minute_state.get("bucket") != bucket:
            minute_state.clear()
            minute_state.update({"bucket": bucket, "tokensUsed": 0, "requestsUsed": 0})
            provider_state["rateLimitedUntil"] = ""
            provider_state["rateLimitReason"] = ""
        return minute_state

    def _reserve_minute_quota_locked(self, provider: str, provider_state: dict[str, Any], estimated_tokens: int) -> None:
        safe_tokens = max(1, int(self._minute_token_limit(provider) * self._minute_soft_cap_ratio(provider)))
        safe_requests = max(1, int(self._minute_request_limit(provider) * self._minute_soft_cap_ratio(provider)))
        minute_state = self._minute_state_locked(provider_state)
        next_tokens = int(minute_state.get("tokensUsed", 0)) + estimated_tokens
        next_requests = int(minute_state.get("requestsUsed", 0)) + 1
        if next_tokens >= safe_tokens or next_requests > safe_requests:
            retry_after = self._next_minute_iso()
            reason = "provider per-minute safe rate limit reached"
            provider_state["rateLimitedUntil"] = retry_after
            provider_state["rateLimitReason"] = reason
            self._write_state_locked()
            raise ApiRateLimited(f"{provider}: {reason}; retry after {retry_after}")
        minute_state["tokensUsed"] = next_tokens
        minute_state["requestsUsed"] = next_requests

    def _ensure_key_state(self, provider_state: dict[str, Any], key_hash: str, index: int) -> dict[str, Any]:
        key_state = provider_state.setdefault("keys", {}).setdefault(
            key_hash,
            {
                "keyIndex": index,
                "tokensUsed": 0,
                "requestsUsed": 0,
                "lockedUntil": "",
                "status": "healthy",
                "lastError": "",
            },
        )
        key_state["keyIndex"] = index
        return key_state

    def _rotation_start_index_locked(self, provider_state: dict[str, Any], key_count: int) -> int:
        if key_count <= 0:
            return 1
        start = int(provider_state.get("nextKeyStartIndex", 1) or 1)
        if start < 1 or start > key_count:
            start = 1
        return start

    def _rotated_key_order(self, start_index: int, key_count: int) -> list[int]:
        if key_count <= 0:
            return []
        return [((start_index - 1 + offset) % key_count) + 1 for offset in range(key_count)]

    def _key_roles(self, provider: str, key_count: int) -> list[str]:
        """Optional per-key role assignment (e.g. 3 keys from 3 separate NIM accounts: one
        dedicated to translation, one to vision_ocr/other steps, one held back as backup).
        {PROVIDER}_KEY_ROLES is a CSV aligned index-for-index with the provider's key CSV.
        Unset, empty, or length-mismatched (a misconfiguration -- safer to fail open than to
        silently misroute) all resolve to every key having role "*" (unrestricted, matches any
        capability at normal priority) -- this keeps every existing deployment's behavior
        completely unchanged unless KEY_ROLES is deliberately configured.
        """
        raw = os.environ.get(f"{provider.upper()}_KEY_ROLES", "").strip()
        if not raw:
            return ["*"] * key_count
        roles = [part.strip().lower() or "*" for part in raw.split(",")]
        if len(roles) != key_count:
            LOGGER.warning(
                "%s_KEY_ROLES has %d entries but %d keys are configured -- ignoring roles for this provider",
                provider.upper(),
                len(roles),
                key_count,
            )
            return ["*"] * key_count
        return roles

    def _role_priority(self, role: str, capability: str) -> int | None:
        """0 = this key is a normal-priority candidate for `capability`; 1 = this key is a
        backup, only tried once no priority-0 key is usable; None = this key is dedicated to a
        DIFFERENT specific capability and must be skipped entirely for this request (so a
        translation-only key, under a configured role split, is never touched by vision_ocr
        traffic and vice versa -- the isolation the role feature exists for).
        """
        if role in ("*", capability):
            return 0
        if role == "backup":
            return 1
        return None

    def _role_ordered_candidates(self, rotated_order: list[int], roles: list[str], capability: str) -> list[int]:
        priority_0 = [index for index in rotated_order if self._role_priority(roles[index - 1], capability) == 0]
        priority_1 = [index for index in rotated_order if self._role_priority(roles[index - 1], capability) == 1]
        return priority_0 + priority_1

    def _provider_ready_reason(self, provider: str) -> str:
        config = self.providers.get(provider)
        if not config:
            return "unsupported provider"
        missing = [name for name in config.required_env if not self._configured(os.environ.get(name))]
        if missing:
            return f"missing required env: {', '.join(missing)}"
        keys = self.provider_keys(provider)
        if not keys:
            return "no configured keys"
        if provider == "cloudflare":
            account_ids = self.cloudflare_account_ids()
            if not account_ids:
                return "missing required env: CLOUDFLARE_ACCOUNT_IDS or CLOUDFLARE_ACCOUNT_ID"
            if len(account_ids) not in {1, len(keys)}:
                return f"account id count ({len(account_ids)}) must be 1 or match key count ({len(keys)})"
        return ""

    def reserve_key(self, provider: str, estimated_tokens: int, capability: str = "translation") -> ApiKeyLease:
        reason = self._provider_ready_reason(provider)
        if reason:
            raise ApiProviderUnavailable(f"{provider}: {reason}")
        config = self.providers[provider]
        if capability not in config.capabilities:
            raise ApiProviderUnavailable(f"{provider}: capability {capability} is not enabled")
        keys = self.provider_keys(provider)
        safe_token_limit = max(1, int(self._token_limit(provider) * self._soft_cap_ratio(provider)))
        safe_request_limit = max(1, int(self._request_limit(provider) * self._soft_cap_ratio(provider)))
        limit_scope = self._limit_scope(provider)
        now = self._now_iso()
        with self._synced():
            provider_state = self._provider_state_locked(provider)
            if self._is_provider_locked(provider_state):
                self._write_state_locked()
                raise self._locked_provider_error(provider, provider_state)
            provider_tokens, provider_requests = self._provider_usage_locked(provider_state)
            if limit_scope == "provider":
                next_provider_tokens = provider_tokens + estimated_tokens
                next_provider_requests = provider_requests + 1
                if next_provider_tokens >= safe_token_limit or next_provider_requests >= safe_request_limit:
                    self._soft_lock_provider_locked(provider_state, "provider 80% safe quota reached")
                    self._write_state_locked()
                    raise ApiQuotaExhausted(f"{provider}: provider 80% safe quota reached")
            first_soft_lock_reason = ""
            locked_count = 0
            auth_locked_count = 0
            first_auth_lock_reason = ""
            key_count = len(keys)
            start_index = self._rotation_start_index_locked(provider_state, key_count)
            roles = self._key_roles(provider, key_count)
            # Priority-0 (role matches this capability, or unrestricted "*") candidates are tried
            # first in rotation order; only if none of those are usable do priority-1 ("backup")
            # candidates get tried. Keys dedicated to a DIFFERENT specific capability are excluded
            # entirely -- see _role_priority. With no KEY_ROLES configured, every key is "*" and
            # this candidate list is identical to the unfiltered rotation order (unchanged
            # behavior for every existing deployment).
            candidate_indices = self._role_ordered_candidates(self._rotated_key_order(start_index, key_count), roles, capability)
            for index in candidate_indices:
                key = keys[index - 1]
                key_hash = self._key_hash(key)
                key_state = self._ensure_key_state(provider_state, key_hash, index)
                if self._is_locked(key_state):
                    locked_count += 1
                    if str(key_state.get("status") or "") == "auth_locked":
                        auth_locked_count += 1
                        first_auth_lock_reason = first_auth_lock_reason or (
                            str(key_state.get("softLockReason") or "")
                            or str(key_state.get("lastError") or "")
                            or "provider access is locked"
                        )
                    continue
                next_tokens = int(key_state.get("tokensUsed", 0)) + estimated_tokens
                next_requests = int(key_state.get("requestsUsed", 0)) + 1
                if limit_scope == "key" and (next_tokens >= safe_token_limit or next_requests >= safe_request_limit):
                    key_state["softLocked"] = True
                    key_state["status"] = "soft_cap"
                    key_state["softLockReason"] = "80% safe quota reached"
                    first_soft_lock_reason = first_soft_lock_reason or key_state["softLockReason"]
                    continue
                self._reserve_minute_quota_locked(provider, provider_state, estimated_tokens)
                key_state["tokensUsed"] = next_tokens
                key_state["requestsUsed"] = next_requests
                key_state["status"] = "reserved"
                key_state["lastReservedAt"] = now
                provider_state["nextKeyStartIndex"] = (index % key_count) + 1
                self._write_state_locked()
                return ApiKeyLease(provider, index, key_hash, key, estimated_tokens, capability)
            self._write_state_locked()
        if not candidate_indices:
            raise ApiProviderUnavailable(f"{provider}: no key has a role compatible with capability {capability}")
        if locked_count >= len(candidate_indices) and auth_locked_count == locked_count:
            detail = first_auth_lock_reason or "all keys are access-blocked or auth-locked"
            raise ApiProviderAuthLocked(f"{provider}: {detail}")
        detail = first_soft_lock_reason or first_auth_lock_reason or "all keys are exhausted, locked, or unavailable"
        raise ApiQuotaExhausted(f"{provider}: {detail}")

    def reserve_key_index(
        self,
        provider: str,
        key_index: int,
        estimated_tokens: int,
        capability: str = "translation",
    ) -> ApiKeyLease:
        reason = self._provider_ready_reason(provider)
        if reason:
            raise ApiProviderUnavailable(f"{provider}: {reason}")
        config = self.providers[provider]
        if capability not in config.capabilities:
            raise ApiProviderUnavailable(f"{provider}: capability {capability} is not enabled")
        keys = self.provider_keys(provider)
        if key_index < 1 or key_index > len(keys):
            raise ApiProviderUnavailable(f"{provider}: key index {key_index} is not configured")
        safe_token_limit = max(1, int(self._token_limit(provider) * self._soft_cap_ratio(provider)))
        safe_request_limit = max(1, int(self._request_limit(provider) * self._soft_cap_ratio(provider)))
        limit_scope = self._limit_scope(provider)
        key = keys[key_index - 1]
        key_hash = self._key_hash(key)
        now = self._now_iso()
        with self._synced():
            provider_state = self._provider_state_locked(provider)
            if self._is_provider_locked(provider_state):
                self._write_state_locked()
                raise self._locked_provider_error(provider, provider_state)
            provider_tokens, provider_requests = self._provider_usage_locked(provider_state)
            if limit_scope == "provider":
                next_provider_tokens = provider_tokens + estimated_tokens
                next_provider_requests = provider_requests + 1
                if next_provider_tokens >= safe_token_limit or next_provider_requests >= safe_request_limit:
                    self._soft_lock_provider_locked(provider_state, "provider 80% safe quota reached")
                    self._write_state_locked()
                    raise ApiQuotaExhausted(f"{provider}: provider 80% safe quota reached")
            key_state = self._ensure_key_state(provider_state, key_hash, key_index)
            if self._is_locked(key_state):
                self._write_state_locked()
                raise self._locked_key_error(provider, key_state, key_index)
            next_tokens = int(key_state.get("tokensUsed", 0)) + estimated_tokens
            next_requests = int(key_state.get("requestsUsed", 0)) + 1
            if limit_scope == "key" and (next_tokens >= safe_token_limit or next_requests >= safe_request_limit):
                key_state["softLocked"] = True
                key_state["status"] = "soft_cap"
                key_state["softLockReason"] = "80% safe quota reached"
                self._write_state_locked()
                raise ApiQuotaExhausted(f"{provider}: key {key_index} reached 80% safe quota")
            self._reserve_minute_quota_locked(provider, provider_state, estimated_tokens)
            key_state["tokensUsed"] = next_tokens
            key_state["requestsUsed"] = next_requests
            key_state["status"] = "reserved"
            key_state["lastReservedAt"] = now
            self._write_state_locked()
        return ApiKeyLease(provider, key_index, key_hash, key, estimated_tokens, capability)

    def mark_success(self, lease: ApiKeyLease, response_payload: object | None = None) -> None:
        actual_tokens = self._extract_actual_tokens(response_payload)
        with self._synced():
            provider_state = self._provider_state_locked(lease.provider)
            key_state = self._ensure_key_state(provider_state, lease.key_hash, lease.key_index)
            if actual_tokens and actual_tokens > lease.reserved_tokens:
                key_state["tokensUsed"] = int(key_state.get("tokensUsed", 0)) + (actual_tokens - lease.reserved_tokens)
            if not self._is_locked(key_state):
                key_state["status"] = "healthy"
            key_state["lastSuccessAt"] = self._now_iso()
            key_state["lastError"] = ""
            self._write_state_locked()

    def mark_failure(self, lease: ApiKeyLease, status_code: int | None = None, error: object | None = None) -> None:
        error_text = str(error or "")
        terminal = self.is_terminal_quota_error(status_code, error_text)
        auth_block = status_code in {401, 403}
        lock_status = "healthy"
        lock_reason = ""
        if terminal:
            lock_status = "exhausted"
            if status_code == 402:
                lock_reason = "provider billing or credits unavailable (HTTP 402)"
            elif status_code == 429:
                lock_reason = "provider quota or rate limit reached (HTTP 429)"
            else:
                lock_reason = f"provider quota/rate/billing limit reached (HTTP {status_code or 'unknown'})"
        elif auth_block:
            lock_status = "auth_locked"
            if status_code == 401:
                lock_reason = "provider authentication failed (HTTP 401)"
            elif "access denied" in error_text.lower() or "network settings" in error_text.lower():
                lock_reason = "provider access denied or network-blocked (HTTP 403)"
            else:
                lock_reason = "provider authorization failed (HTTP 403)"
        with self._synced():
            provider_state = self._provider_state_locked(lease.provider)
            key_state = self._ensure_key_state(provider_state, lease.key_hash, lease.key_index)
            key_state["lastError"] = error_text[:600]
            key_state["lastFailureAt"] = self._now_iso()
            if terminal or auth_block:
                key_state["lockedUntil"] = self._locked_until_iso()
                key_state["status"] = lock_status
                key_state["softLocked"] = True
                key_state["softLockReason"] = lock_reason
                provider_wide = False
                if self._limit_scope(lease.provider) == "provider":
                    if terminal:
                        # A token/request budget under limit_scope="provider" is a
                        # genuinely shared pool -- one key hitting its cap means
                        # the pool itself is exhausted for every key, so locking
                        # the whole provider immediately is correct here.
                        provider_wide = True
                    else:
                        # auth_block (401/403) is a fact about ONE credential, not
                        # about the shared pool -- a single revoked/bad key must
                        # not lock siblings that are still valid. Only escalate to
                        # a provider-wide lock once every configured key for this
                        # provider is individually locked.
                        all_keys = self.provider_keys(lease.provider)
                        provider_wide = bool(all_keys) and all(
                            self._is_locked(self._ensure_key_state(provider_state, self._key_hash(k), idx))
                            for idx, k in enumerate(all_keys, start=1)
                        )
                if provider_wide:
                    provider_state["lockedUntil"] = key_state["lockedUntil"]
                    provider_state["status"] = lock_status
                    provider_state["softLocked"] = True
                    provider_state["softLockReason"] = lock_reason
                    provider_state["lastError"] = error_text[:600]
                LOGGER.critical(
                    "Provider key locked provider=%s key_index=%s status=%s reason=%s error=%s",
                    lease.provider,
                    lease.key_index,
                    status_code,
                    lock_reason,
                    error_text[:300],
                )
                write_diagnostic_event(
                    "api.provider_key_locked",
                    {
                        "provider": lease.provider,
                        "keyIndex": lease.key_index,
                        "httpStatus": status_code,
                        "status": lock_status,
                        "reason": lock_reason,
                        "error": error_text[:600],
                        "capability": lease.capability,
                    },
                    source="api_manager",
                    level="error" if auth_block else "warning",
                )
            else:
                key_state["status"] = "error"
            self._write_state_locked()

    def is_terminal_quota_error(self, status_code: int | None, payload: object = "") -> bool:
        text = str(payload).lower()
        if status_code in {402, 429}:
            return True
        return any(
            marker in text
            for marker in (
                "insufficient_quota",
                "insufficient quota",
                "quota exceeded",
                "rate limit",
                "too many requests",
                "payment required",
                "billing",
                "credits",
            )
        )

    def _extract_actual_tokens(self, payload: object | None) -> int | None:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for key in ("total_tokens", "totalTokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    return value
        metadata = payload.get("usageMetadata")
        if isinstance(metadata, dict):
            value = metadata.get("totalTokenCount")
            if isinstance(value, int):
                return value
        return None

    def provider_status(self, provider: str) -> dict[str, Any]:
        config = self.providers.get(provider)
        keys = self.provider_keys(provider)
        token_limit = self._token_limit(provider)
        request_limit = self._request_limit(provider)
        soft_cap_ratio = self._soft_cap_ratio(provider)
        safe_tokens = max(1, int(token_limit * soft_cap_ratio))
        safe_requests = max(1, int(request_limit * soft_cap_ratio))
        minute_soft_cap_ratio = self._minute_soft_cap_ratio(provider)
        safe_minute_tokens = max(1, int(self._minute_token_limit(provider) * minute_soft_cap_ratio))
        safe_minute_requests = max(1, int(self._minute_request_limit(provider) * minute_soft_cap_ratio))
        limit_scope = self._limit_scope(provider)
        reason = self._provider_ready_reason(provider)
        with self._synced():
            provider_state = self._provider_state_locked(provider)
            minute_state = self._minute_state_locked(provider_state)
            rate_limited_until = str(provider_state.get("rateLimitedUntil") or "")
            provider_locked = self._is_provider_locked(provider_state)
            provider_used_tokens, provider_used_requests = self._provider_usage_locked(provider_state)
            key_payload = []
            used_tokens = 0
            used_requests = 0
            locked = 0
            locked_reasons: list[str] = []
            locked_statuses: list[str] = []
            for index, key in enumerate(keys, start=1):
                key_hash = self._key_hash(key)
                key_state = self._ensure_key_state(provider_state, key_hash, index)
                used_tokens += int(key_state.get("tokensUsed", 0))
                used_requests += int(key_state.get("requestsUsed", 0))
                is_locked = provider_locked or self._is_locked(key_state)
                locked += 1 if is_locked else 0
                if is_locked:
                    locked_statuses.append(
                        str(provider_state.get("status") or "")
                        or str(key_state.get("status") or "")
                        or "locked"
                    )
                    locked_reasons.append(
                        (
                            str(provider_state.get("softLockReason") or "")
                            or str(provider_state.get("lastError") or "")
                            or str(key_state.get("softLockReason") or "")
                            or str(key_state.get("lastError") or "")
                            or str(key_state.get("status") or "")
                            or "key is temporarily locked"
                        )[:240]
                    )
                if limit_scope == "provider":
                    remaining_tokens = max(0, safe_tokens - provider_used_tokens)
                    remaining_requests = max(0, safe_requests - provider_used_requests)
                else:
                    remaining_tokens = max(0, safe_tokens - int(key_state.get("tokensUsed", 0)))
                    remaining_requests = max(0, safe_requests - int(key_state.get("requestsUsed", 0)))
                key_payload.append(
                    {
                        "index": index,
                        "fingerprint": key_hash[:8],
                        "status": "locked" if is_locked else key_state.get("status", "healthy"),
                        "remainingTokens": remaining_tokens,
                        "remainingRequests": remaining_requests,
                        "lockedUntil": key_state.get("lockedUntil", ""),
                        "lastError": key_state.get("lastError", ""),
                    }
                )
        if limit_scope == "provider":
            total_safe_tokens = safe_tokens
            total_safe_requests = safe_requests
        else:
            total_safe_tokens = safe_tokens * max(1, len(keys))
            total_safe_requests = safe_requests * max(1, len(keys))
        remaining_tokens = max(0, total_safe_tokens - used_tokens)
        remaining_requests = max(0, total_safe_requests - used_requests)
        remaining_percent = min(
            100.0,
            max(
                0.0,
                min(
                    (remaining_tokens / total_safe_tokens) * 100 if total_safe_tokens else 0,
                    (remaining_requests / total_safe_requests) * 100 if total_safe_requests else 0,
                ),
            ),
        )
        if not keys or reason:
            health = "unconfigured"
        elif locked >= len(keys) or remaining_percent <= 0:
            health = "auth_locked" if locked_statuses and all(status == "auth_locked" for status in locked_statuses) else "exhausted"
        elif rate_limited_until:
            health = "rate_limited"
        elif remaining_percent < 50:
            health = "warning"
        else:
            health = "healthy"
        display_reason = reason
        if health in {"exhausted", "auth_locked"} and not display_reason:
            display_reason = locked_reasons[0] if locked_reasons else "all configured keys are temporarily locked"
        return {
            "provider": provider,
            "configured": bool(keys) and not reason,
            "reason": display_reason,
            "keyCount": len(keys),
            "activeKeys": max(0, len(keys) - locked),
            "lockedKeys": locked,
            "remainingPercent": round(remaining_percent, 1),
            "remainingTokens": remaining_tokens,
            "safeTokenLimit": total_safe_tokens,
            "remainingRequests": remaining_requests,
            "safeRequestLimit": total_safe_requests,
            "softCapRatio": soft_cap_ratio,
            "limitScope": limit_scope,
            "minuteSoftCapRatio": minute_soft_cap_ratio,
            "remainingMinuteTokens": max(0, safe_minute_tokens - int(minute_state.get("tokensUsed", 0))),
            "safeMinuteTokenLimit": safe_minute_tokens,
            "remainingMinuteRequests": max(0, safe_minute_requests - int(minute_state.get("requestsUsed", 0))),
            "safeMinuteRequestLimit": safe_minute_requests,
            "rateLimitedUntil": rate_limited_until,
            "health": health,
            "capabilities": list(config.capabilities if config else ()),
            "keys": key_payload,
        }

    def quota_status(self, providers: list[str] | None = None) -> dict[str, Any]:
        provider_names = providers or list(self.providers)
        provider_statuses = [self.provider_status(provider) for provider in provider_names if provider in self.providers]
        configured = [item for item in provider_statuses if item["configured"]]
        exhausted = bool(configured) and all(item["health"] == "exhausted" for item in configured)
        auth_locked = bool(configured) and all(item["health"] == "auth_locked" for item in configured)
        return {
            "ok": True,
            "envFileLoaded": self._env_path.name if self._env_path else "",
            "softCapRatio": self.soft_cap_ratio,
            "minuteSoftCapRatio": os.environ.get("API_MANAGER_MINUTE_SOFT_CAP_RATIO", "0.90").strip() or "0.90",
            "defaultLimitScope": os.environ.get("API_MANAGER_DEFAULT_LIMIT_SCOPE", "provider").strip().lower() or "provider",
            "lockHours": self.lock_hours,
            "globalLimitReached": exhausted,
            "globalAuthLocked": auth_locked,
            "globalRateLimitReached": bool(configured) and all(item["health"] == "rate_limited" for item in configured),
            "message": DAILY_LIMIT_MESSAGE if exhausted else ("All configured providers are access-blocked. Check API keys, account access, and network restrictions." if auth_locked else ""),
            "providers": provider_statuses,
            "updatedAt": self._now_iso(),
        }

    def all_configured_providers_exhausted(self, providers: list[str]) -> bool:
        statuses = [self.provider_status(provider) for provider in providers if provider in self.providers]
        configured = [status for status in statuses if status["configured"]]
        # A provider stuck at auth_locked (revoked/bad credentials) is just as
        # unavailable to the caller as one that's quota-exhausted -- treat both
        # as "unavailable" so the local NLLB fallback still engages instead of
        # step 7 returning untranslated items when every provider is blocked.
        return bool(configured) and all(status["health"] in {"exhausted", "auth_locked"} for status in configured)

    def reset_state(self) -> None:
        with self._synced():
            self._state = {}
            self._write_state_locked()


API_MANAGER = ApiManager()
