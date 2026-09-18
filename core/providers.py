"""Model provider registry: local, hosted open-weight and OpenAI as one interface.

Ollama, Groq, Together, DeepInfra, Fireworks and OpenRouter all expose an
OpenAI-compatible ``/v1/chat/completions``, so one ``ChatOpenAI`` client drives
every route and only ``base_url`` changes. That is why a provider is data here
rather than a subclass.

Four fields carry the whole difference between routes: ``max_workers`` (real
backend concurrency), ``context_window`` (must be sent explicitly to Ollama,
which otherwise defaults to 2048 and silently truncates), ``max_tokens`` (never
unset for a local model, which may have no stop tokens and will run to the
context limit) and ``cost_per_mtok`` (``None`` means the route is priced in
seconds, not dollars).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SETTINGS_FILE = "provider.json"

# Stages that can be routed independently. Extraction and reports are bulk
# structured work; synthesis is the one call a user actually reads.
STAGES = ("extraction", "reports", "synthesis")


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    model: str
    base_url: Optional[str]          # None = OpenAI's own endpoint
    api_key_env: Optional[str]       # None = no key needed (local)
    context_window: int
    max_workers: int                 # concurrency this backend actually sustains
    max_tokens: int                  # output cap; never unset for local
    supports_json_mode: bool
    cost_per_mtok: Optional[Tuple[float, float]]   # (input, output) USD per 1M
    notes: str = ""

    @property
    def is_local(self) -> bool:
        return self.cost_per_mtok is None

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_env)

    def has_key(self) -> bool:
        return not self.needs_key or bool(os.getenv(self.api_key_env or ""))

    def cost_of(self, tokens_in: int, tokens_out: int) -> float:
        """USD for a call. Local routes cost nothing but wall-clock time."""
        if not self.cost_per_mtok:
            return 0.0
        rate_in, rate_out = self.cost_per_mtok
        return (tokens_in / 1e6) * rate_in + (tokens_out / 1e6) * rate_out


# Hosted open-weight routes are the default recommendation on a TLS-inspected
# network: they are plain HTTPS through the Python stack, which already trusts
# the corporate CA via truststore, whereas `ollama pull` (a Go binary) does not.
PROVIDERS: Dict[str, Provider] = {
    "groq": Provider(
        key="groq",
        label="Groq · Llama 3.1 8B",
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        context_window=131_072,
        max_workers=2,               # free tier rate-limits aggressively
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.05, 0.08),
        notes="Fastest hosted open-weight, but Cloudflare-fronted: measured "
              "2026-09-18 as HTTP 403 (Cloudflare 1010) behind Netskope TLS "
              "inspection on this network.",
    ),
    "together": Provider(
        key="together",
        label="Together · Llama 3.1 8B",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        context_window=131_072,
        max_workers=4,
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.18, 0.18),
        notes="Broadest open-weight catalogue. Also measured HTTP 403 behind "
              "TLS inspection on this network.",
    ),
    "deepinfra": Provider(
        key="deepinfra",
        label="DeepInfra · Llama 3.1 8B",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        context_window=131_072,
        max_workers=4,
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.03, 0.05),
        notes="Cheapest hosted route, and reachable through TLS inspection "
              "(measured HTTP 401, i.e. the request reaches the API).",
    ),
    "fireworks": Provider(
        key="fireworks",
        label="Fireworks · Llama 3.1 8B",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        context_window=131_072,
        max_workers=4,
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.20, 0.20),
        notes="Strong structured output; reachable through TLS inspection.",
    ),
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter · Llama 3.1 8B",
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=131_072,
        max_workers=3,
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.02, 0.05),
        notes="One key across many providers; reachable through TLS inspection.",
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI · gpt-4o-mini",
        model="gpt-4o-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        context_window=128_000,
        max_workers=4,
        max_tokens=2_000,
        supports_json_mode=True,
        cost_per_mtok=(0.15, 0.60),
        notes="Highest quality of the configured routes.",
    ),
    "local": Provider(
        key="local",
        label="Local · Qwen2.5 3B",
        model="qwen2.5-3b-graphrag",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        context_window=8_192,
        max_workers=1,               # Ollama serialises; parallel slots multiply KV cache
        max_tokens=1_500,
        supports_json_mode=True,
        cost_per_mtok=None,
        notes="Free and offline. Measured weak at relationship extraction — "
              "prefer it for synthesis, not extraction.",
    ),
}

# DeepInfra rather than Groq: both are cheap open-weight routes, but Groq and
# Together return HTTP 403 through this network's TLS inspection, while
# DeepInfra, Fireworks and OpenRouter reach their APIs normally.
DEFAULT_ROUTING = {"extraction": "deepinfra", "reports": "deepinfra", "synthesis": "openai"}

PRESETS: Dict[str, Dict[str, str]] = {
    "frugal":   {"extraction": "local", "reports": "local", "synthesis": "local"},
    "balanced": {"extraction": "deepinfra", "reports": "deepinfra", "synthesis": "openai"},
    "best":     {"extraction": "openai", "reports": "openai", "synthesis": "openai"},
}


def get(key: str) -> Provider:
    if key not in PROVIDERS:
        raise KeyError(f"unknown provider {key!r}; known: {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def available() -> List[Provider]:
    """Every provider, in display order: hosted first, then OpenAI, then local."""
    order = ["deepinfra", "fireworks", "openrouter", "groq", "together", "openai", "local"]
    return [PROVIDERS[k] for k in order if k in PROVIDERS]


# -----------------------------------------------------------------------------
# Client construction
# -----------------------------------------------------------------------------
def build_llm(
    provider: Provider,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    streaming: bool = False,
):
    """The only place a chat client is constructed.

    ``max_tokens`` is always sent: a local model with no stop tokens will
    otherwise generate until the context window fills. ``num_ctx`` is sent to
    Ollama because its default is 2048 regardless of what the model supports,
    which silently truncates the tail of the prompt.
    """
    from langchain_openai import ChatOpenAI

    kwargs: Dict[str, Any] = {
        "model": provider.model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens or provider.max_tokens,
        "timeout": 240 if provider.is_local else 90,
        "max_retries": 5,            # covers hosted rate limits and local hiccups
        "streaming": streaming,
    }
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    kwargs["api_key"] = os.getenv(provider.api_key_env) if provider.api_key_env else "local"

    if json_mode and provider.supports_json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if provider.is_local:
        # Ollama takes num_ctx as a non-standard body field, so it has to ride in
        # extra_body — the typed SDK path rejects unknown top-level kwargs. This
        # matters: Ollama's default num_ctx is 2048 whatever the model supports,
        # and it truncates the tail of the prompt silently rather than erroring.
        kwargs["extra_body"] = {"options": {"num_ctx": provider.context_window}}

    return ChatOpenAI(**kwargs)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
_REACH_CACHE: Dict[str, Tuple[float, bool, str]] = {}
_REACH_TTL = 300.0


def _reachable(provider: Provider, timeout: float) -> Tuple[bool, str]:
    """Can we actually get to this API from this network?

    A key check alone is not enough. Measured 2026-09-18: with a valid key,
    Groq and Together return HTTP 403 (Cloudflare error 1010) because this
    network's TLS-inspecting proxy re-originates the connection and the CDN
    rejects its fingerprint. Without this probe the route reports healthy and
    then fails on every chunk of the run.
    """
    import urllib.error
    import urllib.request

    cached = _REACH_CACHE.get(provider.key)
    if cached and (time.time() - cached[0]) < _REACH_TTL:
        return cached[1], cached[2]

    url = f"{provider.base_url}/models" if provider.base_url else "https://api.openai.com/v1/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {os.getenv(provider.api_key_env or '') or 'probe'}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            result = (True, "ready")
    except urllib.error.HTTPError as exc:
        # 401/404 still prove the request reached the API; 403 is the CDN block.
        if exc.code == 403:
            result = (False, "blocked by network (HTTP 403) — try DeepInfra or OpenRouter")
        elif exc.code in (401, 403):
            result = (False, "key rejected")
        else:
            result = (True, "ready")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result = (False, f"unreachable ({type(exc).__name__})")

    _REACH_CACHE[provider.key] = (time.time(), result[0], result[1])
    return result


def health(provider: Provider, timeout: float = 6.0, probe: bool = True) -> Tuple[bool, str]:
    """Is this route usable right now?

    Checked before a run starts, never per chunk: a dead endpoint otherwise
    fails once per chunk, which on a 195-chunk build is 195 slow timeouts.
    """
    if provider.needs_key and not provider.has_key():
        return False, f"{provider.api_key_env} not set"

    if provider.is_local:
        import urllib.error
        import urllib.request
        root = (provider.base_url or "").rsplit("/v1", 1)[0]
        try:
            with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as response:
                names = [m.get("name", "") for m in json.load(response).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError):
            return False, "Ollama not running — start it with `ollama serve`"
        if not any(n.split(":")[0] == provider.model.split(":")[0] for n in names):
            return False, f"model {provider.model} not installed"
        return True, "ready"

    if not probe:
        return True, "key set"
    return _reachable(provider, timeout)


def healthy_routing(routing: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Fall back to any working route for stages whose provider is unusable."""
    warnings: List[str] = []
    resolved = dict(routing)
    working = [p.key for p in available() if health(p)[0]]
    for stage, key in routing.items():
        ok, reason = health(get(key)) if key in PROVIDERS else (False, "unknown provider")
        if ok:
            continue
        if not working:
            warnings.append(f"{stage}: {key} unavailable ({reason}) and no route is configured")
            continue
        resolved[stage] = working[0]
        warnings.append(f"{stage}: {key} unavailable ({reason}) — using {working[0]}")
    return resolved, warnings


# -----------------------------------------------------------------------------
# Persistence — config must survive a restart, so it does not live in session state
# -----------------------------------------------------------------------------
def load_settings(vault_dir: str = "./vault") -> Dict[str, Any]:
    path = Path(vault_dir) / SETTINGS_FILE
    data: Dict[str, Any] = {"routing": dict(DEFAULT_ROUTING), "preset": "balanced"}
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                routing = stored.get("routing")
                if isinstance(routing, dict):
                    data["routing"].update(
                        {k: v for k, v in routing.items() if k in STAGES and v in PROVIDERS}
                    )
                if stored.get("preset") in PRESETS or stored.get("preset") == "custom":
                    data["preset"] = stored["preset"]
        except (ValueError, OSError):
            pass
    return data


def save_settings(settings: Dict[str, Any], vault_dir: str = "./vault") -> None:
    path = Path(vault_dir) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "routing": {k: v for k, v in settings.get("routing", {}).items() if k in STAGES},
        "preset": settings.get("preset", "custom"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def provider_for(stage: str, settings: Optional[Dict[str, Any]] = None,
                 vault_dir: str = "./vault") -> Provider:
    settings = settings or load_settings(vault_dir)
    return get(settings["routing"].get(stage, DEFAULT_ROUTING[stage]))
