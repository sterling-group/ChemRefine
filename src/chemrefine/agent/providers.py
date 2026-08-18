"""Which model the embedded agent talks to — flags beat environment beats preset.

Multi-provider by construction: a preset names an OpenAI-compatible endpoint shape
(``ollama`` / ``vllm`` serve on localhost with a dummy key; ``openai`` uses the
official API via ``OPENAI_API_KEY``), the ``CHEMREFINE_LLM_*`` environment variables
override a preset's pieces, and explicit flags override both. Anything with an
OpenAI-compatible ``/v1`` — OpenRouter, Groq, a lab's own vLLM box — is the ``custom``
preset plus a base URL. A ``provider:model`` string with no base URL is handed to
PydanticAI verbatim, which resolves its native providers (``anthropic:…``,
``google-gla:…``) from their own environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from pydantic_ai.models import Model

_PRESETS: dict[str, tuple[str | None, str | None, str | None]] = {
    # Each preset is (base URL, API key, default model) — None means "not preset".
    "openai": (None, None, None),
    "ollama": ("http://localhost:11434/v1", "ollama", None),
    "vllm": ("http://localhost:8000/v1", "vllm", None),
    "custom": (None, None, None),
}


@dataclass(frozen=True)
class ProviderConfig:
    """The resolved answer: a model name, and (for compatible endpoints) where it lives."""

    model: str
    base_url: str | None
    api_key: str | None

    @classmethod
    def resolve(
        cls,
        provider: str = "custom",
        model: str | None = None,
        base_url: str | None = None,
    ) -> ProviderConfig:
        """Flags → ``CHEMREFINE_LLM_MODEL`` / ``_BASE_URL`` / ``_API_KEY`` → preset.

        A missing model is a :class:`ConfigError` naming all three ways to supply one —
        there is no defensible default across this many providers.
        """
        if provider not in _PRESETS:
            raise ConfigError(f"unknown provider {provider!r}; one of {sorted(_PRESETS)}")
        preset_url, preset_key, preset_model = _PRESETS[provider]
        resolved_model = model or os.environ.get("CHEMREFINE_LLM_MODEL") or preset_model
        if not resolved_model:
            raise ConfigError(
                "no model configured: pass --model, set CHEMREFINE_LLM_MODEL, or use a "
                "provider:model string (e.g. --model openai:gpt-5-mini, "
                "--provider ollama --model qwen3)"
            )
        return cls(
            model=resolved_model,
            base_url=base_url or os.environ.get("CHEMREFINE_LLM_BASE_URL") or preset_url,
            api_key=os.environ.get("CHEMREFINE_LLM_API_KEY") or preset_key,
        )

    def build_model(self) -> Model | str:
        """A PydanticAI model: an explicit OpenAI-compatible endpoint, or the name itself.

        With a ``base_url`` the model is pinned to that endpoint (Ollama/vLLM/OpenRouter
        speak the OpenAI chat API); without one, the plain string lets PydanticAI infer
        the provider from a ``provider:model`` spelling.
        """
        if self.base_url is None:
            return self.model
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            self.model,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key or "unset"),
        )


@dataclass(frozen=True)
class CheckReport:
    """What ``chemrefine agent --check`` found: usable or not, and the fixes by name."""

    ok: bool
    findings: tuple[str, ...]


def _fixes(base_url: str, problem: str) -> tuple[str, ...]:
    """The actionable next step(s) for one problem class at one endpoint.

    Ollama is recognized by its conventional port so its one-command fixes can be named
    verbatim; every other endpoint gets the generic remedy. Deliberately *suggestions*,
    never actions — verification must not download models or start services on the
    user's behalf (model choice is a site-policy decision; see the docs disclaimer).
    """
    ollama = ":11434" in base_url
    if problem == "unreachable":
        return ("start it with: ollama serve",) if ollama else ("is the endpoint URL right?",)
    if problem == "missing-model":
        return (
            ("pull it with: ollama pull <model>",)
            if ollama
            else ("if this endpoint does not enumerate models, retry without --check",)
        )
    return ("set CHEMREFINE_LLM_API_KEY",)


def check(config: ProviderConfig, *, timeout: float = 5.0) -> CheckReport:
    """Probe the resolved provider without starting a chat: reachable, and model listed?

    Verification, not provisioning — one ``GET {base_url}/models`` (the OpenAI-compatible
    listing Ollama, vLLM, Groq and friends all serve), never a download. A
    provider-native ``provider:model`` string (no ``base_url``) cannot be probed
    generically; it reports usable with a note, and PydanticAI checks credentials on the
    first real request. Works without the ``[agent]`` extra installed — this module
    imports no SDK at runtime — so the preflight can run before anything else is set up.
    """
    if config.base_url is None:
        return CheckReport(
            ok=True,
            findings=(
                f"model {config.model!r} is provider-native; not probed — credentials "
                "come from that provider's own environment on first use",
            ),
        )
    if not config.base_url.startswith(("http://", "https://")):
        return CheckReport(ok=False, findings=(f"base URL {config.base_url!r} is not HTTP(S)",))
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = config.base_url.rstrip("/") + "/models"
    request = Request(  # noqa: S310 — scheme constrained to http(s) above
        url, headers={"Authorization": f"Bearer {config.api_key or 'unset'}"}
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — scheme checked above
            listing = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        if e.code in (401, 403):
            return CheckReport(
                ok=False,
                findings=(f"{url}: authentication rejected ({e.code})", *_fixes(url, "auth")),
            )
        return CheckReport(ok=False, findings=(f"{url}: HTTP {e.code}",))
    except (URLError, OSError, TimeoutError) as e:
        return CheckReport(
            ok=False, findings=(f"{url}: unreachable ({e})", *_fixes(url, "unreachable"))
        )
    except ValueError:  # JSONDecodeError, UnicodeDecodeError — an HTML page, a binary body
        listing = None
    # A reachable endpoint answering something other than ``{"data": [{"id": ...}]}`` is a
    # misconfiguration, and turning misconfiguration into a finding is this function's whole
    # contract — a base URL pointing at a web app or a proxy's login page answers 200 with
    # HTML, and some gateways answer a bare list. Every one of those used to raise out of
    # here, so the preflight crashed on the mistake it exists to diagnose.
    entries = listing.get("data") if isinstance(listing, dict) else None
    served = (
        [e["id"] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]
        if isinstance(entries, list)
        else []
    )
    # `{"data": []}` is a listing: an endpoint that serves nothing yet. Entries that yield
    # no id at all are a different answer — the shape is wrong, not the inventory — and
    # saying "model not served" there would send the reader looking for the wrong fix.
    if not isinstance(entries, list) or (entries and not served):
        return CheckReport(
            ok=False,
            findings=(
                f"{url}: reachable, but the reply is not an OpenAI-style model listing",
                *_fixes(url, "missing-model"),
            ),
        )
    if config.model in served:
        return CheckReport(
            ok=True, findings=(f"{url}: reachable; model {config.model!r} is served",)
        )
    shown = ", ".join(sorted(served)[:8]) or "none listed"
    return CheckReport(
        ok=False,
        findings=(
            f"{url}: reachable, but model {config.model!r} is not served (has: {shown})",
            *_fixes(url, "missing-model"),
        ),
    )
