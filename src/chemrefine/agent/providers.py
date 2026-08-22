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

_OPENAI_URL = "https://api.openai.com/v1"
"""Where OpenAI-the-service lives, for the one thing that needs to know: the preflight.

Deliberately *not* in :data:`_PRESETS`. Putting it there would make every ``openai`` launch
take the pinned-endpoint branch, so ``--provider openai --model openai:gpt-5-mini`` would
hand that whole string to ``OpenAIChatModel`` as a model id. The preset stays empty and the
URL is used only where it is safe to know it."""

_PRESETS: dict[str, tuple[str | None, str | None, str | None]] = {
    # Each preset is (base URL, API key, default model) — None means "not preset".
    "openai": (None, None, None),
    "ollama": ("http://localhost:11434/v1", "ollama", None),
    "vllm": ("http://localhost:8000/v1", "vllm", None),
    "custom": (None, None, None),
}


def preset_shapes() -> dict[str, dict[str, object]]:
    """Per provider: where it lives when nobody says otherwise, and whether it wants a key.

    What a front end needs to decide which fields to show, derived here rather than
    copied there. ``default_url`` is the *effective* default, which is why ``openai`` is
    not simply its (empty) preset row: ``OpenAIProvider`` resolves api.openai.com on its
    own, so openai needs no URL from the user while ``custom`` — the only entry that
    genuinely has nowhere to go — needs one.

    A caller must treat ``default_url`` as a placeholder, never as a value to send back:
    :meth:`ProviderConfig.resolve` lets an explicit URL beat ``CHEMREFINE_LLM_BASE_URL``,
    so echoing it would override an environment the user set deliberately.
    """
    return {
        name: {
            "default_url": _OPENAI_URL if name == "openai" else url,
            "needs_key": key is None,
        }
        for name, (url, key, _model) in _PRESETS.items()
    }


@dataclass(frozen=True)
class ProviderConfig:
    """The resolved answer: a model name, and (for compatible endpoints) where it lives."""

    model: str
    base_url: str | None
    api_key: str | None
    provider: str = "custom"
    """Which preset resolved this, so :meth:`build_model` can tell OpenAI-the-service from
    an OpenAI-*compatible* box. Defaulted, because three call sites construct this by
    keyword and none of them cared until the GUI grew a key field."""

    @classmethod
    def resolve(
        cls,
        provider: str = "custom",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ProviderConfig:
        """Flags → ``CHEMREFINE_LLM_MODEL`` / ``_BASE_URL`` / ``_API_KEY`` → preset.

        A missing model is a :class:`ConfigError` naming all three ways to supply one —
        there is no defensible default across this many providers. A base URL that is
        not HTTP(S) is refused here, at the one place every caller resolves through —
        the GUI's chat endpoint takes a request-supplied base URL, and the resolved
        config pairs it with ``CHEMREFINE_LLM_API_KEY``, so an unchecked scheme would
        hand the bearer to whatever ``urlopen``/the SDK makes of it.

        ``api_key`` is the same story one tier up: the GUI's panel holds a key in the
        page for the session and sends it per request, so it takes precedence over the
        environment exactly as ``--model`` and ``--base-url`` do. Blank falls through, so
        an empty box still means "use ``CHEMREFINE_LLM_API_KEY``".
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
        resolved_url = base_url or os.environ.get("CHEMREFINE_LLM_BASE_URL") or preset_url
        if resolved_url is not None and not resolved_url.startswith(("http://", "https://")):
            raise ConfigError(f"base URL {resolved_url!r} is not HTTP(S)")
        return cls(
            model=resolved_model,
            base_url=resolved_url,
            api_key=api_key or os.environ.get("CHEMREFINE_LLM_API_KEY") or preset_key,
            provider=provider,
        )

    @property
    def keyed_openai(self) -> bool:
        """Whether this resolves to OpenAI-the-service through a key we were handed.

        The one predicate behind both :attr:`probe_url` and :meth:`build_model`, so the
        preflight can never validate a key the chat then declines to use. A
        ``provider:model`` spelling is excluded on purpose: that spelling is the
        instruction to let PydanticAI resolve the provider, credentials included.
        """
        return (
            self.base_url is None
            and self.provider == "openai"
            and bool(self.api_key)
            and ":" not in self.model
        )

    @property
    def probe_url(self) -> str | None:
        """Where :func:`check` should look, or ``None`` when there is nothing to probe.

        Usually the explicit ``base_url``. The exception is :attr:`keyed_openai`: it has
        no ``base_url`` because ``OpenAIProvider`` supplies its own, but we know where it
        lives, and probing it is the only way a preflight can tell a good key from a typo
        — which is the whole reason the panel has a key box.
        """
        if self.base_url is not None:
            return self.base_url
        return _OPENAI_URL if self.keyed_openai else None

    def build_model(self) -> Model | str:
        """A PydanticAI model: an explicit OpenAI-compatible endpoint, or the name itself.

        With a ``base_url`` the model is pinned to that endpoint (Ollama/vLLM/OpenRouter
        speak the OpenAI chat API); without one, the plain string lets PydanticAI infer
        the provider from a ``provider:model`` spelling.

        Between those sits OpenAI-the-service with a key the caller supplied. It has no
        ``base_url`` — ``OpenAIProvider`` defaults to ``api.openai.com`` on its own — so
        without this branch the key was silently dropped and PydanticAI read the server's
        ``OPENAI_API_KEY`` instead, which is not a key the GUI's user can set. A
        ``provider:model`` spelling still goes through verbatim: that spelling *is* the
        instruction to let PydanticAI resolve it, and handing ``openai:gpt-5-mini`` to
        ``OpenAIChatModel`` would send that literal string as a model id.
        """
        if self.base_url is None and not self.keyed_openai:
            return self.model
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = (
            OpenAIProvider(api_key=self.api_key)
            if self.keyed_openai
            else OpenAIProvider(base_url=self.base_url, api_key=self.api_key or "unset")
        )
        return OpenAIChatModel(self.model, provider=provider)


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
    provider-native ``provider:model`` string with nothing to aim at cannot be probed
    generically; it reports usable with a note, and PydanticAI checks credentials on the
    first real request. Works without the ``[agent]`` extra installed — this module
    imports no SDK at runtime — so the preflight can run before anything else is set up.

    What counts as "somewhere to aim at" is :attr:`ProviderConfig.probe_url`, not
    ``base_url``: OpenAI-the-service with a supplied key has no ``base_url`` and is still
    probeable, and validating that key is the most useful thing this can do for it.
    """
    probe = config.probe_url
    if probe is None:
        return CheckReport(
            ok=True,
            findings=(
                f"model {config.model!r} is provider-native; not probed — credentials "
                "come from that provider's own environment on first use",
            ),
        )
    if not probe.startswith(("http://", "https://")):
        return CheckReport(ok=False, findings=(f"base URL {probe!r} is not HTTP(S)",))
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = probe.rstrip("/") + "/models"
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
