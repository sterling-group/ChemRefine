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
