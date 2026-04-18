"""Pluggable Gen AI backend.

``GenAIClient`` is a Protocol. Three implementations ship:

- ``AnthropicClient`` — Claude with Extended Thinking enabled.
- ``OpenAIClient``    — GPT-4o / GPT-4.1 / GPT-5-class models.
- ``EchoClient``      — deterministic fake used in tests and in the
  worked example (so the repo is runnable without an API key).

Swapping providers is a ``build_client(cfg)`` dispatch; nodes A–E never
import this module.

Cost tracking
-------------
Prices live in ``_PRICE_TABLE`` in **dollars per 1M tokens**, as
published by each vendor. Tune the numbers when vendors change them —
the computation itself is provider-agnostic.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from .config import EngineConfig
from .exceptions import ConfigurationError, GenAIBackendError
from .logging_setup import audit
from .models import InferenceRequest, InferenceResponse


# (input_usd_per_1m, output_usd_per_1m). ``thinking`` tokens are billed
# as output tokens by Anthropic; no separate lane required.
_PRICE_TABLE: dict[str, tuple[Decimal, Decimal]] = {
    "claude-3-5-sonnet-20241022": (Decimal("3.00"), Decimal("15.00")),
    "claude-3-opus-20240229":     (Decimal("15.00"), Decimal("75.00")),
    "claude-3-5-haiku-20241022":  (Decimal("0.80"), Decimal("4.00")),
    "gpt-4o":                     (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini":                (Decimal("0.15"), Decimal("0.60")),
    "echo-test":                  (Decimal("0"), Decimal("0")),
}


def compute_cost(
    model: str, input_tokens: int, output_tokens: int,
) -> Decimal:
    if model not in _PRICE_TABLE:
        raise GenAIBackendError("no price entry for model", model=model)
    in_price, out_price = _PRICE_TABLE[model]
    million = Decimal("1000000")
    cost = (
        Decimal(input_tokens) * in_price / million
        + Decimal(output_tokens) * out_price / million
    )
    return cost.quantize(Decimal("0.000001"))


class GenAIClient(Protocol):
    provider: str
    model: str

    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


# --------------------------------------------------------------------------- #
# Anthropic                                                                   #
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, api_key: str, model: str, thinking_budget: int) -> None:
        if not api_key:
            raise ConfigurationError("anthropic api_key is required")
        try:
            import anthropic                                  # noqa: F401
        except ImportError as exc:                            # pragma: no cover
            raise ConfigurationError(
                "anthropic sdk missing — `pip install anthropic`"
            ) from exc
        self.provider = "anthropic"
        self.model = model
        self._thinking_budget = thinking_budget
        self._api_key = api_key

    def _client(self):                                        # pragma: no cover
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        prompt = _render_prompt(request)
        try:                                                  # pragma: no cover
            resp = self._client().messages.create(
                model=self.model,
                max_tokens=8192,
                thinking={
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget,
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:                              # pragma: no cover
            raise GenAIBackendError(
                "anthropic call failed", cause=repr(exc)
            ) from exc
        answer, thinking = "", None                           # pragma: no cover
        for block in resp.content:                            # pragma: no cover
            if block.type == "text":
                answer += block.text
            elif block.type == "thinking":
                thinking = (thinking or "") + block.thinking
        usage = resp.usage                                    # pragma: no cover
        inp = getattr(usage, "input_tokens", 0)               # pragma: no cover
        out = getattr(usage, "output_tokens", 0)              # pragma: no cover
        thought = getattr(usage, "thinking_tokens", None)     # pragma: no cover
        return InferenceResponse(                             # pragma: no cover
            request_id=request.request_id,
            provider=self.provider, model=self.model,
            answer=answer, thinking=thinking,
            input_tokens=inp, output_tokens=out,
            thinking_tokens=thought,
            cost_usd=compute_cost(self.model, inp, out),
        )


# --------------------------------------------------------------------------- #
# OpenAI                                                                      #
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ConfigurationError("openai api_key is required")
        try:
            import openai                                     # noqa: F401
        except ImportError as exc:                            # pragma: no cover
            raise ConfigurationError(
                "openai sdk missing — `pip install openai`"
            ) from exc
        self.provider = "openai"
        self.model = model
        self._api_key = api_key

    def _client(self):                                        # pragma: no cover
        import openai
        return openai.OpenAI(api_key=self._api_key)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        prompt = _render_prompt(request)
        try:                                                  # pragma: no cover
            resp = self._client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:                              # pragma: no cover
            raise GenAIBackendError(
                "openai call failed", cause=repr(exc)
            ) from exc
        answer = resp.choices[0].message.content or ""        # pragma: no cover
        usage = resp.usage                                    # pragma: no cover
        inp = usage.prompt_tokens                             # pragma: no cover
        out = usage.completion_tokens                         # pragma: no cover
        return InferenceResponse(                             # pragma: no cover
            request_id=request.request_id,
            provider=self.provider, model=self.model,
            answer=answer, thinking=None,
            input_tokens=inp, output_tokens=out,
            thinking_tokens=None,
            cost_usd=compute_cost(self.model, inp, out),
        )


# --------------------------------------------------------------------------- #
# Echo (deterministic fake)                                                   #
# --------------------------------------------------------------------------- #
class EchoClient:
    """No network. Returns a summary of the request it received.

    Used in the example and in tests so the whole pipeline is runnable
    without a vendor key. Emits a fixed synthetic token count so the
    cost-tracking machinery is still exercised.
    """

    def __init__(self, model: str = "echo-test") -> None:
        self.provider = "echo"
        self.model = model

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        n_signals = len(request.context_payload.get("signals", []))
        answer = (
            f"[ECHO] Considering {n_signals} validated signals for "
            f"state {request.state_id}. Question: {request.question}"
        )
        inp = max(1, len(answer) // 4)
        out = max(1, len(answer) // 4)
        audit(
            "node_genai.echo",
            state_id=str(request.state_id),
            input_tokens=inp, output_tokens=out,
        )
        return InferenceResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            answer=answer,
            thinking=None,
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=None,
            cost_usd=compute_cost(self.model, inp, out),
        )


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def build_client(cfg: EngineConfig) -> GenAIClient:
    """One-line-swap entry point — the only place providers are named."""
    if cfg.genai_provider == "anthropic":
        if not cfg.genai_api_key:
            # Fallback to Echo so the sample stays runnable without a key.
            audit("node_genai.no_key.fallback_echo", provider="anthropic")
            return EchoClient()
        return AnthropicClient(
            api_key=cfg.genai_api_key,
            model=cfg.genai_model,
            thinking_budget=cfg.genai_thinking_budget_tokens,
        )
    if cfg.genai_provider == "openai":
        if not cfg.genai_api_key:
            audit("node_genai.no_key.fallback_echo", provider="openai")
            return EchoClient()
        return OpenAIClient(api_key=cfg.genai_api_key, model=cfg.genai_model)
    raise ConfigurationError(
        "unknown provider", provider=cfg.genai_provider
    )


def _render_prompt(req: InferenceRequest) -> str:
    import json
    return (
        "You are a forex macro analyst. Use ONLY the context below.\n"
        f"QUESTION: {req.question}\n\n"
        f"CONTEXT:\n{json.dumps(req.context_payload, default=str, indent=2)}"
    )
