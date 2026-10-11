"""Model access for the strategist, with a hard per-request time limit.

One job: send messages, get back a JSON object matching a schema, inside a
deadline. Pokémon is on a clock (55 s a battle decision out of a 7-minute
bank), so ``timeout`` is not optional decoration — a provider that blows it
costs a turn.

Two implementations, chosen by whichever key is set:

- ``AnthropicProvider`` — the official ``anthropic`` SDK against
  ``claude-sonnet-5``, using adaptive thinking plus structured outputs
  (``output_config.format``). Effort is dialled per decision so a 10-second
  draft pick does not think like a 90-second Team Preview. Set
  ``VGC_MODEL=claude-opus-5`` for stronger payoff estimates at roughly
  2.5x the price; the game theory is deterministic either way.
- ``OpenAIProvider`` — Chat Completions with strict ``json_schema``, over
  the starter's existing ``httpx`` dependency.

The OpenAI client here deliberately repeats ``examples/llm/providers.py``
rather than importing it: ``examples`` is sample code and is not part of the
installed package (see ``pyproject.toml``), so importing it would make
``agent`` break when run from outside the repo.

Keys only ever go into the client that needs them — never into a log, a
``repr``, or an error message. Provider error bodies can echo parts of a
key, so only the HTTP status is ever reported.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from typing import Protocol

import httpx

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
MAX_TOKENS = 8_000  # a payoff matrix and a few labels; thinking is billed separately
# The effort value that means "answer without thinking". Battle turns use it:
# with thinking on, a turn took 13-40 s and timed out on 2 of 7 in the
# 2026-10-09 match. Thinking is billed and timed, never shown, so its only
# value is answer quality, and a timed-out answer has none.
NO_THINKING = "none"


class ProviderError(RuntimeError):
    """The request failed, timed out, or did not come back as a JSON object."""


class StrategistProvider(Protocol):
    model: str
    # The model's own reasoning for the most recent call, when the provider
    # can supply it, for the match log. PRIVATE deliberation: it goes to the
    # local audit trail only and must never reach a public reasoning_summary.
    last_reasoning: str | None

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float, effort: str = "medium"
    ) -> dict:
        """Return the model's answer as a dict matching ``schema``.

        ``timeout`` bounds this one request in seconds; ``effort`` is a hint
        about how hard to think (``"low"``, ``"medium"``, ``"high"``).
        Raises ``ProviderError`` on any failure, so the caller can fall back
        to a deterministic choice rather than lose the turn.
        """
        ...


class AnthropicProvider:
    """Claude via the official SDK, with structured outputs and adaptive thinking."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "the anthropic package is not installed. Run `pip install anthropic`, "
                "or set OPENAI_API_KEY to use the OpenAI provider instead."
            ) from exc
        # Checked here, not on the first turn: structured outputs arrived in
        # anthropic 1.x, and an 0.x install would otherwise raise TypeError
        # mid-match. Caught at startup it is one clear message; caught at
        # turn 3 it is a forfeited game.
        if not _supports_structured_outputs(anthropic):
            raise RuntimeError(
                f"anthropic {getattr(anthropic, '__version__', '?')} is too old for the "
                "strategist: it has no `output_config`, so structured outputs are unavailable. "
                "Run `pip install -U 'anthropic>=1.0'` with the SAME interpreter you play with "
                "(a venv's python, not a different one on PATH), or set OPENAI_API_KEY instead."
            )
        self.model = model
        self.last_reasoning: str | None = None
        # Some models reject disabled thinking (Sonnet 5.5, Opus 5.5, Fable).
        # Learned on the first 400 and remembered, so it costs one call.
        self._thinking_off_ok = True
        self._anthropic = anthropic
        # No api_key argument means the SDK resolves ANTHROPIC_API_KEY, then
        # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float, effort: str = "medium"
    ) -> dict:
        system = [message["content"] for message in messages if message.get("role") == "system"]
        turns = [message for message in messages if message.get("role") != "system"]
        self.last_reasoning = None
        started = time.monotonic()
        thinking_off = effort == NO_THINKING and self._thinking_off_ok
        try:
            try:
                response = self._create(system, turns, schema, timeout, effort, thinking_off)
            except self._anthropic.BadRequestError:
                if not thinking_off:
                    raise
                # This model needs thinking on: use the least of it, now and after.
                self._thinking_off_ok = False
                left = timeout - (time.monotonic() - started)
                response = self._create(system, turns, schema, left, effort, False)
        except self._anthropic.APIStatusError as exc:
            # The API's reason, so a rejected request is diagnosable from the
            # console (82908a24 showed only "HTTP 400"). No secrets in it.
            reason = " ".join(str(getattr(exc, "message", "") or "").split())[:240]
            raise ProviderError(f"Claude request failed (HTTP {exc.status_code}{': ' + reason if reason else ''})") from None
        except self._anthropic.APIError as exc:
            raise ProviderError(f"Claude request failed ({type(exc).__name__})") from None
        except TypeError as exc:
            # An SDK that does not accept one of these arguments. The
            # constructor screens for this, so reaching here means a shape
            # changed under us: report it as a provider failure so the turn
            # falls back instead of the match dying.
            raise ProviderError(f"Claude request was rejected by the SDK: {exc}") from None

        if response.stop_reason == "refusal":
            raise ProviderError("the model declined to answer")
        self.last_reasoning = _thinking_text(response) or None
        text = next((block.text for block in response.content if block.type == "text"), "")
        return _as_object(text, "Claude")

    def _create(self, system: list[str], turns: list[dict], schema: dict, timeout: float, effort: str, thinking_off: bool):
        if thinking_off:
            thinking = {"type": "disabled"}
        else:
            # "summarized" is what makes the reasoning auditable. It is
            # free: thinking happens and is billed identically under every
            # display setting, and the default ("omitted") just returns
            # empty thinking text. The raw chain of thought is never
            # exposed on any model, so this is the most we can log.
            thinking = {"type": "adaptive", "display": "summarized"}
        # The system prompt is the same on every call of an agent: cache it
        # (ignored by the API below its minimum cacheable length).
        text = "\n\n".join(system)
        system_arg = ([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
                      if text else self._anthropic.NOT_GIVEN)
        return self._client.with_options(timeout=timeout, max_retries=0).messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_arg,
            messages=turns,
            thinking=thinking,
            output_config={
                "effort": "low" if effort == NO_THINKING else effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )


class OpenAIProvider:
    """OpenAI Chat Completions with strict structured output, over httpx."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.last_reasoning = None  # Chat Completions exposes no reasoning text
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float, effort: str = "medium"
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        try:
            response = self._http.post("/chat/completions", json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise ProviderError(f"OpenAI request failed (HTTP {response.status_code})")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("OpenAI response had no message") from None
        if message.get("refusal"):
            raise ProviderError("the model declined to answer")
        return _as_object(message.get("content") or "", "OpenAI")


NO_CREDENTIALS = (
    "No model configured: the strategist will play deterministic heuristics only, with no "
    "payoff matrix, equilibrium, or opponent modelling. To enable them, set ANTHROPIC_API_KEY "
    "(Claude, default claude-sonnet-5) or OPENAI_API_KEY in the starter's .env. See .env.example."
)


def anthropic_credentials() -> bool:
    """Whether the Anthropic SDK can find a credential of any kind.

    Not just ``ANTHROPIC_API_KEY``: the SDK also resolves
    ``ANTHROPIC_AUTH_TOKEN``, an ``ant auth login`` profile on disk, and
    workload identity federation. A constructed client reports what it
    resolved through ``auth_headers``, which is empty only when it found
    nothing — so asking the client is right where checking one env var is
    wrong.
    """
    try:
        import anthropic
    except ImportError:
        return False
    try:
        return bool(dict(anthropic.Anthropic().auth_headers))
    except Exception:
        # A malformed profile or an SDK that constructs differently: treat
        # it as "no usable credential" rather than failing the match.
        return False


def provider_from_env(log=None) -> StrategistProvider | None:
    """Pick a provider from the environment, or ``None`` for no model.

    Anthropic wins when it has any credential, then ``OPENAI_API_KEY``.
    With neither the strategist still plays every phase from its
    deterministic heuristics, so a missing key is a downgrade, not a crash.

    A provider that cannot be built — an SDK too old for structured
    outputs, say — is reported through ``log`` and skipped rather than
    raised, so a broken preference falls through to the next option instead
    of forfeiting every match. ``VGC_MODEL`` overrides the model id for
    whichever provider is chosen.
    """
    report = log or (lambda _message: None)
    override = os.environ.get("VGC_MODEL")

    if anthropic_credentials():
        try:
            return AnthropicProvider(model=override or DEFAULT_ANTHROPIC_MODEL)
        except Exception as exc:
            report(f"[vgc] Claude is configured but unusable, skipping it: {exc}")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIProvider(
                os.environ["OPENAI_API_KEY"],
                override or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
            )
        except Exception as exc:
            report(f"[vgc] OpenAI is configured but unusable, skipping it: {exc}")

    return None


def _supports_structured_outputs(anthropic_module: object) -> bool:
    """Whether this SDK's ``messages.create`` accepts ``output_config``.

    Asked of the installed signature rather than the version string: the
    parameter is what the call actually needs, and a version comparison
    would have to track pre-releases and forks to say the same thing.
    """
    try:
        client = anthropic_module.Anthropic(api_key="probe-not-sent")
        parameters = inspect.signature(client.messages.create).parameters
    except Exception:
        return False
    return "output_config" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _thinking_text(response: object) -> str:
    """Join the response's thinking-block summaries, if any came back.

    Tolerant by design: a model or platform that returns no thinking blocks,
    or empty ones, must not break a turn over a logging nicety.
    """
    blocks = getattr(response, "content", None) or []
    parts = []
    for block in blocks:
        if getattr(block, "type", None) != "thinking":
            continue
        text = getattr(block, "thinking", None) or getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _as_object(text: str, vendor: str) -> dict:
    try:
        data = json.loads(text)
    except ValueError:
        raise ProviderError(f"the {vendor} reply was not valid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError(f"the {vendor} reply was not a JSON object")
    return data
