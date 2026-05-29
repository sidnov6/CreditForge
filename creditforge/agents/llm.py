"""Groq client + a generic tool-calling loop.

Free, fast, OpenAI-compatible function-calling. The loop lets an agent call the
real platform tools, feeds results back, and returns the final text plus the
trace (which tools ran) and any charts the tools produced.
"""
from __future__ import annotations

import json
import os

from creditforge.agents import tools
from creditforge.config import load_config


class LLMNotConfigured(RuntimeError):
    pass


def _client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and export GROQ_API_KEY=... (or add it as a Space secret).")
    from groq import Groq
    return Groq(api_key=key)


# Smaller model with a much larger free-tier daily token budget — used as an
# automatic fallback when the primary model hits its rate/limit.
FALLBACK_MODEL = "llama-3.1-8b-instant"


def _model() -> str:
    return os.environ.get("CF_AGENT_MODEL") or str(load_config().agents.model)


def complete(messages, tool_names=None, temperature=None, tool_choice="auto"):
    """One chat completion (optionally with tools). Falls back to a lighter model
    on rate-limit so the free tier stays usable."""
    cfg = load_config()
    kwargs = {"messages": messages,
              "temperature": temperature if temperature is not None
              else float(cfg.agents.temperature)}
    if tool_names:
        kwargs["tools"] = tools.schemas_for(tool_names)
        kwargs["tool_choice"] = tool_choice

    from groq import RateLimitError
    client = _client()
    model = _model()
    try:
        return client.chat.completions.create(model=model, **kwargs)
    except RateLimitError:
        if model == FALLBACK_MODEL:
            raise
        return client.chat.completions.create(model=FALLBACK_MODEL, **kwargs)


def run_tool_loop(system: str, user: str, tool_names: list[str],
                  max_iters: int | None = None) -> dict:
    """Run an agent: LLM <-> tools until it answers (or hits the iteration cap)."""
    cfg = load_config()
    max_iters = max_iters or int(cfg.agents.max_tool_iters)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    trace: list[dict] = []
    collected_charts: list[dict] = []

    for i in range(max_iters):
        # Force a tool call on the first turn so the agent always grounds its
        # answer in real numbers (weak models otherwise hallucinate "need data").
        choice = "required" if i == 0 else "auto"
        resp = complete(messages, tool_names=tool_names, tool_choice=choice)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"text": msg.content or "", "trace": trace, "charts": collected_charts}

        # Record the assistant turn that requested the tool calls
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):   # models sometimes emit "null"/lists
                args = {}
            result = tools.call_tool(name, args)
            chart = result.pop("_chart", None)
            if chart:
                collected_charts.append(chart)
            trace.append({"tool": name, "args": args,
                          "chart": chart["id"] if chart else None})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result.get("summary", result),
                                                   default=str)})

    # Ran out of iterations — ask for a final synthesis with no tools
    messages.append({"role": "user",
                     "content": "Summarize your findings now in a final answer."})
    final = complete(messages, tool_names=None).choices[0].message.content or ""
    return {"text": final, "trace": trace, "charts": collected_charts}
