"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls an LLM to write a grounded answer from the retrieved context.
  TODO: fill in your provider of choice (Anthropic, OpenAI, a local model via
  Ollama, etc). A minimal Anthropic example is sketched below — install the
  `anthropic` package and set the ANTHROPIC_API_KEY environment variable to use it.
"""

import os
from typing import List, Tuple

from .ingest import Chunk


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Call a configured LLM to produce a grounded answer.

    This function attempts to use Anthropic if ANTHROPIC_API_KEY is set and the
    `anthropic` package is installed. If not available, it will try OpenAI when
    OPENAI_API_KEY is set and `openai` is installed. If no API keys are set, it
    falls back to the extractive answer so the app works out-of-the-box.

    The implementation uses dynamic imports and clear fallback messages so the
    starter app continues to run without the optional LLM packages.
    """
    if not retrieved:
        return "No relevant passages were found for that query."

    # Build grounding context from retrieved chunks
    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the sources below. Cite the source title(s) you used.\n\n"
        f"{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Try Anthropic first if configured
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic  # type: ignore
        except Exception:
            return (
                "[LLM mode not fully configured] ANTHROPIC_API_KEY is set but the 'anthropic' "
                "package is not installed. Install it or unset ANTHROPIC_API_KEY.\n\n"
                "Prompt to use:\n\n" + prompt
            )

        try:
            # Support multiple anthropic client names across versions
            Client = getattr(anthropic, "Anthropic", None) or getattr(anthropic, "Client", None)
            if Client is None:
                return (
                    "[LLM mode error] The installed 'anthropic' package does not expose a compatible client."
                )
            client = Client(api_key=anthropic_key)

            # Many anthropic SDKs provide a `completions.create` or a `messages.create` API.
            if hasattr(client, "completions"):
                resp = client.completions.create(model="claude-2", prompt=prompt, max_tokens=500)
                # Try common response shapes
                return getattr(resp, "completion", getattr(resp, "text", resp["completion"]))
            if hasattr(client, "messages"):
                resp = client.messages.create(model="claude-2", messages=[{"role": "user", "content": prompt}], max_tokens=500)
                return getattr(resp, "content", resp["completion"]) if resp else ""

            return "[LLM mode error] Unexpected Anthropic client shape."
        except Exception as e:  # pragma: no cover - runtime/network error
            return f"[LLM request failed] Anthropic request error: {e}\n\nPrompt:\n{prompt}"

    # Try OpenAI if configured
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai  # type: ignore
        except Exception:
            return (
                "[LLM mode not fully configured] OPENAI_API_KEY is set but the 'openai' "
                "package is not installed. Install it or unset OPENAI_API_KEY.\n\n"
                "Prompt to use:\n\n" + prompt
            )

        try:
            openai.api_key = openai_key
            # Use ChatCompletion if available, fall back to Completion
            if hasattr(openai, "ChatCompletion"):
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=500)
                return resp["choices"][0]["message"]["content"]
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=500)
            return resp["choices"][0]["text"]
        except Exception as e:  # pragma: no cover - runtime/network error
            return f"[LLM request failed] OpenAI request error: {e}\n\nPrompt:\n{prompt}"

    # No LLM configured — fall back to extractive answer
    return (
        "[LLM mode not configured] Set ANTHROPIC_API_KEY or OPENAI_API_KEY (and install the corresponding "
        "Python package) to enable grounded LLM answers. Falling back to extractive mode:\n\n"
        + extractive_answer(query, retrieved)
    )


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
