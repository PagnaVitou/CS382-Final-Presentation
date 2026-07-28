"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality.
- "llm": calls Claude (Anthropic) to write a grounded answer from the retrieved context.
  Set the ANTHROPIC_API_KEY environment variable to use this mode.
"""

import os
from typing import List, Tuple

from .ingest import Chunk


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Combine retrieved chunks into a simple answer without LLM processing."""
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, similarity score={score:.2f}]")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Use Claude (via Anthropic API) to generate a grounded answer from retrieved chunks.
    Falls back to extractive mode if API key is not set or call fails.
    """
    if not retrieved:
        return "No relevant passages were found for that query."

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured]\n\n"
            "To enable Claude-powered answers, install the anthropic package and set ANTHROPIC_API_KEY:\n\n"
            "  pip install anthropic\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context_parts = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        context_parts.append(f"Source {i} ({chunk.doc_title}, similarity: {score:.2f}):\n{chunk.text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = (
        "You are a helpful AI assistant. Answer the question using ONLY the provided sources. "
        "Always cite which source(s) you used by referencing the source title or number. "
        "If you cannot find a good answer in the sources, say so clearly.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)

        return "".join(text_parts) if text_parts else ""
    except ImportError:
        return (
            "[Anthropic SDK not installed]\n\n"
            "Install it with: pip install anthropic\n\n"
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )
    except Exception as e:
        return (
            f"[LLM error: {str(e)}]\n\n"
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    """
    Generate an answer to the query using retrieved chunks.
    
    Args:
        query: The user's question
        retrieved: List of (Chunk, similarity_score) tuples from retrieval
        mode: "extractive" or "llm"
    
    Returns:
        Generated answer string with citations
    """
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
