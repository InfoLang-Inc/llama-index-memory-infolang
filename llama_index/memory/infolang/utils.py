from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole

DEFAULT_INTRO = (
    "Below is context recalled from InfoLang semantic memory that may be "
    "relevant to the current conversation:"
)
DEFAULT_OUTRO = "This is the end of the recalled context."

# Chunk scores below this floor are treated as weak matches. Mirrors the
# confidence floor used by the InfoLang SDK's RecallResult.weak property.
WEAK_MATCH_FLOOR = 0.85


def convert_chunks_to_system_message(
    chunks: List[object],
    existing_system_message: Optional[ChatMessage] = None,
) -> ChatMessage:
    """
    Render recalled InfoLang chunks into a single system message.

    ``chunks`` are ``infolang.types.Chunk`` instances (duck-typed here via
    ``text``/``score`` attributes so this module has no hard dependency on
    the SDK types). Chunks scoring below :data:`WEAK_MATCH_FLOOR` are still
    included but annotated as weak so the LLM can weigh them accordingly.
    """
    formatted = "\n\n" + DEFAULT_INTRO + "\n"
    for chunk in chunks:
        formatted += f"\n{format_chunk(chunk)}\n"
    formatted += "\n" + DEFAULT_OUTRO

    content = formatted
    if existing_system_message is not None and existing_system_message.content:
        prefix = existing_system_message.content.split(DEFAULT_INTRO)[0]
        content = prefix + formatted

    return ChatMessage(content=content, role=MessageRole.SYSTEM)


def format_chunk(chunk: object) -> str:
    text = getattr(chunk, "text", None) or ""
    score = getattr(chunk, "score", None)
    if score is not None and score < WEAK_MATCH_FLOOR:
        return f"[weak match, score={score:.2f}] {text}"
    return text


def convert_messages_to_query(
    messages: List[ChatMessage], input: Optional[str] = None, limit: int = 5
) -> Optional[str]:
    """
    Build a recall query from recent chat history and/or explicit input.

    Returns ``None`` when there is nothing to search on (empty history and
    no explicit ``input``), so callers can skip the recall round-trip.
    """
    if input:
        return input

    recent = [m for m in messages[-limit:] if m.content]
    if not recent:
        return None
    return "\n".join(f"{m.role.value}: {m.content}" for m in recent)


def is_memorable(message: ChatMessage) -> bool:
    """Whether a message should be written back to InfoLang memory."""
    return message.role in (MessageRole.USER, MessageRole.ASSISTANT) and bool(
        message.content
    )
