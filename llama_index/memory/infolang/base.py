from typing import Any, List, Optional

from infolang import InfoLang

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    model_serializer,
)
from llama_index.core.memory import BaseMemory
from llama_index.core.memory import Memory as LlamaIndexMemory
from llama_index.memory.infolang.utils import (
    convert_chunks_to_system_message,
    convert_messages_to_query,
    is_memorable,
)

DEFAULT_TOP_K = 5
DEFAULT_SOURCE = "llama_index"


class InfoLangMemory(BaseMemory):
    """
    Chat memory backed by InfoLang semantic memory.

    Local chat history is kept in an in-process ``llama_index.core.memory.Memory``
    buffer (``primary_memory``); every user/assistant message is additionally
    written to InfoLang via ``remember`` so it is retrievable across sessions.
    ``get`` augments the local chat history with a system message built from
    an InfoLang ``recall`` over the current turn, mirroring the pattern used
    by the other memory integrations in this repo (e.g. Mem0Memory).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_memory: SerializeAsAny[LlamaIndexMemory] = Field(
        description="Local chat history buffer."
    )
    namespace: Optional[str] = Field(
        default=None, description="InfoLang namespace to read from and write to."
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="Number of chunks to recall per turn.",
    )
    search_msg_limit: int = Field(
        default=DEFAULT_TOP_K,
        description="Number of recent chat messages folded into the recall query "
        "when no explicit `input` is given to `get`.",
    )
    remember_source: Optional[str] = Field(
        default=DEFAULT_SOURCE,
        description="`source` tag attached to memories written by this integration.",
    )

    _client: InfoLang = PrivateAttr()

    def __init__(self, client: Optional[InfoLang] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @model_serializer
    def serialize_memory(self) -> dict:
        # `memory_blocks_template` and `insert_method` on the underlying
        # `Memory` buffer aren't JSON-serializable (e.g. InsertMethod is an
        # enum wrapped in a way workflows' JsonSerializer chokes on), so
        # exclude them -- mirrors Mem0Memory/AgentCoreMemory in this repo.
        return {
            "primary_memory": self.primary_memory.model_dump(
                exclude={"memory_blocks_template", "insert_method"}
            ),
            "namespace": self.namespace,
            "top_k": self.top_k,
            "search_msg_limit": self.search_msg_limit,
            "remember_source": self.remember_source,
        }

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "InfoLangMemory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "InfoLangMemory":
        raise NotImplementedError("Use InfoLangMemory.from_client or from_api_key")

    @classmethod
    def from_client(
        cls,
        client: InfoLang,
        namespace: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        search_msg_limit: int = DEFAULT_TOP_K,
        remember_source: Optional[str] = DEFAULT_SOURCE,
        **kwargs: Any,
    ) -> "InfoLangMemory":
        """Build an InfoLangMemory around an already-constructed InfoLang client."""
        primary_memory = LlamaIndexMemory.from_defaults()
        return cls(
            client=client,
            primary_memory=primary_memory,
            namespace=namespace or client.namespace,
            top_k=top_k,
            search_msg_limit=search_msg_limit,
            remember_source=remember_source,
            **kwargs,
        )

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        base_url: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        search_msg_limit: int = DEFAULT_TOP_K,
        remember_source: Optional[str] = DEFAULT_SOURCE,
        **kwargs: Any,
    ) -> "InfoLangMemory":
        """
        Build an InfoLangMemory from an InfoLang hosted-API key.

        Talks to InfoLang's managed cloud edge (``https://api.infolang.ai``)
        by default, or ``base_url`` when self-hosting the runtime.
        """
        client = InfoLang(api_key=api_key, base_url=base_url, namespace=namespace)
        return cls.from_client(
            client,
            namespace=namespace,
            top_k=top_k,
            search_msg_limit=search_msg_limit,
            remember_source=remember_source,
            **kwargs,
        )

    def _remember(self, message: ChatMessage) -> None:
        if not is_memorable(message):
            return
        self._client.remember(
            message.content,
            namespace=self.namespace,
            source=self.remember_source,
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history, prefixed with a system message of recalled context."""
        messages = self.primary_memory.get(input=input, **kwargs)
        query = convert_messages_to_query(
            messages, input=input, limit=self.search_msg_limit
        )
        if query is None:
            return messages

        result = self._client.recall(query, namespace=self.namespace, top_k=self.top_k)
        if not result.chunks:
            return messages

        existing_system_message = (
            messages[0] if messages and messages[0].role == MessageRole.SYSTEM else None
        )
        system_message = convert_chunks_to_system_message(
            result.chunks, existing_system_message=existing_system_message
        )
        if existing_system_message is not None:
            messages[0] = system_message
        else:
            messages.insert(0, system_message)
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Returns all local chat history (does not query InfoLang)."""
        return self.primary_memory.get_all()

    def put(self, message: ChatMessage) -> None:
        """Add message to chat history and write it to InfoLang memory."""
        self.primary_memory.put(message)
        self._remember(message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history, writing only the newly-added messages to InfoLang."""
        initial_len = len(self.primary_memory.get_all())
        new_messages = messages[initial_len:]
        self.primary_memory.set(messages)
        for message in new_messages:
            self._remember(message)

    def reset(self) -> None:
        """Reset local chat history. Does not delete InfoLang memories."""
        self.primary_memory.reset()
