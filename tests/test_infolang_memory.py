import json

import httpx
import pytest
import respx

from infolang import InfoLang
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.memory.infolang import InfoLangMemory
from workflows.context.serializers import JsonSerializer

BASE_URL = "https://api.infolang.ai"


def _make_memory(**kwargs) -> InfoLangMemory:
    client = InfoLang(api_key="il_live_test", namespace="ns-test")
    return InfoLangMemory.from_client(client, **kwargs)


@respx.mock
def test_from_api_key_builds_cloud_client():
    memory = InfoLangMemory.from_api_key(api_key="il_live_test", namespace="ns-test")
    assert isinstance(memory, InfoLangMemory)
    assert memory.namespace == "ns-test"
    assert memory._client.namespace == "ns-test"
    assert memory._client._base_url == BASE_URL


def test_from_defaults_not_implemented():
    with pytest.raises(NotImplementedError):
        InfoLangMemory.from_defaults()


def test_from_client_defaults_namespace_from_client():
    client = InfoLang(api_key="il_live_test", namespace="ns-from-client")
    memory = InfoLangMemory.from_client(client)
    assert memory.namespace == "ns-from-client"
    assert memory.top_k == 5
    assert memory.search_msg_limit == 5
    assert memory.remember_source == "llama_index"


@respx.mock
def test_put_writes_user_and_assistant_messages_to_infolang():
    route = respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1", "namespace": "ns-test"})
    )
    memory = _make_memory()

    memory.put(ChatMessage(role=MessageRole.USER, content="My name is Alice"))

    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "text": "My name is Alice",
        "namespace": "ns-test",
        "source": "llama_index",
    }
    assert memory.primary_memory.get_all() == [
        ChatMessage(role=MessageRole.USER, content="My name is Alice")
    ]


@respx.mock
def test_put_skips_system_messages():
    route = respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    memory = _make_memory()

    memory.put(ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful bot"))

    assert route.call_count == 0
    assert len(memory.primary_memory.get_all()) == 1


@respx.mock
def test_put_skips_empty_content():
    route = respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    memory = _make_memory()

    memory.put(ChatMessage(role=MessageRole.USER, content=""))

    assert route.call_count == 0


@respx.mock
def test_set_only_remembers_new_messages():
    route = respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    memory = _make_memory()

    first_batch = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
    ]
    memory.set(first_batch)
    assert route.call_count == 2

    second_batch = [
        *first_batch,
        ChatMessage(role=MessageRole.USER, content="How are you?"),
    ]
    memory.set(second_batch)

    # Only the one new message should have triggered a further remember call.
    assert route.call_count == 3
    last_body = json.loads(route.calls[-1].request.content)
    assert last_body["text"] == "How are you?"
    assert memory.primary_memory.get_all() == second_batch


@respx.mock
def test_get_with_no_history_and_no_input_skips_recall():
    route = respx.post(f"{BASE_URL}/v1/recall")
    memory = _make_memory()

    result = memory.get()

    assert result == []
    assert route.call_count == 0


@respx.mock
def test_get_prepends_system_message_from_recall():
    respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    recall_route = respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {"id": "c1", "text": "User's name is Alice.", "similarity": 0.93},
                    {
                        "id": "c2",
                        "text": "User prefers formal replies.",
                        "similarity": 0.81,
                    },
                ]
            },
        )
    )
    memory = _make_memory()
    memory.put(ChatMessage(role=MessageRole.USER, content="What do you know about me?"))

    result = memory.get()

    assert recall_route.call_count == 1
    request_body = json.loads(recall_route.calls[0].request.content)
    assert request_body["namespace"] == "ns-test"
    assert request_body["top_k"] == 5

    assert result[0].role == MessageRole.SYSTEM
    assert "User's name is Alice." in result[0].content
    # Weak match (score < 0.85) is included but annotated.
    assert "weak match" in result[0].content
    assert "User prefers formal replies." in result[0].content
    assert result[1:] == [
        ChatMessage(role=MessageRole.USER, content="What do you know about me?")
    ]


@respx.mock
def test_get_merges_with_existing_system_message():
    respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200,
            json={"hits": [{"id": "c1", "text": "recalled fact", "similarity": 0.9}]},
        )
    )
    memory = _make_memory()
    memory.primary_memory.put(
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
    )
    memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

    result = memory.get()

    assert result[0].role == MessageRole.SYSTEM
    assert "You are a helpful assistant." in result[0].content
    assert "recalled fact" in result[0].content
    # Only one system message, not two.
    assert sum(1 for m in result if m.role == MessageRole.SYSTEM) == 1


@respx.mock
def test_get_explicit_input_used_as_query():
    respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_1"})
    )
    recall_route = respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    memory = _make_memory()

    result = memory.get(input="explicit query")

    assert recall_route.call_count == 1
    body = json.loads(recall_route.calls[0].request.content)
    assert body["query"] == "explicit query"
    # No hits -> no system message injected.
    assert result == []


def test_get_all_returns_local_history_only():
    memory = _make_memory()
    memory.primary_memory.put(ChatMessage(role=MessageRole.USER, content="hi"))
    assert memory.get_all() == [ChatMessage(role=MessageRole.USER, content="hi")]


def test_reset_clears_local_history_only():
    memory = _make_memory()
    memory.primary_memory.put(ChatMessage(role=MessageRole.USER, content="hi"))
    memory.reset()
    assert memory.get_all() == []


def test_class_name():
    assert InfoLangMemory.class_name() == "InfoLangMemory"


def test_ser_deser_memory():
    memory = _make_memory()
    element = memory.model_dump()
    assert "primary_memory" in element
    assert element["namespace"] == "ns-test"
    assert element["top_k"] == 5

    serialized = JsonSerializer().serialize(memory)
    assert serialized is not None

    deserialized = JsonSerializer().deserialize(serialized)
    assert isinstance(deserialized, InfoLangMemory)
    assert deserialized.namespace == "ns-test"
