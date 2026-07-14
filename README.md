# LlamaIndex Memory Integration: InfoLang

[InfoLang](https://infolang.ai) is a hosted semantic memory API: `remember` stores text
into a namespace, `recall`/`recall_hybrid` retrieves the most relevant chunks for a
query. This package adapts it to LlamaIndex's `BaseMemory` interface.

## Installation

```bash
pip install llama-index-memory-infolang
```

This pulls in `infolang` (the client SDK) as a dependency.

## Usage

```python
from llama_index.memory.infolang import InfoLangMemory

memory = InfoLangMemory.from_api_key(
    api_key="il_live_...",
    namespace="user-123",  # optional; falls back to the key's default namespace
    top_k=5,  # optional, default is 5
)
```

You can also pass an already-constructed `infolang.InfoLang` client, e.g. to point at
a self-hosted runtime or reuse an existing client:

```python
from infolang import InfoLang
from llama_index.memory.infolang import InfoLangMemory

client = InfoLang(dev_key="il_dev_...", base_url="http://127.0.0.1:8766")
memory = InfoLangMemory.from_client(client, namespace="user-123")
```

Each `put`/`set` call writes new user/assistant messages to InfoLang via `remember`.
Each `get` call folds the current turn (or an explicit `input` string) into an InfoLang
`recall`, and prepends the recalled chunks to the local chat history as a system
message — local chat history itself is kept in-process and is not read back from
InfoLang, so `reset()` only clears the local buffer; it does not delete anything
server-side.

### With a chat engine

```python
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleChatEngine

os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
llm = OpenAI(model="gpt-4o")

chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
response = chat_engine.chat("Hi, my name is Mayank")
print(response)
```

### With an agent

```python
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(tools=[...], llm=llm)
response = await agent.run("Hi, my name is Mayank", memory=memory)
print(response)
```

> Note: For a full walkthrough see the [example notebook](https://github.com/InfoLang-Inc/llama-index-memory-infolang/blob/main/examples/InfoLangMemory.ipynb).

## Configuration

`InfoLangMemory.from_api_key` / `from_client` accept:

- `namespace` — InfoLang namespace to read from and write to; defaults to the client's own namespace.
- `top_k` — number of chunks to recall per turn (default `5`).
- `search_msg_limit` — number of recent chat messages folded into the recall query when `get` isn't given an explicit `input` (default `5`).
- `remember_source` — `source` tag attached to memories written by this integration (default `"llama_index"`).

## References

- [InfoLang](https://infolang.ai)
- [InfoLang Python SDK](https://infolang.ai/docs/sdk/python)

## License

MIT — see [LICENSE](LICENSE).
