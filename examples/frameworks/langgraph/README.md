# LangGraph Examples

These examples demonstrate the same patterns from PicoAgents implemented using LangGraph. LangGraph is LangChain's framework for building stateful, multi-actor applications with LLMs, using a graph-based approach.

## Setup

```bash
# Install LangGraph
pip install langgraph langchain-openai python-dotenv

# Option 1: Azure OpenAI (used in examples)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"

# Option 2: OpenAI directly
export OPENAI_API_KEY="your-api-key"
```

## Examples

### Agents

| Example                       | PicoAgents Equivalent         | Description                             |
| ----------------------------- | ----------------------------- | --------------------------------------- |
| `agents/basic_agent.py`       | `agents/basic-agent.py`       | ReAct agent with weather and calculator |
| `agents/memory.py`            | `agents/memory.py`            | Checkpointer-based conversation memory  |
| `agents/structured_output.py` | `agents/structured-output.py` | Pydantic model responses                |

### Workflows

| Example                   | PicoAgents Equivalent     | Description               |
| ------------------------- | ------------------------- | ------------------------- |
| `workflows/sequential.py` | `workflows/sequential.py` | Sequential node pipeline  |

### Orchestration

| Example                        | PicoAgents Equivalent          | Description                      |
| ------------------------------ | ------------------------------ | -------------------------------- |
| `orchestration/round_robin.py` | `orchestration/round-robin.py` | Cyclic graph for agent turns     |
| `orchestration/supervisor.py`  | `orchestration/supervisor.py`  | Supervisor-controlled delegation |

## Key Differences from PicoAgents

1. **Graph-centric**: Everything is nodes and edges in a StateGraph
2. **Channels & Reducers**: State is managed through typed channels with reducers
3. **Checkpointing**: Memory is handled via checkpointers (MemorySaver, SQLite, etc.)
4. **Conditional routing**: Edges can be conditional based on state
5. **Built-in ReAct**: `create_react_agent` provides tool-calling agents

## Running Examples

```bash
# From the examples/frameworks/langgraph directory
python agents/basic_agent.py
python workflows/sequential.py
python orchestration/round_robin.py
```

## Model Configuration

Examples use Azure OpenAI with `gpt-4.1-mini` via `AzureChatOpenAI`. To use OpenAI directly instead, change the import and client:

```python
# Azure OpenAI (current)
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", ...)

# OpenAI directly
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", ...)
```

Compatible models:

- `gpt-4o-mini` / `gpt-4.1-mini` - Fast, cost-effective
- `gpt-4o` - More capable for complex tasks
- Any LangChain-compatible model (Anthropic, Google, etc.)
