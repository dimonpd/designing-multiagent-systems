# Microsoft Agent Framework Examples

These examples demonstrate the same patterns from PicoAgents implemented using Microsoft's Agent Framework. Note that Microsoft's Agent Framework is built by the same team that developed AutoGen and Semantic Kernel and is the successor to both libraries.

## Setup

```bash
# Install agent-framework with Azure support
pip install agent-framework[azure]

# Authentication options:

# Option 1: API Key (set environment variables)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4.1-mini"
export AZURE_OPENAI_API_KEY="your-api-key"

# Option 2: Azure CLI (recommended for development)
az login
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4.1-mini"
```

## Examples

### Agents

| Example                       | PicoAgents Equivalent         | Description                             |
| ----------------------------- | ----------------------------- | --------------------------------------- |
| `agents/basic_agent.py`       | `agents/basic-agent.py`       | Agent with weather and calculator tools |
| `agents/memory.py`            | `agents/memory.py`            | Context provider for memory injection   |
| `agents/structured_output.py` | `agents/structured-output.py` | Pydantic model responses                |

### Workflows

| Example                   | PicoAgents Equivalent     | Description               |
| ------------------------- | ------------------------- | ------------------------- |
| `workflows/sequential.py` | `workflows/sequential.py` | Sequential agent pipeline |

### Orchestration

| Example                        | PicoAgents Equivalent          | Description                    |
| ------------------------------ | ------------------------------ | ------------------------------ |
| `orchestration/round_robin.py` | `orchestration/round-robin.py` | Poet and critic collaboration  |
| `orchestration/handoff.py`     | (new pattern)                  | Agent-to-agent handoff routing |

## Key Differences from PicoAgents

1. **Stateless agents**: Agent Framework agents don't store conversation history internally - use `AgentThread` for state
2. **Context providers**: More structured approach to memory injection with `invoking()` and `invoked()` hooks
3. **Workflow builders**: Fluent API pattern (`SequentialBuilder`, `HandoffBuilder`, etc.)
4. **Event streaming**: All operations emit structured `WorkflowEvent` types

## Running Examples

```bash
# From the examples/frameworks/agent-framework directory
python agents/basic_agent.py
python workflows/sequential.py
python orchestration/round_robin.py
```
