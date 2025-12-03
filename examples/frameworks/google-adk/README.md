# Google ADK (Agent Development Kit) Examples

These examples demonstrate the same patterns from PicoAgents implemented using Google's Agent Development Kit (ADK). ADK is Google's framework for building AI agents, optimized for Gemini models.

## Setup

```bash
# Install Google ADK
pip install google-adk

# Set your Google API key
export GOOGLE_API_KEY="your-api-key"

# Or use Application Default Credentials
gcloud auth application-default login
```

## Examples

### Agents

| Example                       | PicoAgents Equivalent         | Description                             |
| ----------------------------- | ----------------------------- | --------------------------------------- |
| `agents/basic_agent.py`       | `agents/basic-agent.py`       | Agent with weather and calculator tools |
| `agents/memory.py`            | `agents/memory.py`            | Session state for memory management     |
| `agents/structured_output.py` | `agents/structured-output.py` | Pydantic model responses                |

### Workflows

| Example                   | PicoAgents Equivalent     | Description               |
| ------------------------- | ------------------------- | ------------------------- |
| `workflows/sequential.py` | `workflows/sequential.py` | Sequential agent pipeline |

### Orchestration

| Example                       | PicoAgents Equivalent          | Description                   |
| ----------------------------- | ------------------------------ | ----------------------------- |
| `orchestration/loop_agent.py` | `orchestration/round-robin.py` | Poet and critic collaboration |
| `orchestration/parallel.py`   | (new pattern)                  | Parallel agent execution      |

## Key Differences from PicoAgents

1. **Google-optimized**: Default model is Gemini, tight integration with Google Cloud
2. **ToolContext**: Tools receive `ToolContext` with `state`, `user_id`, `session_id`
3. **Output key pattern**: Agents store output to session state via `output_key`
4. **State interpolation**: Instructions can reference state variables: `{generated_code}`
5. **Native workflow agents**: `SequentialAgent`, `ParallelAgent`, `LoopAgent` are first-class

## Running Examples

```bash
# From the examples/frameworks/google-adk directory
python agents/basic_agent.py
python workflows/sequential.py
python orchestration/loop_agent.py
```

## Model Configuration

All examples use `gemini-flash-latest` as the default model. You can also use:

- `gemini-flash-latest` - Fast, efficient (default)
- `gemini-1.5-pro` - More capable for complex tasks
- OpenAI/Anthropic models via `OpenAiLlm` or `AnthropicLlm`
