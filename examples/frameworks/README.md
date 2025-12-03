# Framework Comparison Examples

This directory contains equivalent implementations of multi-agent patterns across different frameworks. The goal is to demonstrate that the core patterns taught in this book are universal and can be implemented in any framework.

## Structure

Each framework subdirectory mirrors the main `examples/` structure:

```
frameworks/
├── agent-framework/     # Microsoft Agent Framework
│   ├── agents/          # Basic agents, memory, structured output
│   ├── workflows/       # Sequential workflows
│   └── orchestration/   # Round-robin, handoff patterns
├── google-adk/          # Google Agent Development Kit
│   ├── agents/          # Basic agents, memory, structured output
│   ├── workflows/       # Sequential workflows
│   └── orchestration/   # Loop agent, parallel patterns
└── langgraph/           # LangGraph (LangChain's graph-based agents)
    ├── agents/          # Basic agents, memory, structured output
    ├── workflows/       # Sequential workflows
    └── orchestration/   # Round-robin, supervisor patterns
```

## Pattern Mapping

| Pattern | PicoAgents | Agent Framework | Google ADK | LangGraph |
|---------|------------|-----------------|------------|-----------|
| Basic agent with tools | `Agent` + function | `ChatAgent` + `@ai_function` | `Agent` + function | `create_react_agent` + `@tool` |
| Memory/context | `ListMemory` | `ContextProvider` | `ToolContext.state` | `MemorySaver` checkpointer |
| Sequential workflow | `Workflow.chain()` | `SequentialBuilder` | `SequentialAgent` | `StateGraph` + edges |
| Round-robin orchestration | `RoundRobinOrchestrator` | `WorkflowBuilder` (cyclic) | `LoopAgent` | `StateGraph` + conditional edges |
| Parallel orchestration | (manual asyncio) | `ConcurrentBuilder` | `ParallelAgent` | (manual asyncio) |
| Supervisor orchestration | `SupervisorOrchestrator` | (manual) | (sub_agents) | `StateGraph` + routing |
| Handoff orchestration | (manual) | `HandoffBuilder` | (sub_agents routing) | `Command`/`Send` |
| Structured output | `output_format=Model` | `response_format=Model` | `response_schema=Model` | `with_structured_output` |

## Running Examples

### Microsoft Agent Framework

```bash
# Install agent-framework
pip install agent-framework[azure]

# Set environment variables
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"  # Or use Azure CLI auth

# Run examples
python examples/frameworks/agent-framework/agents/basic_agent.py
python examples/frameworks/agent-framework/orchestration/round_robin.py
```

### Google ADK

```bash
# Install Google ADK
pip install google-adk

# Set environment variable
export GOOGLE_API_KEY="your-api-key"

# Run examples
python examples/frameworks/google-adk/agents/basic_agent.py
python examples/frameworks/google-adk/orchestration/loop_agent.py
```

### LangGraph

```bash
# Install LangGraph and LangChain OpenAI
pip install langgraph langchain-openai python-dotenv

# Option 1: Azure OpenAI (used in examples)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"

# Option 2: OpenAI directly (see README for code changes)
export OPENAI_API_KEY="your-api-key"

# Run examples
python examples/frameworks/langgraph/agents/basic_agent.py
python examples/frameworks/langgraph/workflows/sequential.py
python examples/frameworks/langgraph/orchestration/supervisor.py
```

## Comparison Philosophy

These examples aim to:

1. **Show equivalence**: Same task, same behavior, different syntax
2. **Highlight patterns**: Core patterns are framework-agnostic
3. **Be practical**: Runnable examples, not pseudo-code
4. **Stay focused**: Only replicate what maps cleanly between frameworks

We intentionally skip framework-specific features that don't have clear equivalents.
