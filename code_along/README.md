# PicoAgents Code Along

Minimal, progressive implementations of the picoagents library - **same API, fewer lines**.

These files let you build understanding incrementally. Each version is self-contained and runnable. Code you write works unchanged with the full picoagents library.

## Files

| File | Lines | What It Adds | Chapter Section |
|------|-------|--------------|-----------------|
| `ch04_v1_agent.py` | ~100 | Core agent loop, `run()` | 4.1-4.3 |
| `ch04_v2_tools.py` | ~180 | Tool calling, function-to-schema | 4.4 |
| `ch04_v3_memory.py` | ~230 | ListMemory, conversation history | 4.5 |
| `ch04_v4_streaming.py` | ~260 | `run_stream()`, event types | 4.1 |

## Quick Start

```bash
# Set your API keys
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="..."

# Or for OpenAI (modify the client in the file)
export OPENAI_API_KEY="..."

# Run any version
python ch04_v1_agent.py
python ch04_v2_tools.py
python ch04_v3_memory.py
python ch04_v4_streaming.py
```

## Same API as Full Library

```python
# Code-along version
from ch04_v2_tools import Agent

# Full library - just change the import
from picoagents import Agent

# Same code works with both
agent = Agent(
    name="assistant",
    instructions="You are helpful.",
    tools=[get_weather]
)
response = await agent.run("What's the weather?")
print(response.final_content)
```

## What's Included vs Omitted

| Feature | Code Along | Full Library |
|---------|--------------|--------------|
| Agent class | ✓ | ✓ |
| `run()` method | ✓ | ✓ |
| `run_stream()` | ✓ (v4) | ✓ |
| Tool calling | ✓ (v2+) | ✓ |
| ListMemory | ✓ (v3+) | ✓ |
| Middleware | - | ✓ |
| OpenTelemetry | - | ✓ |
| CancellationTokens | - | ✓ |
| Component serialization | - | ✓ |
| BaseTool abstract class | - | ✓ |
| BaseMemory abstract class | - | ✓ |
| ChromaDB/vector memory | - | ✓ |
| Agents-as-tools | - | ✓ |

## Model Client

These examples use Azure OpenAI. To use other providers, modify the client initialization:

```python
# Azure OpenAI (default)
from openai import AsyncAzureOpenAI
client = AsyncAzureOpenAI(api_version="2024-12-01-preview")

# OpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI()  # Uses OPENAI_API_KEY

# Any OpenAI-compatible endpoint
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="unused")
```

## Why This Exists

Chapter 4 teaches agent concepts progressively, but the full library is 1000+ lines with production concerns. These files bridge the gap:

1. **Runnable at each stage** - Not just snippets, complete programs
2. **Same API** - Code transfers to full library
3. **Minimal** - Only what's needed to understand the concept
4. **Progressive** - Each version builds on the last
