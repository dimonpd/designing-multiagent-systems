"""
PicoAgents Code Along - Chapter 4.3: Adding Memory

Builds on v2 by adding memory for context across conversations. Same API as picoagents.

What this adds:
- ListMemory for in-memory storage
- Message history persists across run() calls
- Memory context injected into prompts

What's omitted (see later versions or full library):
- Streaming, Middleware, BaseMemory abstract class, vector/RAG memory

Run: python ch04_v3_memory.py

Model Client: Uses Azure OpenAI. See ch04_v1_agent.py for alternatives.
"""

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncAzureOpenAI, NOT_GIVEN


@dataclass
class Message:
    content: str
    source: str = "assistant"


@dataclass
class ToolMessage(Message):
    source: str = "tool"
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass
class AgentResponse:
    messages: List[Message] = field(default_factory=list)
    source: str = ""

    @property
    def final_content(self) -> str:
        for msg in reversed(self.messages):
            if msg.source == "assistant" and msg.content:
                return msg.content
        return ""


# --- Memory ---

@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ListMemory:
    """Simple list-based memory - same interface as picoagents.memory.ListMemory."""

    def __init__(self, max_memories: int = 100):
        self.memories: List[MemoryItem] = []
        self.max_memories = max_memories

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.memories.append(MemoryItem(content=content, metadata=metadata or {}))
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    async def query(self, query: str, max_results: int = 10) -> List[str]:
        query_lower = query.lower()
        return [m.content for m in reversed(self.memories)
                if query_lower in m.content.lower()][:max_results]

    async def get_context(self, max_items: int = 10) -> List[str]:
        return [m.content for m in self.memories[-max_items:]]

    async def clear(self) -> None:
        self.memories = []


# --- Tool utilities ---

def _get_type_string(annotation) -> str:
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return type_map.get(annotation, "string")


def _function_to_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            param_type = _get_type_string(param.annotation)
        properties[name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {"type": "object", "properties": properties, "required": required}
        }
    }


# --- Agent ---

class Agent:
    """
    Agent with tools and memory - same interface as picoagents.Agent.

    Memory enables conversation context across multiple run() calls.
    """

    def __init__(
        self,
        name: str,
        instructions: str = "You are a helpful assistant.",
        model: str = "gpt-4.1-mini",
        tools: Optional[List[Callable]] = None,
        memory: Optional[ListMemory] = None,
        description: str = "",
        max_iterations: int = 10,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.description = description or f"Agent: {name}"
        self.max_iterations = max_iterations
        self.memory = memory

        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Any] = []
        if tools:
            for tool in tools:
                self._tools[tool.__name__] = tool
                self._tool_schemas.append(_function_to_schema(tool))

        self._message_history: List[Dict] = []
        self._client = AsyncAzureOpenAI(api_version="2024-12-01-preview")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' not found"
        try:
            return str(self._tools[name](**args))
        except Exception as e:
            return f"Error: {e}"

    async def _prepare_system_message(self) -> str:
        system_content = self.instructions
        if self.memory:
            context = await self.memory.get_context(max_items=5)
            if context:
                system_content += f"\n\nContext from memory:\n" + "\n".join(f"- {c}" for c in context)
        return system_content

    async def run(self, task: str) -> AgentResponse:
        """Execute agent with memory support."""
        all_messages: List[Message] = [Message(content=task, source="user")]

        system_content = await self._prepare_system_message()
        api_messages: List[Any] = [{"role": "system", "content": system_content}]
        api_messages.extend(self._message_history)
        api_messages.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=self._tool_schemas if self._tool_schemas else NOT_GIVEN
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                all_messages.append(Message(content=content, source="assistant"))

                # Update history for next run()
                self._message_history.append({"role": "user", "content": task})
                self._message_history.append({"role": "assistant", "content": content})

                # Store in memory
                if self.memory:
                    await self.memory.add(f"User: {task[:100]}")
                    await self.memory.add(f"Assistant: {content[:100]}")

                return AgentResponse(messages=all_messages, source=self.name)

            # Execute tool calls
            api_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}  # type: ignore[union-attr]
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                name = tc.function.name  # type: ignore[union-attr]
                args = json.loads(tc.function.arguments)  # type: ignore[union-attr]
                print(f"  [tool] {name}({args})")

                result = self._execute_tool(name, args)
                print(f"  [result] {result}")

                all_messages.append(ToolMessage(content=result, tool_call_id=tc.id, tool_name=name))
                api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        all_messages.append(Message(content="Max iterations reached.", source="assistant"))
        return AgentResponse(messages=all_messages, source=self.name)

    def reset(self) -> None:
        """Clear conversation history (memory persists)."""
        self._message_history = []


# Example tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is sunny, 72°F"


async def main():
    print("=== Code Along v3: With Memory ===\n")

    memory = ListMemory()
    agent = Agent(
        name="assistant",
        instructions="You are helpful. Remember what the user tells you.",
        model="gpt-4.1-mini",
        tools=[get_weather],
        memory=memory
    )

    # First conversation
    print("--- Turn 1 ---")
    print("User: My name is Alice and I live in Paris.")
    r1 = await agent.run("My name is Alice and I live in Paris.")
    print(f"Agent: {r1.final_content}\n")

    # Second conversation - agent should remember
    print("--- Turn 2 ---")
    print("User: What's the weather where I live?")
    r2 = await agent.run("What's the weather where I live?")
    print(f"Agent: {r2.final_content}\n")

    # Third conversation
    print("--- Turn 3 ---")
    print("User: What's my name?")
    r3 = await agent.run("What's my name?")
    print(f"Agent: {r3.final_content}\n")

    # Show memory
    print("--- Memory ---")
    for item in await memory.get_context(10):
        print(f"  {item}")


if __name__ == "__main__":
    asyncio.run(main())
