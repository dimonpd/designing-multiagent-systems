"""
PicoAgents From Scratch - Chapter 4.4: Adding Streaming

Builds on v3 by adding streaming support. Same API as picoagents.

What this adds:
- run_stream() async generator for real-time output
- Event types for different stages (tool calls, results, etc.)

What's omitted (see full library):
- Middleware, OpenTelemetry, CancellationTokens, Component serialization

Run: python ch04_v4_streaming.py

Model Client: Uses Azure OpenAI. See ch04_v1_agent.py for alternatives.
"""

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from openai import AsyncAzureOpenAI


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


# --- Events for streaming ---

@dataclass
class ToolCallEvent:
    """Emitted when agent calls a tool."""
    tool_name: str
    parameters: Dict[str, Any]
    source: str = ""


@dataclass
class ToolResultEvent:
    """Emitted when tool returns result."""
    tool_name: str
    result: str
    source: str = ""


# --- Memory ---

@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ListMemory:
    def __init__(self, max_memories: int = 100):
        self.memories: List[MemoryItem] = []
        self.max_memories = max_memories

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.memories.append(MemoryItem(content=content, metadata=metadata or {}))
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    async def get_context(self, max_items: int = 10) -> List[str]:
        return [m.content for m in self.memories[-max_items:]]


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


# Type alias for stream items
StreamItem = Union[Message, ToolCallEvent, ToolResultEvent, AgentResponse]


# --- Agent ---

class Agent:
    """
    Agent with streaming support - same interface as picoagents.Agent.

    Provides both run() and run_stream() methods.
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
        self._tool_schemas: List[Dict] = []
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
        """Execute agent and return final response."""
        response = None
        async for item in self.run_stream(task):
            if isinstance(item, AgentResponse):
                response = item
        return response or AgentResponse(messages=[], source=self.name)

    async def run_stream(self, task: str) -> AsyncGenerator[StreamItem, None]:
        """
        Execute agent with streaming output.

        Yields messages and events as they occur, enabling real-time UI updates.
        """
        all_messages: List[Message] = []

        # Yield user message
        user_msg = Message(content=task, source="user")
        all_messages.append(user_msg)
        yield user_msg

        system_content = await self._prepare_system_message()
        api_messages = [{"role": "system", "content": system_content}]
        api_messages.extend(self._message_history)
        api_messages.append({"role": "user", "content": task})

        for _ in range(self.max_iterations):
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=self._tool_schemas if self._tool_schemas else None
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                assistant_msg = Message(content=content, source="assistant")
                all_messages.append(assistant_msg)
                yield assistant_msg

                # Update history
                self._message_history.append({"role": "user", "content": task})
                self._message_history.append({"role": "assistant", "content": content})

                if self.memory:
                    await self.memory.add(f"User: {task[:100]}")
                    await self.memory.add(f"Assistant: {content[:100]}")

                # Yield final response
                yield AgentResponse(messages=all_messages, source=self.name)
                return

            # Execute tool calls
            api_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                # Yield tool call event
                yield ToolCallEvent(tool_name=name, parameters=args, source=self.name)

                result = self._execute_tool(name, args)

                # Yield tool result event
                yield ToolResultEvent(tool_name=name, result=result, source=self.name)

                tool_msg = ToolMessage(content=result, tool_call_id=tc.id, tool_name=name)
                all_messages.append(tool_msg)
                yield tool_msg

                api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Max iterations
        final_msg = Message(content="Max iterations reached.", source="assistant")
        all_messages.append(final_msg)
        yield final_msg
        yield AgentResponse(messages=all_messages, source=self.name)

    def reset(self) -> None:
        self._message_history = []


# Example tools
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is sunny, 72°F"


def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


async def main():
    print("=== From Scratch v4: With Streaming ===\n")

    agent = Agent(
        name="assistant",
        instructions="You are helpful. Use tools when appropriate.",
        model="gpt-4.1-mini",
        tools=[get_weather, calculate]
    )

    print("Query: What's the weather in Tokyo and what is 15 * 24?\n")
    print("--- Streaming events ---")

    async for item in agent.run_stream("What's the weather in Tokyo and what is 15 * 24?"):
        if isinstance(item, Message):
            prefix = f"[{item.source}]"
            print(f"{prefix} {item.content[:80]}{'...' if len(item.content) > 80 else ''}")
        elif isinstance(item, ToolCallEvent):
            print(f"[tool_call] {item.tool_name}({item.parameters})")
        elif isinstance(item, ToolResultEvent):
            print(f"[tool_result] {item.result}")
        elif isinstance(item, AgentResponse):
            print(f"\n--- Final Response ---")
            print(f"Agent: {item.final_content}")


if __name__ == "__main__":
    asyncio.run(main())
