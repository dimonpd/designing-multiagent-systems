"""
PicoAgents Code Along - Chapter 4.2: Adding Tools

Builds on v1 by adding tool calling. Same API as picoagents.

What this adds:
- tools parameter accepts Python functions
- Automatic function-to-schema conversion
- Tool execution loop

What's omitted (see later versions or full library):
- Memory, Streaming, Middleware, BaseTool class

Run: python ch04_v2_tools.py

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


def _get_type_string(annotation) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return type_map.get(annotation, "string")


def _function_to_schema(func: Callable) -> Dict[str, Any]:
    """Convert a Python function to OpenAI tool schema."""
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


class Agent:
    """
    Agent with tool calling - same interface as picoagents.Agent.

    Usage:
        def get_weather(location: str) -> str:
            return f"Sunny in {location}"

        agent = Agent(name="assistant", tools=[get_weather])
        response = await agent.run("Weather in Paris?")
    """

    def __init__(
        self,
        name: str,
        instructions: str = "You are a helpful assistant.",
        model: str = "gpt-4.1-mini",
        tools: Optional[List[Callable]] = None,
        memory=None,  # API compat - not used in v2
        description: str = "",
        max_iterations: int = 10,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.description = description or f"Agent: {name}"
        self.max_iterations = max_iterations
        self._memory = memory

        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Any] = []
        if tools:
            for tool in tools:
                self._tools[tool.__name__] = tool
                self._tool_schemas.append(_function_to_schema(tool))

        self._client = AsyncAzureOpenAI(api_version="2024-12-01-preview")

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' not found"
        try:
            return str(self._tools[name](**args))
        except Exception as e:
            return f"Error: {e}"

    async def run(self, task: str) -> AgentResponse:
        """Execute agent with tool calling loop."""
        all_messages: List[Message] = [Message(content=task, source="user")]
        api_messages: List[Any] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": task}
        ]

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
    print("=== Code Along v2: With Tools ===\n")

    agent = Agent(
        name="assistant",
        instructions="You are helpful. Use tools when appropriate.",
        model="gpt-4.1-mini",
        tools=[get_weather, calculate]
    )

    print("Query: What's the weather in Tokyo and what is 15 * 24?\n")
    response = await agent.run("What's the weather in Tokyo and what is 15 * 24?")
    print(f"\nAgent: {response.final_content}")


if __name__ == "__main__":
    asyncio.run(main())
