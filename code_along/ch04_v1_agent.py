"""
PicoAgents Code Along - Chapter 4.1: The Core Agent

Minimal Agent with the SAME API as picoagents. Swap the import to use full library.

What this implements:
- Agent class with name, instructions, model
- run() method that calls LLM and returns response

What's omitted (see later versions or full library):
- Tools, Memory, Streaming, Middleware

Run: python ch04_v1_agent.py

Model Client: Uses Azure OpenAI. For other clients:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()  # Uses OPENAI_API_KEY
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

from openai import AsyncAzureOpenAI


@dataclass
class Message:
    content: str
    source: str = "assistant"


@dataclass
class UserMessage(Message):
    source: str = "user"


@dataclass
class AssistantMessage(Message):
    source: str = "assistant"


@dataclass
class AgentResponse:
    messages: List[Message] = field(default_factory=list)
    source: str = ""

    @property
    def final_content(self) -> str:
        return self.messages[-1].content if self.messages else ""


class Agent:
    """
    Minimal Agent - same interface as picoagents.Agent.

    Usage:
        agent = Agent(name="assistant", instructions="You are helpful.")
        response = await agent.run("What is 2+2?")
        print(response.final_content)
    """

    def __init__(
        self,
        name: str,
        instructions: str = "You are a helpful assistant.",
        model: str = "gpt-4.1-mini",
        tools: Optional[List] = None,  # API compat - not used in v1
        memory=None,                    # API compat - not used in v1
        description: str = "",
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.description = description or f"Agent: {name}"
        self._tools = tools
        self._memory = memory

        self._client = AsyncAzureOpenAI(api_version="2024-12-01-preview")

    async def run(self, task: str) -> AgentResponse:
        """Execute agent on a task."""
        messages: List[Any] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": task}
        ]

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        content = response.choices[0].message.content or ""
        return AgentResponse(
            messages=[
                Message(content=task, source="user"),
                Message(content=content, source="assistant")
            ],
            source=self.name
        )


async def main():
    print("=== Code Along v1: Core Agent ===\n")

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Be concise.",
        model="gpt-4.1-mini"
    )

    response = await agent.run("What is the capital of France?")
    print(f"Agent: {response.final_content}")


if __name__ == "__main__":
    asyncio.run(main())
