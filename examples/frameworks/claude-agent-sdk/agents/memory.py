"""
Memory/Conversation Example - Claude Agent SDK

Equivalent to: examples/agents/memory.py (PicoAgents)

This example demonstrates multi-turn conversations with context preservation
using Anthropic's Claude Agent SDK. The ClaudeSDKClient maintains conversation
state across multiple queries within a session.

Run: python examples/frameworks/claude-agent-sdk/agents/memory.py

Prerequisites:
    - pip install claude-agent-sdk
    - Claude Code CLI installed (https://docs.anthropic.com/en/docs/claude-code)
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


async def main():
    """Demonstrate multi-turn conversation with memory."""
    print("=== Memory Example (Claude Agent SDK) ===\n")

    # Configure agent options
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a helpful assistant. Remember information the user "
            "tells you and use it in future responses. Be concise."
        ),
        permission_mode="bypassPermissions",
        max_turns=5,
    )

    # Use ClaudeSDKClient for stateful multi-turn conversations
    # The client maintains conversation context across queries
    async with ClaudeSDKClient(options=options) as client:
        print("Starting multi-turn conversation...\n")

        # Connect and start the session
        await client.connect()

        # First turn: introduce information
        print("User: My name is Alice and I'm a software engineer.")
        await client.query("My name is Alice and I'm a software engineer.")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Assistant: {block.text}\n")

        # Second turn: ask about previously mentioned info
        print("User: What's my profession?")
        await client.query("What's my profession?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Assistant: {block.text}\n")

        # Third turn: add more context
        print("User: I'm working on a Python project about AI agents.")
        await client.query("I'm working on a Python project about AI agents.")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Assistant: {block.text}\n")

        # Fourth turn: reference both pieces of context
        print("User: Given what you know about me, what advice would you give?")
        await client.query(
            "Given what you know about me, what advice would you give?"
        )

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Assistant: {block.text}\n")
            elif isinstance(message, ResultMessage):
                print(f"[Session: {message.session_id}, "
                      f"Turns: {message.num_turns}]")


async def demonstrate_session_resume():
    """Show how to resume a previous session."""
    print("\n=== Session Resume Example ===\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant. Be concise.",
        permission_mode="bypassPermissions",
        max_turns=3,
    )

    # First session: establish context
    session_id = None
    async with ClaudeSDKClient(options=options) as client:
        await client.connect()
        await client.query("Remember this number: 42")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"First session: {block.text}")
            elif isinstance(message, ResultMessage):
                session_id = message.session_id
                print(f"[Session ID saved: {session_id}]\n")

    # Resume the session later
    if session_id:
        resume_options = ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            system_prompt="You are a helpful assistant. Be concise.",
            permission_mode="bypassPermissions",
            continue_conversation=True,
            resume=session_id,
            max_turns=3,
        )

        async with ClaudeSDKClient(options=resume_options) as client:
            await client.connect()
            await client.query("What number did I ask you to remember?")

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"Resumed session: {block.text}")


if __name__ == "__main__":
    asyncio.run(main())
    # Uncomment to test session resume (requires persistent storage)
    # asyncio.run(demonstrate_session_resume())
