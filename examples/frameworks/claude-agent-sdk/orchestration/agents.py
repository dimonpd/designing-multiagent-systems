"""
Multi-Agent Orchestration Example - Claude Agent SDK

Equivalent to: examples/orchestration/roundrobin.py (PicoAgents)

This example demonstrates multi-agent orchestration using the Claude Agent SDK's
custom agents feature. Unlike PicoAgents' explicit round-robin or handoff patterns,
Claude Agent SDK uses a declarative approach where you define agent capabilities
and Claude decides when to delegate to each agent.

Run: python examples/frameworks/claude-agent-sdk/orchestration/agents.py

Prerequisites:
    - pip install claude-agent-sdk
    - Claude Code CLI installed (https://docs.anthropic.com/en/docs/claude-code)
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


async def multi_agent_example():
    """Demonstrate multi-agent delegation with custom agents."""
    print("=== Multi-Agent Orchestration (Claude Agent SDK) ===\n")

    # Define specialized agents
    # Claude will automatically delegate tasks to the appropriate agent
    # based on the task description and agent capabilities
    agents = {
        "code-reviewer": AgentDefinition(
            description=(
                "Reviews code for bugs, security issues, and best practices. "
                "Use this agent when you need to analyze or review code."
            ),
            prompt=(
                "You are an expert code reviewer. Analyze code for:\n"
                "- Bugs and logic errors\n"
                "- Security vulnerabilities\n"
                "- Performance issues\n"
                "- Code style and best practices\n"
                "Provide specific, actionable feedback."
            ),
            tools=["Read", "Grep"],
            model="sonnet",
        ),
        "doc-writer": AgentDefinition(
            description=(
                "Writes technical documentation, README files, and API docs. "
                "Use this agent when you need to create or improve documentation."
            ),
            prompt=(
                "You are a technical writer. Create clear, comprehensive "
                "documentation that includes:\n"
                "- Clear explanations for different skill levels\n"
                "- Code examples where appropriate\n"
                "- Proper formatting and structure\n"
                "Focus on clarity and completeness."
            ),
            tools=["Read", "Write"],
            model="sonnet",
        ),
        "test-generator": AgentDefinition(
            description=(
                "Generates unit tests and test cases for code. "
                "Use this agent when you need to create tests."
            ),
            prompt=(
                "You are a testing expert. Generate comprehensive tests that:\n"
                "- Cover edge cases and error conditions\n"
                "- Follow testing best practices (AAA pattern)\n"
                "- Use appropriate mocking where needed\n"
                "- Are clear and maintainable"
            ),
            tools=["Read", "Write"],
            model="sonnet",
        ),
    }

    # Configure options with custom agents
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a senior software engineer with access to specialized "
            "agents. Delegate tasks to the appropriate agent based on their "
            "expertise. Coordinate their work and synthesize results."
        ),
        agents=agents,
        permission_mode="bypassPermissions",
        max_turns=15,
    )

    print("Configured agents:")
    for name, agent in agents.items():
        print(f"  - {name}: {agent.description[:50]}...")
    print()

    # Task that requires multiple agents
    task = (
        "I have a Python function that calculates factorial. "
        "Please: 1) Review it for issues, 2) Write documentation for it, "
        "and 3) Generate test cases. Here's the code:\n\n"
        "def factorial(n):\n"
        "    if n < 0:\n"
        "        return None\n"
        "    if n == 0:\n"
        "        return 1\n"
        "    return n * factorial(n-1)"
    )

    print(f"Task: {task[:100]}...\n")
    print("=" * 60)

    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    # Print response, handling potential agent delegation info
                    text = block.text
                    if text.strip():
                        print(f"\n{text}")
        elif isinstance(message, ResultMessage):
            print("\n" + "=" * 60)
            print(f"[Completed in {message.num_turns} turns, "
                  f"Cost: ${message.total_cost_usd:.4f}]")


async def hook_based_routing():
    """Alternative: Use hooks for fine-grained control over agent routing."""
    print("\n=== Hook-Based Routing Example ===\n")
    print("Hooks allow programmatic control over when agents are invoked.\n")

    # This example shows the concept - actual implementation would use
    # the hooks parameter in ClaudeAgentOptions

    from claude_agent_sdk import (
        HookContext,
        HookInput,
        HookJSONOutput,
        HookMatcher,
    )

    async def routing_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Route tasks to appropriate agents based on content."""
        # Analyze the input and decide routing
        # This is a simplified example - real implementation would be
        # more sophisticated
        return {
            "continue_": True,
            "suppressOutput": False,
        }

    # Define hook-based options (conceptual)
    options_with_hooks = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant.",
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="Task",  # Hook into Task tool
                    hooks=[routing_hook],
                    timeout=30.0,
                )
            ],
        },
        permission_mode="bypassPermissions",
        max_turns=10,
    )

    print("Hook-based routing configured (conceptual example)")
    print("Hooks intercept tool calls and can modify behavior dynamically.")


async def simple_delegation():
    """Simpler pattern: Single agent delegation via prompt."""
    print("\n=== Simple Delegation Pattern ===\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a project coordinator. When asked to perform specialized "
            "tasks, describe what a specialist would do and provide the output "
            "as if delegating to that specialist."
        ),
        permission_mode="bypassPermissions",
        max_turns=5,
    )

    tasks = [
        "Act as a code reviewer and analyze: x = x + 1",
        "Act as a technical writer and document this function: def add(a, b)",
    ]

    for task in tasks:
        print(f"Task: {task}")
        async for message in query(prompt=task, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Response: {block.text[:200]}...")
                        break
        print()


async def main():
    """Run orchestration examples."""
    await multi_agent_example()
    await simple_delegation()


if __name__ == "__main__":
    asyncio.run(main())
