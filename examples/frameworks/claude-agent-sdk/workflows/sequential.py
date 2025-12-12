"""
Sequential Workflow Example - Claude Agent SDK

Equivalent to: examples/workflows/sequential_workflow.py (PicoAgents)

This example demonstrates sequential task execution using the Claude Agent SDK.
Unlike explicit workflow patterns in PicoAgents, Claude Agent SDK handles
task sequencing through multi-turn conversations with the ClaudeSDKClient.

For more complex workflows, you can chain multiple query() calls or use
the client's session management to maintain state between steps.

Run: python examples/frameworks/claude-agent-sdk/workflows/sequential.py

Prerequisites:
    - pip install claude-agent-sdk
    - Claude Code CLI installed (https://docs.anthropic.com/en/docs/claude-code)
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


# Workflow state to pass between steps
workflow_state: dict[str, Any] = {}


# Define tools for each workflow step
@tool(
    name="research_topic",
    description="Research and gather information about a topic",
    input_schema={"topic": str},
)
async def research_topic(args: dict[str, Any]) -> dict[str, Any]:
    """Step 1: Research a topic."""
    topic = args["topic"]
    # Simulated research results
    research = {
        "ai agents": {
            "key_points": [
                "AI agents are autonomous systems",
                "They can use tools and make decisions",
                "Multi-agent systems enable collaboration",
            ],
            "sources": ["arxiv.org", "research papers", "documentation"],
        },
        "climate change": {
            "key_points": [
                "Global temperatures rising",
                "Impact on ecosystems",
                "Renewable energy solutions",
            ],
            "sources": ["IPCC reports", "NASA data", "scientific journals"],
        },
    }
    result = research.get(topic.lower(), {
        "key_points": [f"Information about {topic}"],
        "sources": ["general sources"],
    })
    workflow_state["research"] = result
    return {
        "content": [{
            "type": "text",
            "text": f"Research completed for '{topic}': "
                    f"{len(result['key_points'])} key points found"
        }],
        "is_error": False,
    }


@tool(
    name="create_outline",
    description="Create an outline based on research",
    input_schema={"title": str},
)
async def create_outline(args: dict[str, Any]) -> dict[str, Any]:
    """Step 2: Create an outline from research."""
    title = args["title"]
    research = workflow_state.get("research", {})
    key_points = research.get("key_points", [])

    outline = {
        "title": title,
        "sections": [
            {"heading": "Introduction", "content": "Overview of the topic"},
        ] + [
            {"heading": f"Section {i+1}", "content": point}
            for i, point in enumerate(key_points)
        ] + [
            {"heading": "Conclusion", "content": "Summary and takeaways"},
        ],
    }
    workflow_state["outline"] = outline
    return {
        "content": [{
            "type": "text",
            "text": f"Outline created: '{title}' with "
                    f"{len(outline['sections'])} sections"
        }],
        "is_error": False,
    }


@tool(
    name="write_draft",
    description="Write a draft based on the outline",
    input_schema={"style": str},
)
async def write_draft(args: dict[str, Any]) -> dict[str, Any]:
    """Step 3: Write a draft from the outline."""
    style = args["style"]
    outline = workflow_state.get("outline", {})

    # Simulated draft generation
    sections_text = []
    for section in outline.get("sections", []):
        sections_text.append(
            f"## {section['heading']}\n{section['content']}"
        )

    draft = {
        "title": outline.get("title", "Untitled"),
        "style": style,
        "content": "\n\n".join(sections_text),
        "word_count": len(" ".join(sections_text).split()),
    }
    workflow_state["draft"] = draft
    return {
        "content": [{
            "type": "text",
            "text": f"Draft written in {style} style: "
                    f"{draft['word_count']} words"
        }],
        "is_error": False,
    }


@tool(
    name="review_draft",
    description="Review and provide feedback on the draft",
    input_schema={},
)
async def review_draft(args: dict[str, Any]) -> dict[str, Any]:
    """Step 4: Review the draft."""
    draft = workflow_state.get("draft", {})

    review = {
        "status": "approved",
        "feedback": [
            "Good structure and flow",
            "Clear introduction",
            "Strong conclusion",
        ],
        "suggestions": [
            "Consider adding more examples",
            "Could expand the technical details",
        ],
    }
    workflow_state["review"] = review
    return {
        "content": [{
            "type": "text",
            "text": f"Review complete: {review['status']}. "
                    f"{len(review['feedback'])} positive points, "
                    f"{len(review['suggestions'])} suggestions."
        }],
        "is_error": False,
    }


# Create MCP server with workflow tools
workflow_server = create_sdk_mcp_server(
    name="workflow",
    version="1.0.0",
    tools=[research_topic, create_outline, write_draft, review_draft],
)


async def run_sequential_workflow():
    """Execute a sequential content creation workflow."""
    print("=== Sequential Workflow Example (Claude Agent SDK) ===\n")
    print("Workflow: Research → Outline → Draft → Review\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt=(
            "You are a content creation assistant. Execute workflows "
            "step by step using the available tools. After each step, "
            "confirm completion before moving to the next step. "
            "Be concise in your responses."
        ),
        mcp_servers={"workflow": workflow_server},
        allowed_tools=[
            "mcp__workflow__research_topic",
            "mcp__workflow__create_outline",
            "mcp__workflow__write_draft",
            "mcp__workflow__review_draft",
        ],
        permission_mode="bypassPermissions",
        max_turns=20,
    )

    # Use ClaudeSDKClient for multi-turn workflow execution
    async with ClaudeSDKClient(options=options) as client:
        await client.connect()

        # Define workflow steps
        workflow_steps = [
            "Step 1: Research the topic 'AI agents'",
            "Step 2: Create an outline titled 'Understanding AI Agents'",
            "Step 3: Write a draft in 'professional' style",
            "Step 4: Review the draft and provide feedback",
        ]

        for i, step in enumerate(workflow_steps, 1):
            print(f"\n{'='*50}")
            print(f"Executing: {step}")
            print('='*50)

            await client.query(step)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\nAssistant: {block.text}")
                elif isinstance(message, ResultMessage):
                    print(f"\n[Step {i} complete]")

        # Summary
        print("\n" + "="*50)
        print("WORKFLOW SUMMARY")
        print("="*50)
        if "draft" in workflow_state:
            print(f"Title: {workflow_state['draft'].get('title')}")
            print(f"Style: {workflow_state['draft'].get('style')}")
            print(f"Words: {workflow_state['draft'].get('word_count')}")
        if "review" in workflow_state:
            print(f"Status: {workflow_state['review'].get('status')}")


async def run_chained_queries():
    """Alternative: Chain independent queries for simpler workflows."""
    print("\n=== Chained Queries Pattern ===\n")
    print("For simpler workflows, you can chain independent query() calls.\n")

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a helpful assistant. Be very concise.",
        permission_mode="bypassPermissions",
        max_turns=3,
    )

    # Define a simple sequential workflow
    steps = [
        "What are the three main benefits of AI agents?",
        "Based on those benefits, what industries would benefit most?",
        "Summarize in one sentence.",
    ]

    from claude_agent_sdk import query

    context = ""
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")

        # Include previous context in the prompt
        full_prompt = f"{context}\n\n{step}" if context else step

        async for message in query(prompt=full_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response = block.text
                        print(f"Response: {response}\n")
                        context += f"\nQ: {step}\nA: {response}"
                        break


async def main():
    """Run workflow examples."""
    await run_sequential_workflow()
    await run_chained_queries()


if __name__ == "__main__":
    asyncio.run(main())
