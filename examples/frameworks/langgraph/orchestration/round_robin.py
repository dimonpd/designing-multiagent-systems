"""
Round-Robin Orchestration Example - LangGraph

Equivalent to: examples/orchestration/round-robin.py (PicoAgents)

This example demonstrates a cyclic conversation between a poet and
a critic. The poet writes haikus, and the critic provides feedback
until the haiku is approved.

In PicoAgents, we use RoundRobinOrchestrator.
In LangGraph, we use a StateGraph with conditional routing.

Run: python examples/frameworks/langgraph/orchestration/round_robin.py

Prerequisites:
    - pip install langgraph langchain-openai python-dotenv
    - Set OPENAI_API_KEY environment variable
"""

from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


def get_llm(temperature: float = 0):
    """Create Azure OpenAI LLM client."""
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        api_version="2024-08-01-preview",
        temperature=temperature,
    )


class ConversationState(TypedDict):
    """State for the poet/critic conversation."""

    topic: str
    current_haiku: str
    feedback: str
    iteration: int
    max_iterations: int
    approved: bool


def poet(state: ConversationState) -> ConversationState:
    """Poet node - writes or revises haikus."""
    llm = get_llm(temperature=0.8)

    if state["iteration"] == 0:
        # First iteration - write initial haiku
        prompt = (
            f"You are a haiku poet. Write a haiku about: {state['topic']}\n"
            f"Output only the haiku, formatted with each line on a new line."
        )
    else:
        # Subsequent iterations - revise based on feedback
        prompt = (
            f"You are a haiku poet. Revise your haiku based on the feedback.\n\n"
            f"Original haiku:\n{state['current_haiku']}\n\n"
            f"Feedback: {state['feedback']}\n\n"
            f"Output only the revised haiku, formatted with each line on a new line."
        )

    response = llm.invoke(prompt)
    return {"current_haiku": response.content, "iteration": state["iteration"] + 1}


def critic(state: ConversationState) -> ConversationState:
    """Critic node - reviews haikus and decides if approved."""
    llm = get_llm(temperature=0)

    prompt = (
        f"You are a haiku critic. Review the following haiku:\n\n"
        f"{state['current_haiku']}\n\n"
        f"Provide 2-3 specific, actionable suggestions for improvement. "
        f"Focus on imagery, syllable count (5-7-5), seasonal words, "
        f"or emotional impact.\n\n"
        f"If the haiku is excellent and needs no changes, respond with "
        f"exactly: APPROVED\n"
        f"Otherwise, provide constructive feedback."
    )

    response = llm.invoke(prompt)
    content = response.content

    approved = "APPROVED" in content.upper()
    return {"feedback": content, "approved": approved}


def should_continue(state: ConversationState) -> Literal["poet", "end"]:
    """Decide whether to continue the loop or end."""
    if state["approved"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "poet"


def main():
    """Demonstrate round-robin poet/critic collaboration."""
    print("=== Round-Robin Orchestration Example (LangGraph) ===\n")

    # Create the workflow graph
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("poet", poet)
    workflow.add_node("critic", critic)

    # Define edges
    workflow.set_entry_point("poet")
    workflow.add_edge("poet", "critic")

    # Conditional edge - continue or end
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "poet": "poet",
            "end": END,
        },
    )

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    topic = "cherry blossoms in spring"
    print(f"Topic: {topic}")
    print("Poet and Critic collaboration:\n")
    print("=" * 50)

    # Initial state
    initial_state = {
        "topic": topic,
        "current_haiku": "",
        "feedback": "",
        "iteration": 0,
        "max_iterations": 4,
        "approved": False,
    }

    # Stream through the workflow
    current_iteration = 0
    for step in app.stream(initial_state):
        for node_name, output in step.items():
            if node_name == "poet":
                current_iteration = output.get("iteration", current_iteration)
                print(f"\n--- Round {current_iteration} ---")
                print(f"\n[poet]")
                print(output.get("current_haiku", ""))
            elif node_name == "critic":
                print(f"\n[critic]")
                print(output.get("feedback", ""))

                if output.get("approved"):
                    print("\n" + "=" * 50)
                    print("Haiku approved!")

    # Check final state
    final_state = app.invoke(initial_state)
    if not final_state["approved"]:
        print("\n" + "=" * 50)
        print(f"Max iterations ({final_state['max_iterations']}) reached.")

    print("=== Loop Complete ===")


if __name__ == "__main__":
    main()
