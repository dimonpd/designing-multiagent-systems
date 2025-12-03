"""
Sequential Workflow Example - LangGraph

Equivalent to: examples/workflows/sequential.py (PicoAgents)

This example demonstrates a sequential workflow where nodes process
a task one after another, passing state through the graph.

In PicoAgents, we use Workflow.chain() with FunctionSteps.
In LangGraph, we use StateGraph with sequential edges.

Run: python examples/frameworks/langgraph/workflows/sequential.py

Prerequisites:
    - pip install langgraph langchain-openai python-dotenv
    - Set OPENAI_API_KEY environment variable
"""

from pathlib import Path
from typing import TypedDict

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


# Define the state schema
class WorkflowState(TypedDict):
    """State passed through the workflow."""

    task: str
    research_output: str
    written_content: str
    final_output: str


def researcher(state: WorkflowState) -> WorkflowState:
    """Research node - gathers facts about the topic."""
    llm = get_llm(temperature=0)

    prompt = (
        f"You are a researcher. Provide 2-3 key facts and data points "
        f"about the following topic. Be concise and factual.\n\n"
        f"Topic: {state['task']}"
    )

    response = llm.invoke(prompt)
    return {"research_output": response.content}


def writer(state: WorkflowState) -> WorkflowState:
    """Writer node - creates engaging content from research."""
    llm = get_llm(temperature=0.7)

    prompt = (
        f"You are a writer. Based on the research below, write a short, "
        f"engaging paragraph (2-3 sentences). Make it accessible.\n\n"
        f"Research: {state['research_output']}"
    )

    response = llm.invoke(prompt)
    return {"written_content": response.content}


def editor(state: WorkflowState) -> WorkflowState:
    """Editor node - polishes and finalizes content."""
    llm = get_llm(temperature=0)

    prompt = (
        f"You are an editor. Review the content below and make minor "
        f"improvements for clarity and flow. Output the final polished version.\n\n"
        f"Content: {state['written_content']}"
    )

    response = llm.invoke(prompt)
    return {"final_output": response.content}


def main():
    """Run a sequential workflow with researcher -> writer -> editor."""
    print("=== Sequential Workflow Example (LangGraph) ===\n")

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.add_node("editor", editor)

    # Define sequential edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    task = "Write about the benefits of electric vehicles"
    print(f"Task: {task}\n")
    print("=== Workflow Execution ===\n")

    # Initial state
    initial_state = {
        "task": task,
        "research_output": "",
        "written_content": "",
        "final_output": "",
    }

    # Stream through the workflow to see each step
    for step in app.stream(initial_state):
        # Each step returns the node name and its output
        for node_name, output in step.items():
            print(f"--- {node_name} ---")
            # Print the relevant output for this node
            if "research_output" in output and output["research_output"]:
                print(output["research_output"])
            elif "written_content" in output and output["written_content"]:
                print(output["written_content"])
            elif "final_output" in output and output["final_output"]:
                print(output["final_output"])
            print()

    print("=== Workflow Complete ===")


if __name__ == "__main__":
    main()
