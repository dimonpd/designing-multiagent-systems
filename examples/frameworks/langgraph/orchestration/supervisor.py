"""
Supervisor Orchestration Example - LangGraph

Equivalent to: examples/orchestration/supervisor.py (PicoAgents)

This example demonstrates a supervisor pattern where a central agent
delegates tasks to specialized worker agents.

In PicoAgents, we use SupervisorOrchestrator.
In LangGraph, we use a StateGraph with a supervisor node that routes.

Run: python examples/frameworks/langgraph/orchestration/supervisor.py

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


class SupervisorState(TypedDict):
    """State for supervisor-controlled workflow."""

    task: str
    next_worker: str
    research_output: str
    analysis_output: str
    final_response: str
    completed: bool


def supervisor(state: SupervisorState) -> SupervisorState:
    """Supervisor decides which worker to delegate to."""
    llm = get_llm(temperature=0)

    # Check what work has been done
    has_research = bool(state.get("research_output"))
    has_analysis = bool(state.get("analysis_output"))

    prompt = f"""You are a supervisor managing a team of workers.

Task: {state['task']}

Current progress:
- Research completed: {has_research}
- Analysis completed: {has_analysis}

Available workers:
- researcher: Gathers information and facts
- analyst: Analyzes data and provides insights
- FINISH: Complete the task and provide final response

Based on the task and current progress, which worker should handle the next step?
Respond with exactly one of: researcher, analyst, or FINISH"""

    response = llm.invoke(prompt)
    next_worker = response.content.strip().lower()

    # Normalize the response
    if "researcher" in next_worker:
        next_worker = "researcher"
    elif "analyst" in next_worker:
        next_worker = "analyst"
    else:
        next_worker = "FINISH"

    return {"next_worker": next_worker}


def researcher(state: SupervisorState) -> SupervisorState:
    """Research worker - gathers information."""
    llm = get_llm(temperature=0)

    prompt = (
        f"You are a researcher. Gather key facts and information about:\n"
        f"{state['task']}\n\n"
        f"Provide 3-4 relevant facts or data points."
    )

    response = llm.invoke(prompt)
    return {"research_output": response.content}


def analyst(state: SupervisorState) -> SupervisorState:
    """Analyst worker - analyzes information."""
    llm = get_llm(temperature=0)

    research = state.get("research_output", "No research available")

    prompt = (
        f"You are an analyst. Based on the following research, provide "
        f"insights and recommendations:\n\n"
        f"Research: {research}\n\n"
        f"Provide 2-3 key insights."
    )

    response = llm.invoke(prompt)
    return {"analysis_output": response.content}


def finisher(state: SupervisorState) -> SupervisorState:
    """Finisher - creates the final response."""
    llm = get_llm(temperature=0)

    research = state.get("research_output", "")
    analysis = state.get("analysis_output", "")

    prompt = (
        f"Create a brief final summary for the task: {state['task']}\n\n"
        f"Research findings:\n{research}\n\n"
        f"Analysis:\n{analysis}\n\n"
        f"Provide a concise 2-3 sentence summary."
    )

    response = llm.invoke(prompt)
    return {"final_response": response.content, "completed": True}


def route_to_worker(
    state: SupervisorState,
) -> Literal["researcher", "analyst", "finisher"]:
    """Route based on supervisor's decision."""
    next_worker = state.get("next_worker", "")
    if next_worker == "researcher":
        return "researcher"
    elif next_worker == "analyst":
        return "analyst"
    else:
        return "finisher"


def main():
    """Demonstrate supervisor-controlled delegation."""
    print("=== Supervisor Orchestration Example (LangGraph) ===\n")

    # Create the workflow graph
    workflow = StateGraph(SupervisorState)

    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("finisher", finisher)

    # Define edges
    workflow.set_entry_point("supervisor")

    # Supervisor routes to workers
    workflow.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "finisher": "finisher",
        },
    )

    # Workers return to supervisor (except finisher)
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("finisher", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    task = "Analyze the impact of remote work on productivity"
    print(f"Task: {task}")
    print("Supervisor delegating to workers:\n")
    print("=" * 50)

    # Initial state
    initial_state = {
        "task": task,
        "next_worker": "",
        "research_output": "",
        "analysis_output": "",
        "final_response": "",
        "completed": False,
    }

    # Stream through the workflow
    for step in app.stream(initial_state):
        for node_name, output in step.items():
            if node_name == "supervisor":
                next_w = output.get("next_worker", "")
                print(f"\n[supervisor] Delegating to: {next_w}")
            elif node_name == "researcher":
                print(f"\n[researcher]")
                print(output.get("research_output", "")[:200] + "...")
            elif node_name == "analyst":
                print(f"\n[analyst]")
                print(output.get("analysis_output", "")[:200] + "...")
            elif node_name == "finisher":
                print(f"\n[finisher]")
                print(output.get("final_response", ""))

    print("\n" + "=" * 50)
    print("=== Workflow Complete ===")


if __name__ == "__main__":
    main()
