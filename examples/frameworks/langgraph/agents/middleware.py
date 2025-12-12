"""
Middleware Example - LangChain

Equivalent to: examples/agents/middleware.py (PicoAgents)

This example demonstrates middleware patterns using LangChain's create_agent
with the AgentMiddleware system. LangChain v1 provides a comprehensive middleware
architecture with hooks for:
- before_agent / after_agent: Wrap entire agent execution (once)
- before_model / after_model: Wrap each LLM call
- wrap_model_call: Intercept model requests (most powerful - can block calls)
- wrap_tool_call: Intercept tool execution

Run: python examples/frameworks/langgraph/agents/middleware.py

Prerequisites:
    - pip install langchain langchain-openai python-dotenv
    - Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables
"""

import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.runtime import Runtime

# Load environment variables from picoagents .env
env_path = Path(__file__).parents[4] / "picoagents" / ".env"
load_dotenv(env_path)


def get_llm():
    """Create Azure OpenAI LLM client."""
    return AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        api_version="2024-08-01-preview",
        temperature=0,
    )


# =============================================================================
# Tools
# =============================================================================


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "new york": "72°F, Partly Cloudy",
        "london": "59°F, Rainy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Overcast",
    }
    return weather_data.get(
        location.lower(), f"Weather data not available for {location}"
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_customer_info(customer_id: str) -> str:
    """Get customer information (simulated with PII for demo)."""
    return f"Customer {customer_id}: John Doe, Phone: 555-123-4567, Email: john@example.com"


# =============================================================================
# Custom Middleware Classes
# =============================================================================


class LoggingMiddleware(AgentMiddleware):
    """
    Logging Middleware - Logs agent and model execution timing.

    Equivalent to PicoAgents LoggingMiddleware.
    Uses before_agent/after_agent and before_model/after_model hooks.
    """

    @property
    def name(self) -> str:
        return "LoggingMiddleware"

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Called once at the start of agent execution."""
        print(f"[Agent] Starting execution...")
        return None

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Called once at the end of agent execution."""
        print(f"[Agent] Execution completed")
        return None

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Called before each LLM call."""
        print(f"  [Model] Calling LLM...")
        return None

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Called after each LLM call."""
        print(f"  [Model] Response received")
        return None


class SecurityMiddleware(AgentMiddleware):
    """
    Security Middleware - Blocks potentially malicious input.

    Uses wrap_model_call to intercept model requests and block them
    before they reach the LLM. This is the most robust approach as it
    prevents the model call entirely rather than just adding a response.
    """

    def __init__(self):
        super().__init__()
        self.blocked_topics = [
            "password",
            "secret",
            "confidential",
            "hack",
            "exploit",
        ]

    @property
    def name(self) -> str:
        return "SecurityMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept model calls and block potentially unsafe requests."""
        messages = request.messages

        # Check the last user message for blocked topics
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = str(msg.content).lower()
                for topic in self.blocked_topics:
                    if topic in content:
                        print(f"  [Security] BLOCKED: Contains '{topic}'")
                        # Return early without calling the model
                        return AIMessage(
                            content=f"I cannot process requests about '{topic}' for security reasons."
                        )
                break

        # If safe, proceed with the model call
        return handler(request)


class RateLimitMiddleware(AgentMiddleware):
    """
    Rate Limit Middleware - Limits model calls per minute.

    Uses wrap_model_call to count and block excessive calls.
    """

    def __init__(self, max_calls_per_minute: int = 10):
        super().__init__()
        self.max_calls = max_calls_per_minute
        self._call_times: list[float] = []

    @property
    def name(self) -> str:
        return "RateLimitMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Check rate limit before allowing model call."""
        current_time = time.time()

        # Clean old entries
        self._call_times = [
            t for t in self._call_times if current_time - t < 60
        ]

        if len(self._call_times) >= self.max_calls:
            print(f"  [RateLimit] BLOCKED: Too many requests")
            return AIMessage(content="Rate limit exceeded. Please try again later.")

        self._call_times.append(current_time)
        return handler(request)


# =============================================================================
# Decorator-based Middleware (Alternative Syntax)
# =============================================================================


@wrap_model_call
def uppercase_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | AIMessage:
    """Middleware that uppercases model responses (demonstration only)."""
    response = handler(request)
    if response.result:
        ai_msg = response.result[0]
        if isinstance(ai_msg, AIMessage) and isinstance(ai_msg.content, str):
            return ModelResponse(
                result=[AIMessage(content=ai_msg.content.upper())],
                structured_response=response.structured_response,
            )
    return response


# =============================================================================
# Demo Functions
# =============================================================================


def demo_logging_middleware():
    """Demonstrate logging middleware."""
    print("\n" + "=" * 60)
    print("DEMO: Logging Middleware")
    print("=" * 60)

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[get_weather, calculate],
        system_prompt="You are a helpful assistant.",
        middleware=[LoggingMiddleware()],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What's the weather in Tokyo?")],
        }
    )

    print(f"\nResponse: {result['messages'][-1].content}")


def demo_security_middleware():
    """Demonstrate security middleware blocking unsafe input."""
    print("\n" + "=" * 60)
    print("DEMO: Security Middleware (wrap_model_call)")
    print("=" * 60)

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[calculate],
        system_prompt="You are a helpful assistant.",
        middleware=[SecurityMiddleware(), LoggingMiddleware()],
    )

    # Test normal input
    print("\nTest 1: Normal input")
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What is 2 + 2?")],
        }
    )
    print(f"Response: {result['messages'][-1].content}")

    # Test blocked input (contains 'password')
    print("\nTest 2: Blocked input (should be intercepted)")
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What is a good password?")],
        }
    )
    print(f"Response: {result['messages'][-1].content}")


def demo_rate_limit_middleware():
    """Demonstrate rate limiting middleware."""
    print("\n" + "=" * 60)
    print("DEMO: Rate Limit Middleware")
    print("=" * 60)

    llm = get_llm()

    # Create with a low rate limit for demo (2 calls per minute)
    rate_limiter = RateLimitMiddleware(max_calls_per_minute=2)

    agent = create_agent(
        model=llm,
        tools=[calculate],
        system_prompt="You are a calculator assistant. Answer briefly.",
        middleware=[rate_limiter, LoggingMiddleware()],
    )

    for i in range(4):
        print(f"\nRequest {i+1}:")
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=f"What is {i} + {i}?")],
            }
        )
        response = result["messages"][-1].content
        # Truncate long responses
        if len(response) > 80:
            response = response[:80] + "..."
        print(f"Response: {response}")


def demo_decorator_middleware():
    """Demonstrate decorator-based middleware syntax."""
    print("\n" + "=" * 60)
    print("DEMO: Decorator Middleware (@wrap_model_call)")
    print("=" * 60)

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant. Be brief.",
        middleware=[uppercase_middleware, LoggingMiddleware()],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="Say hello in one word")],
        }
    )

    print(f"\nResponse (should be uppercase): {result['messages'][-1].content}")


def main():
    """Run all middleware demonstrations."""
    print("\n" + "=" * 70)
    print("LANGCHAIN - MIDDLEWARE EXAMPLES")
    print("=" * 70)
    print("""
Middleware Patterns in LangChain v1:

1. Class-based (AgentMiddleware):
   - before_agent / after_agent: Wrap entire agent execution
   - before_model / after_model: Wrap each LLM call
   - wrap_model_call: Intercept and control model requests
   - wrap_tool_call: Intercept tool execution

2. Decorator-based:
   - @wrap_model_call: Create middleware from a function
   - @wrap_tool_call: Create tool interceptor from a function

Key Patterns:
- wrap_model_call can return AIMessage to short-circuit (skip LLM call)
- Return ModelResponse to modify the full response
- Call handler(request) to proceed with the model call
- Middleware runs in order: first middleware is outermost layer
    """)

    import os
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("ERROR: Set AZURE_OPENAI_ENDPOINT environment variable")
        return

    try:
        demo_logging_middleware()
        demo_security_middleware()
        demo_rate_limit_middleware()
        demo_decorator_middleware()

        print("\n" + "=" * 70)
        print("All middleware examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
