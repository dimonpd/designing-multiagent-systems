"""
Middleware/Hooks Example - Claude Agent SDK

Equivalent to: examples/agents/middleware.py (PicoAgents)

This example demonstrates hook patterns using Claude Agent SDK.
Claude Agent SDK uses hooks configured via ClaudeAgentOptions:
- PreToolUse: Before tool execution
- PostToolUse: After tool execution
- UserPromptSubmit: When user submits a prompt
- SessionStart: When a session begins

Hooks can return:
- permissionDecision: "allow" | "deny" to control tool execution
- continue_: False to stop execution
- additionalContext: To inject context

Run: python examples/frameworks/claude-agent-sdk/agents/middleware.py

Prerequisites:
    - pip install claude-agent-sdk
    - Set ANTHROPIC_API_KEY environment variable
"""

import asyncio
import logging
import re
import time
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    Message,
    ResultMessage,
    TextBlock,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def display_message(msg: Message) -> None:
    """Display agent messages."""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"Claude: {block.text}")
    elif isinstance(msg, ResultMessage):
        print("[Session ended]")


# =============================================================================
# Hook Functions (Middleware Equivalent)
# =============================================================================


async def logging_pre_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Logging Hook - Logs tool execution start.

    Equivalent to PicoAgents LoggingMiddleware.process_request()
    """
    tool_name = input_data.get("tool_name", "unknown")
    print(f"  [Tool] Starting: {tool_name}")

    # Store start time in context for duration tracking
    # Note: Hook context is per-invocation, so we use a simple approach
    logging_pre_tool_hook._start_times = getattr(
        logging_pre_tool_hook, '_start_times', {}
    )
    if tool_use_id:
        logging_pre_tool_hook._start_times[tool_use_id] = time.time()

    return {}  # Continue normal execution


async def logging_post_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Logging Hook - Logs tool execution completion.

    Equivalent to PicoAgents LoggingMiddleware.process_response()
    """
    tool_name = input_data.get("tool_name", "unknown")
    elapsed = 0

    if tool_use_id and hasattr(logging_pre_tool_hook, '_start_times'):
        start_time = logging_pre_tool_hook._start_times.pop(tool_use_id, None)
        if start_time:
            elapsed = time.time() - start_time

    print(f"  [Tool] Completed: {tool_name} ({elapsed:.3f}s)")
    return {}


async def security_pre_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Security Hook - Blocks dangerous commands.

    Equivalent to PicoAgents SecurityMiddleware.
    Uses permissionDecision to deny execution.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Block dangerous bash commands
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        dangerous_patterns = [
            r"rm\s+-rf",
            r"sudo\s+",
            r"chmod\s+777",
            r">\s*/dev/",
            r"mkfs\.",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Blocked dangerous command: {command}")
                return {
                    "reason": f"Command blocked by security policy: {pattern}",
                    "systemMessage": "🛡️ Security: Dangerous command blocked",
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "Security policy violation",
                    },
                }

    # Block writes to sensitive files
    if tool_name == "Write":
        file_path = tool_input.get("file_path", "").lower()
        sensitive_paths = [
            "/etc/",
            "/system/",
            ".env",
            "credentials",
            "secrets",
            "password",
        ]

        for sensitive in sensitive_paths:
            if sensitive in file_path:
                logger.warning(f"Blocked write to: {file_path}")
                return {
                    "reason": f"Cannot write to sensitive path: {sensitive}",
                    "systemMessage": "🛡️ Security: Write to sensitive path blocked",
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "Protected file path",
                    },
                }

    return {}  # Allow


async def rate_limit_pre_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Rate Limit Hook - Limits tool calls.

    Equivalent to PicoAgents RateLimitMiddleware.
    """
    # Track calls using function attribute
    if not hasattr(rate_limit_pre_tool_hook, '_call_times'):
        rate_limit_pre_tool_hook._call_times = []

    current_time = time.time()
    max_calls_per_minute = 10

    # Clean old entries
    rate_limit_pre_tool_hook._call_times = [
        t for t in rate_limit_pre_tool_hook._call_times
        if current_time - t < 60
    ]

    if len(rate_limit_pre_tool_hook._call_times) >= max_calls_per_minute:
        logger.warning("Rate limit exceeded")
        return {
            "reason": "Too many tool calls in the last minute",
            "systemMessage": "⏱️ Rate limit exceeded",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Rate limit exceeded",
            },
        }

    rate_limit_pre_tool_hook._call_times.append(current_time)
    return {}


async def pii_post_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    PII Redaction Hook - Adds warning when PII detected in output.

    Equivalent to PicoAgents PIIRedactionMiddleware.
    Note: Claude SDK hooks can add context but not modify tool output directly.
    """
    tool_response = input_data.get("tool_response", "")

    # Check for PII patterns
    pii_patterns = {
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': 'phone number',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': 'email',
        r'\b\d{3}[-]?\d{2}[-]?\d{4}\b': 'SSN',
    }

    detected_pii = []
    for pattern, pii_type in pii_patterns.items():
        if re.search(pattern, str(tool_response)):
            detected_pii.append(pii_type)

    if detected_pii:
        logger.warning(f"PII detected: {detected_pii}")
        return {
            "systemMessage": f"⚠️ Warning: Output may contain sensitive data ({', '.join(detected_pii)})",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "Note: The tool output contains sensitive information. Handle with care.",
            },
        }

    return {}


async def stop_on_error_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Error Handler Hook - Stops execution on critical errors.

    Uses continue_: False to halt the session.
    """
    tool_response = input_data.get("tool_response", "")

    if "critical" in str(tool_response).lower():
        logger.error("Critical error detected - stopping execution")
        return {
            "continue_": False,
            "stopReason": "Critical error detected - execution halted",
            "systemMessage": "🛑 Execution stopped due to critical error",
        }

    return {"continue_": True}


async def context_injection_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """
    Context Injection Hook - Adds custom context at session start.

    Demonstrates UserPromptSubmit/SessionStart hook patterns.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "Important context: "
                "1. Always verify information before presenting it. "
                "2. If uncertain, ask clarifying questions. "
                "3. Prioritize user safety in all recommendations."
            ),
        }
    }


# =============================================================================
# Demo Functions
# =============================================================================


async def demo_logging_hooks():
    """Demonstrate logging hooks."""
    print("\n" + "=" * 60)
    print("DEMO: Logging Hooks")
    print("=" * 60)

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[logging_pre_tool_hook]),
            ],
            "PostToolUse": [
                HookMatcher(matcher="Bash", hooks=[logging_post_tool_hook]),
            ],
        },
    )

    async with ClaudeSDKClient(options=options) as client:
        print("\nUser: Echo 'Hello from hooks!'")
        await client.query("Run: echo 'Hello from hooks!'")

        async for msg in client.receive_response():
            display_message(msg)


async def demo_security_hooks():
    """Demonstrate security hooks blocking dangerous commands."""
    print("\n" + "=" * 60)
    print("DEMO: Security Hooks")
    print("=" * 60)

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Write"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[security_pre_tool_hook]),
                HookMatcher(matcher="Write", hooks=[security_pre_tool_hook]),
            ],
        },
    )

    async with ClaudeSDKClient(options=options) as client:
        # Test safe command
        print("\nTest 1: Safe command")
        print("User: List files in current directory")
        await client.query("Run: ls -la")

        async for msg in client.receive_response():
            display_message(msg)

        print("\n" + "-" * 40)

        # Test dangerous command (should be blocked)
        print("\nTest 2: Dangerous command (should be blocked)")
        print("User: Remove all files recursively")
        await client.query("Run: rm -rf /tmp/test")

        async for msg in client.receive_response():
            display_message(msg)


async def demo_rate_limit_hooks():
    """Demonstrate rate limiting hooks."""
    print("\n" + "=" * 60)
    print("DEMO: Rate Limit Hooks (simulated)")
    print("=" * 60)

    # Reset rate limit tracking
    if hasattr(rate_limit_pre_tool_hook, '_call_times'):
        rate_limit_pre_tool_hook._call_times = []

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[rate_limit_pre_tool_hook]),
            ],
        },
    )

    async with ClaudeSDKClient(options=options) as client:
        for i in range(3):
            print(f"\nRequest {i+1}:")
            await client.query(f"Run: echo 'Request {i+1}'")

            async for msg in client.receive_response():
                display_message(msg)


async def demo_context_injection():
    """Demonstrate context injection at session start."""
    print("\n" + "=" * 60)
    print("DEMO: Context Injection Hook")
    print("=" * 60)

    options = ClaudeAgentOptions(
        hooks={
            "UserPromptSubmit": [
                HookMatcher(matcher=None, hooks=[context_injection_hook]),
            ],
        },
    )

    async with ClaudeSDKClient(options=options) as client:
        print("\nUser: What guidelines should you follow?")
        await client.query("What guidelines should you follow?")

        async for msg in client.receive_response():
            display_message(msg)


async def main():
    """Run all hook demonstrations."""
    print("\n" + "=" * 70)
    print("CLAUDE AGENT SDK - HOOKS/MIDDLEWARE EXAMPLES")
    print("=" * 70)
    print("""
Hook Patterns in Claude Agent SDK:
- PreToolUse: Intercept before tool execution (can deny)
- PostToolUse: Process after tool execution (can add context)
- UserPromptSubmit: When user submits prompt (can inject context)
- SessionStart: When session begins

Hook Output Fields:
- permissionDecision: "allow" | "deny" - Control tool execution
- permissionDecisionReason: Explanation for decision
- continue_: False - Stop session execution
- stopReason: Explanation for stopping
- additionalContext: Inject context for Claude
- systemMessage: Display message to user
    """)

    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return

    try:
        await demo_logging_hooks()
        await demo_security_hooks()
        await demo_rate_limit_hooks()
        await demo_context_injection()

        print("\n" + "=" * 70)
        print("All hook examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
