"""A2A client tools for the Partner Team's OpenAI Agents SDK agent (`openai_agent.py`)."""

import os
from agents import function_tool
from .openai_utils import AGENT_CARD_WELL_KNOWN_PATH
from .openai_utils import send_a2a_request

ADK_A2A_SERVER_URL = os.getenv("ADK_A2A_SERVER_URL", "http://localhost:8003")


@function_tool
async def call_core_analytics_a2a_agent(request: str) -> str:
  """Delegates a workforce or demographic analytics request to the Core Analytics Team's Google ADK A2A Server (`http://localhost:8003`).

  Fetches `http://localhost:8003/.well-known/agent-card.json` and drives the
  stateful A2A Task Lifecycle (`message/stream` SSE: `submitted` -> `working` ->
  `input-required` human-in-the-loop `[Y/n]` confirmation gates -> `completed` +
  `tasks/get` verification).

  Args:
    request: The natural-language analytics question or command to delegate to
      the Core Analytics Team's ADK A2A Server.

  Returns:
    The completed analytics report returned by the Google ADK A2A Server.
  """
  # Build the standard A2A Agent Card discovery URL for `core_analytics_agent` (:8003).
  card_url = f"{ADK_A2A_SERVER_URL.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}"
  return await send_a2a_request(card_url, request)


@function_tool
async def contact_discovered_a2a_agent(
    agent_url: str,
    request: str,
) -> str:
  """Connects to any A2A server URL via its `/.well-known/agent-card.json` and delegates a task.

  Args:
    agent_url: Base URL of the target A2A server (e.g., `http://localhost:8003`).
    request: The natural-language question or command to send to the remote A2A server.

  Returns:
    The response returned by the remote A2A server.
  """
  # Resolve the remote A2A server's Agent Card URL and delegate the request.
  card_url = f"{agent_url.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}"
  return await send_a2a_request(card_url, request)
