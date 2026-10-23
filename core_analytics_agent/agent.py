"""Main Google ADK Agent definition for `core_analytics_agent`."""

import logging
import os
import warnings

logging.disable(level=logging.WARNING)
warnings.filterwarnings("ignore")

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool

from .instructions import agent_instructions
from .tools import analyze_name_in_bigquery
from .tools import contact_discovered_a2a_agent
from .tools import query_local_mcp_kb
from .tools import write_local_mcp_kb
from .utils import GATEWAY_URL
from .utils import get_gateway_httpx_client
from .utils import log_after_tool
from .utils import log_before_tool

# --- A2A Sub-Agent: Gateway Service Registry & OIDC-to-JWT Token Broker ---
# Connects to the remote Gateway Agent via its `/.well-known/agent-card.json`
# and attaches out-of-band Google Cloud OIDC credentials via `get_gateway_httpx_client()`.
gateway_agent = RemoteA2aAgent(
    name="gateway_agent",
    description=(
        "Remote Gateway Agent for service discovery and OIDC-to-JWT token"
        " brokering."
    ),
    agent_card=f"{GATEWAY_URL}{AGENT_CARD_WELL_KNOWN_PATH}",
    httpx_client=get_gateway_httpx_client(),
)

# --- Main Orchestrator Agent ---
# Combines local tools (BigQuery SQL + Local MCP read/write) with A2A discovery
# (`gateway_agent`) and dynamic A2A invocation (`contact_discovered_a2a_agent`).
root_agent = Agent(
    name="core_analytics_agent",
    model=os.getenv("MODEL_NAME", "gemini-3.6-flash"),
    description=(
        "US and Corporate Name Analytics Orchestrator powered by BigQuery,"
        " Local MCP, and the Agent2Agent (A2A) protocol on Gemini Agent Platform."
    ),
    instruction=agent_instructions,
    tools=[
        analyze_name_in_bigquery,
        query_local_mcp_kb,
        write_local_mcp_kb,
        AgentTool(agent=gateway_agent),
        contact_discovered_a2a_agent,
    ],
    before_tool_callback=log_before_tool,
    after_tool_callback=log_after_tool,
)
