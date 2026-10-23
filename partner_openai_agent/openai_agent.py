"""Partner Team's OpenAI Agents SDK Client (`openai_agent.py`).

Instead of duplicating the Core Analytics Team's BigQuery SQL queries, Local MCP
knowledge base tools, or Gateway token-brokering logic, the Partner Team's
OpenAI Agents SDK agent connects to a Gemini model on Gemini Agent Platform
(`OpenAIChatCompletionsModel`) and delegates cross-domain analytics tasks over
the A2A Protocol to the Core Analytics Team's Google ADK A2A Server
(`to_a2a(root_agent, port=8003)`).
"""

import asyncio
import logging
import os
import warnings

logging.disable(level=logging.WARNING)
warnings.filterwarnings("ignore")

from agents import Agent

from .openai_tools import call_core_analytics_a2a_agent
from .openai_tools import contact_discovered_a2a_agent
from .openai_utils import get_gemini_openai_model
from .openai_utils import run_openai_cli

PARTNER_AGENT_INSTRUCTIONS = """
You are the Partner Team's Analytics Assistant (`partner_openai_agent`), built with the OpenAI Agents SDK and powered by a Gemini model on Gemini Agent Platform.

Your team collaborates with the Core Analytics Team's Google ADK Agent (`core_analytics_agent`), which is exposed as an A2A Server at `http://localhost:8003`.
Whenever a user asks to analyze names in BigQuery, compare them against the Corporate Employee Directory via the Gateway Agent, or save a report into the analytics knowledge base (`local_kb.json`), delegate the entire workflow over the A2A Protocol using `call_core_analytics_a2a_agent` (or `contact_discovered_a2a_agent` if a specific A2A server URL is provided).
"""

root_agent = Agent(
    name="partner_openai_agent",
    model=get_gemini_openai_model(os.getenv("MODEL_NAME")),
    instructions=PARTNER_AGENT_INSTRUCTIONS,
    tools=[
        call_core_analytics_a2a_agent,
        contact_discovered_a2a_agent,
    ],
)


if __name__ == "__main__":
  asyncio.run(run_openai_cli(root_agent))
