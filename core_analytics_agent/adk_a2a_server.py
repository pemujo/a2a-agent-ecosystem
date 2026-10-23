"""Exposes `core_analytics_agent` as a standards-compliant A2A Server via `to_a2a()`.

Wraps `root_agent` in a Starlette ASGI application that automatically serves:
  - `GET /.well-known/agent-card.json` (machine-readable Agent Card discovery)
  - `POST /` (JSON-RPC 2.0 `message/send`, `message/stream` SSE, `tasks/get`, and `tasks/cancel`)
"""

import os
from core_analytics_agent.agent import root_agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Automatically generates /.well-known/agent-card.json and mounts JSON-RPC 2.0 endpoints
a2a_app = to_a2a(root_agent, port=int(os.getenv("PORT", "8003")))
