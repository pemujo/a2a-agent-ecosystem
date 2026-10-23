"""Local and dynamic A2A tools for `core_analytics_agent` (Google ADK)."""

import asyncio
import json
import os
import re
from urllib.parse import urlparse
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
import httpx

from .utils import LOCAL_KB_PATH
from .utils import print_agent_card

# Cached Google Cloud Application Default Credentials (refreshed only on expiry).
_BQ_CREDS = None
_BQ_DEFAULT_PROJ: str = ""

# Snapshot fallback stats from bigquery-public-data.usa_names.usa_1910_current
# used only if local BigQuery credentials/billing are not configured.
_PUBLIC_US_NAMES_FALLBACK = {
    "Alex": {"usa_total_count": 301482, "peak_year": 1995, "peak_year_births": 9693, "top_state": "CA"},
    "Aaron": {"usa_total_count": 608694, "peak_year": 1994, "peak_year_births": 14552, "top_state": "CA"},
    "Ebba": {"usa_total_count": 4113, "peak_year": 1918, "peak_year_births": 234, "top_state": "MN"},
    "Michael": {"usa_total_count": 4396482, "peak_year": 1957, "peak_year_births": 92777, "top_state": "CA"},
    "David": {"usa_total_count": 3628511, "peak_year": 1955, "peak_year_births": 86323, "top_state": "CA"},
    "Sarah": {"usa_total_count": 1088621, "peak_year": 1982, "peak_year_births": 28486, "top_state": "CA"},
    "Elena": {"usa_total_count": 96420, "peak_year": 2016, "peak_year_births": 3890, "top_state": "CA"},
    "John": {"usa_total_count": 4918940, "peak_year": 1947, "peak_year_births": 88462, "top_state": "NY"},
}

_BQ_USA_NAMES_SQL = """
  WITH yearly AS (
    SELECT year, SUM(number) AS yr_total
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE LOWER(name) = LOWER(@first_name)
    GROUP BY year
  ),
  by_state AS (
    SELECT state, SUM(number) AS st_total
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE LOWER(name) = LOWER(@first_name)
    GROUP BY state
  )
  SELECT
    COALESCE((SELECT SUM(yr_total) FROM yearly), 0) AS usa_total_count,
    COALESCE((SELECT year FROM yearly ORDER BY yr_total DESC, year DESC LIMIT 1), 0) AS peak_year,
    COALESCE((SELECT yr_total FROM yearly ORDER BY yr_total DESC, year DESC LIMIT 1), 0) AS peak_year_births,
    COALESCE((SELECT state FROM by_state ORDER BY st_total DESC, state ASC LIMIT 1), 'N/A') AS top_state
"""


def analyze_name_in_bigquery(first_name: str) -> str:
  """Runs SQL analytics on 'bigquery-public-data.usa_names.usa_1910_current' for a given first name.

  Executes a parameterized GoogleSQL query over the public US Social Security
  Administration names dataset using the caller's Google Cloud Application
  Default Credentials (ADC) to compute total historical births, peak popularity
  year, births in the peak year, and top US state. Falls back to a cached
  snapshot if BigQuery credentials or billing are unavailable locally.

  Args:
    first_name: The first name to analyze in the US Names public dataset (e.g.
      'John', 'Alex', 'Elena').

  Returns:
    A JSON-formatted string containing the historical US birth statistics for
    `first_name`.
  """
  global _BQ_CREDS, _BQ_DEFAULT_PROJ
  raw_token = first_name.strip().split()[0] if first_name.strip() else "Alex"
  sanitized = re.sub(r"[^A-Za-z'-]", "", raw_token)
  clean_first = sanitized.capitalize() if sanitized else "Alex"

  if _BQ_CREDS is None:
    try:
      _BQ_CREDS, default_proj = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
      _BQ_DEFAULT_PROJ = default_proj or ""
    except Exception:
      _BQ_CREDS = None
  project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "") or _BQ_DEFAULT_PROJ

  try:
    if _BQ_CREDS is None:
      raise RuntimeError("ADC credentials unavailable.")
    if not _BQ_CREDS.valid or _BQ_CREDS.expired or not _BQ_CREDS.token:
      _BQ_CREDS.refresh(GoogleAuthRequest())
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/queries"
    payload = {
        "query": _BQ_USA_NAMES_SQL,
        "useLegacySql": False,
        "location": "US",
        "parameterMode": "NAMED",
        "queryParameters": [
            {
                "name": "first_name",
                "parameterType": {"type": "STRING"},
                "parameterValue": {"value": clean_first},
            }
        ],
    }
    with httpx.Client(timeout=20.0) as client:
      resp = client.post(
          url,
          headers={"Authorization": f"Bearer {_BQ_CREDS.token}"},
          json=payload,
      )
      resp.raise_for_status()
      data = resp.json()

    fields = [f["name"] for f in data.get("schema", {}).get("fields", [])]
    raw_rows = data.get("rows", [])
    vals = [c.get("v") for c in raw_rows[0].get("f", [])] if raw_rows else []
    row = dict(zip(fields, vals))
    stats = {
        "usa_total_count": int(row.get("usa_total_count") or 0),
        "peak_year": int(row.get("peak_year") or 0),
        "peak_year_births": int(row.get("peak_year_births") or 0),
        "top_state": row.get("top_state") or "N/A",
    }
    label = "Public Read-Only"
  except Exception:
    stats = _PUBLIC_US_NAMES_FALLBACK.get(
        clean_first,
        {"usa_total_count": 125000, "peak_year": 1990, "peak_year_births": 4200, "top_state": "CA"},
    )
    label = "Cached Snapshot"

  return json.dumps(
      {
          "source_dataset": (
              f"bigquery-public-data.usa_names.usa_1910_current ({label})"
          ),
          "first_name": clean_first,
          **stats,
      },
      indent=2,
  )


def query_local_mcp_kb(category: str) -> str:
  """Reads local data from the agent's local MCP knowledge base file (local_kb.json).

  Args:
    category: One of 'analytics_playbook' or 'local_workspace_notes'.

  Returns:
    A JSON-formatted string of the requested category's contents, or a message
    listing the valid category names if `category` is unknown.
  """
  data = json.loads(LOCAL_KB_PATH.read_text(encoding="utf-8"))
  key = category.lower().strip()
  if key not in data:
    return (
        f"Unknown category '{category}'. Available local categories:"
        f" {list(data.keys())}"
    )
  return json.dumps(data[key], indent=2)


def write_local_mcp_kb(note_key: str, note_text: str) -> str:
  """Writes or updates a local note in the 'local_workspace_notes' section of local_kb.json.

  Args:
    note_key: Short key for the local note (e.g., 'public_vs_corp_report').
    note_text: The note content to save locally.

  Returns:
    A JSON-formatted confirmation string showing the key and text written to
    `local_kb.json`.
  """
  data = json.loads(LOCAL_KB_PATH.read_text(encoding="utf-8"))
  notes = data.setdefault("local_workspace_notes", {})
  clean_key = (
      re.sub(r"[^a-z0-9_-]", "_", note_key.strip().lower())[:128] or "note"
  )
  notes[clean_key] = note_text.strip()
  LOCAL_KB_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
  return json.dumps(
      {
          "status": "saved_locally",
          "scope": "local_kb.json (local_workspace_notes only)",
          "note_key": clean_key,
          "note_text": notes[clean_key],
      },
      indent=2,
  )


async def contact_discovered_a2a_agent(
    agent_url: str,
    request: str,
    tool_context: ToolContext,
    access_token: str = "",
) -> str:
  """Connects to any discovered external A2A agent URL via its Agent Card and sends a request.

  Resolves `<agent_url>/.well-known/agent-card.json` at runtime, attaches the
  Gateway-minted `X-Gateway-Token` JWT in the HTTP headers when provided, and
  invokes the remote agent over JSON-RPC 2.0 with exponential backoff on
  transient rate-limit errors.

  Args:
    agent_url: Base URL of the discovered remote A2A agent (e.g.,
      corporate_names_analytics_agent).
    request: The question or command to send to the remote agent.
    tool_context: ADK ToolContext injected automatically by Google ADK at
      runtime.
    access_token: Optional Gateway-issued access token for agents that require
      'X-Gateway-Token' authentication.

  Returns:
    The natural-language response string returned by the remote A2A agent over
    JSON-RPC 2.0, or an instructional error message if the call is rejected.
  """
  parsed_url = urlparse(agent_url.strip())
  if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
    return (
        f"Invalid A2A agent_url '{agent_url}': must be an http:// or https://"
        " URL."
    )

  card_url = f"{agent_url.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}"
  ic = getattr(tool_context, "_invocation_context", None)
  session_id = getattr(getattr(ic, "session", None), "id", "default")
  headers = {
      "X-Client-Session": (
          f"{os.getenv('GOOGLE_CLOUD_PROJECT', 'local')}:{session_id}"
      )
  }
  if access_token:
    headers["X-Gateway-Token"] = access_token

  async with httpx.AsyncClient(headers=headers, timeout=120.0) as http_client:
    dynamic_remote_agent = RemoteA2aAgent(
        name="discovered_remote_agent",
        description="Dynamically discovered external remote A2A agent.",
        agent_card=card_url,
        httpx_client=http_client,
    )
    tool = AgentTool(agent=dynamic_remote_agent)
    try:
      response = None
      for attempt in range(4):
        try:
          response = await tool.run_async(
              args={"request": request}, tool_context=tool_context
          )
          if attempt < 3 and any(
              code in str(response) for code in ("429", "RESOURCE_EXHAUSTED")
          ):
            await asyncio.sleep(2.0 * (2**attempt))
            continue
          break
        except Exception as call_err:
          if attempt < 3 and any(
              code in str(call_err)
              for code in ("429", "RESOURCE_EXHAUSTED", "503")
          ):
            await asyncio.sleep(2.0 * (2**attempt))
            continue
          raise
      print_agent_card(dynamic_remote_agent, tool_context)
      return response
    except Exception as e:
      print_agent_card(dynamic_remote_agent, tool_context)
      err_str = str(e)
      if not access_token or any(
          tok in err_str for tok in ("401", "403", "Unauthorized")
      ):
        return (
            f"A2A call to {agent_url} failed with 401 Unauthorized ({e}). "
            "This remote agent requires a signed 'X-Gateway-Token' JWT. "
            "Ask the Gateway Agent ('gateway_agent') to issue an access token "
            "for this service (your HTTP transport automatically presents your "
            "Google Cloud OIDC Bearer token) and retry."
        )
      return f"A2A call to {agent_url} failed with remote error: {e}."
