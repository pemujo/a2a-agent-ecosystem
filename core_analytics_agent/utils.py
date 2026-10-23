"""Tracing, Google Cloud OIDC transport auth, JWT redaction, and A2A lifecycle utilities."""

import json
import os
from pathlib import Path
import re
import uuid
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.oauth2 import id_token as google_id_token
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse

DEFAULT_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)
if "retry_options" in Gemini.model_fields:
  Gemini.model_fields["retry_options"].default = DEFAULT_RETRY_OPTIONS
  Gemini.model_rebuild(force=True)

_orig_generate_content_async = Gemini.generate_content_async


async def _generate_content_async_with_retry(
    self, llm_request, stream: bool = False
):
  """Wraps `Gemini.generate_content_async` to inject `HttpRetryOptions` on every LLM call.

  Args:
    self: The `Gemini` model instance invoking the LLM request.
    llm_request: The outgoing `LlmRequest` containing prompt contents and config.
    stream: Whether to stream partial responses from the Gemini endpoint.

  Yields:
    `LlmResponse` chunks produced by the underlying `Gemini.generate_content_async` call.
  """
  llm_request.config = llm_request.config or types.GenerateContentConfig()
  llm_request.config.http_options = (
      llm_request.config.http_options or types.HttpOptions()
  )
  if not llm_request.config.http_options.retry_options:
    llm_request.config.http_options.retry_options = (
        self.retry_options or DEFAULT_RETRY_OPTIONS
    )
  async for resp in _orig_generate_content_async(
      self, llm_request, stream=stream
  ):
    yield resp


Gemini.generate_content_async = _generate_content_async_with_retry


def _load_dotenv_if_present() -> None:
  """Loads `.env` (from agent folder or repo root) if env vars aren't already exported."""
  base_dir = Path(__file__).resolve().parent
  for env_path in (base_dir / ".env", base_dir.parent / ".env"):
    if env_path.exists():
      for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
          continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
  if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")


_load_dotenv_if_present()

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8001").rstrip("/")
LOCAL_KB_PATH = Path(__file__).parent / "local_kb.json"
_OIDC_ADC_CREDS = None


def fetch_google_oidc_bearer_token(target_audience: str = "") -> str:
  """Acquires a real Google Cloud OIDC ID token or ADC OAuth2 token out-of-band.

  Reuses cached Application Default Credentials (`_OIDC_ADC_CREDS`) while valid
  and refreshes only when expired to avoid redundant OAuth2 network calls.

  Args:
    target_audience: Target service URL for OIDC ID token generation.

  Returns:
    A Google-signed OIDC ID token, an ADC OAuth2 access token, or a local dev
    fallback token when running offline.
  """
  global _OIDC_ADC_CREDS
  auth_req = GoogleAuthRequest()
  if target_audience:
    try:
      token = google_id_token.fetch_id_token(auth_req, target_audience)
      if token:
        return token
    except Exception:
      pass

  try:
    if _OIDC_ADC_CREDS is None:
      _OIDC_ADC_CREDS, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
    if (
        not _OIDC_ADC_CREDS.valid
        or _OIDC_ADC_CREDS.expired
        or not _OIDC_ADC_CREDS.token
    ):
      _OIDC_ADC_CREDS.refresh(auth_req)
    id_tok = getattr(_OIDC_ADC_CREDS, "id_token", None)
    if id_tok:
      return id_tok
    if getattr(_OIDC_ADC_CREDS, "token", None):
      return _OIDC_ADC_CREDS.token
  except Exception:
    pass

  return f"local-adc-oidc:{os.getenv('USER', 'developer')}@gcp.local"


class GoogleCloudOIDCAuth(httpx.Auth):
  """`httpx.Auth` flow that attaches `Authorization: Bearer <Google_OIDC_Token>` out-of-band."""

  def __init__(self, target_audience: str = "") -> None:
    """Initializes the OIDC auth handler with the target service audience URL."""
    self.target_audience = target_audience

  def auth_flow(self, request: httpx.Request):
    """Fetches a fresh Google Cloud OIDC/ADC Bearer token and injects it into the request header."""
    request.headers["Authorization"] = (
        f"Bearer {fetch_google_oidc_bearer_token(self.target_audience)}"
    )
    yield request


def get_gateway_httpx_client() -> httpx.AsyncClient:
  """Returns an `httpx.AsyncClient` configured with out-of-band Google Cloud OIDC auth."""
  return httpx.AsyncClient(
      auth=GoogleCloudOIDCAuth(target_audience=GATEWAY_URL),
      timeout=120.0,
  )


def mask_secret(value: str) -> str:
  """Masks a JWT or secret string, keeping only the first 6 and last 6 characters."""
  return "****" if len(value) <= 14 else f"{value[:6]}...{value[-6:]}"


_JWT_REGEX = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)


def extract_jwt(text: str) -> str | None:
  """Extracts an HS256/RS256 JWT (`eyJ...`) from a tool response string if present."""
  match = _JWT_REGEX.search(str(text))
  return match.group(0) if match else None


def redact_log(text: str) -> str:
  """Redacts signed JWTs (`eyJ...`) in terminal and UI log output."""
  return _JWT_REGEX.sub(lambda m: mask_secret(m.group(0)), str(text))


def format_agent_card(remote_agent: RemoteA2aAgent | None) -> str | None:
  """Formats the resolved Agent Card details if available."""
  card = getattr(remote_agent, "_agent_card", None)
  if not card:
    return None
  skills = [s.name for s in (card.skills or [])]
  return (
      f"  🪪 [Agent Card Resolved] name='{card.name}' | skills={skills}"
      f" | description='{card.description}'"
  )


def print_agent_card(
    remote_agent: RemoteA2aAgent, tool_context=None
) -> None:
  """Prints the resolved Agent Card details and caches it if `tool_context` is provided."""
  line = format_agent_card(remote_agent)
  if line:
    if tool_context is not None:
      tool_context.state["temp:last_agent_card_line"] = line
    else:
      print(line, flush=True)


async def _emit_trace(
    tool_context, lines: list[str], *, leading_newline: bool = False
) -> None:
  """Prints trace lines to the terminal and streams them to the ADK Web UI."""
  clean_lines = [line for line in lines if line]
  if not clean_lines:
    return

  prefix = "\n" if leading_newline else ""
  suffix = "" if leading_newline else "\n"
  print(prefix + "\n".join(clean_lines) + suffix, flush=True)

  ic = getattr(tool_context, "_invocation_context", None)
  event_queue = getattr(ic, "_event_queue", None) if ic else None
  if ic is not None and event_queue is not None:
    md_text = "  \n".join(line.strip() for line in clean_lines)
    author = (
        getattr(getattr(ic, "agent", None), "name", None)
        or "core_analytics_agent"
    )
    await ic._enqueue_event(
        Event(
            invocation_id=ic.invocation_id,
            author=author,
            branch=ic.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=md_text, thought=True)],
            ),
        )
    )


async def log_before_tool(tool, args, tool_context):
  """Prints and streams an explicit transport and request trace before every tool call.

  Restores the cached `X-Gateway-Token` JWT from session state when the LLM
  omits or masks `access_token` on calls to `contact_discovered_a2a_agent`
  (unless the user is explicitly testing an unauthenticated request).
  """
  lines: list[str] = []
  if tool.name == "analyze_name_in_bigquery":
    first_name = args.get("first_name", "")
    lines = [
        "  🧭 [Transport: BIGQUERY PUBLIC DATASET] Running SQL analytics on"
        " 'bigquery-public-data.usa_names.usa_1910_current'",
        f"  📊 [BigQuery SQL] Aggregating total births, peak year, and top"
        f" state for name='{first_name}'...",
    ]
  elif tool.name == "query_local_mcp_kb":
    category = args.get("category", "")
    lines = [
        "  🧭 [Transport: LOCAL MCP (READ)] Reading local MCP source (No"
        " remote REST or A2A call)",
        f"  💾 [Local MCP Read] Reading '{category}' from {LOCAL_KB_PATH.name}...",
    ]
  elif tool.name == "write_local_mcp_kb":
    note_key = args.get("note_key", "")
    lines = [
        "  🧭 [Transport: LOCAL MCP (WRITE)] Writing to local MCP source (No"
        " remote REST or A2A call)",
        f"  ✏️  [Local MCP Write] Saving '{note_key}' into {LOCAL_KB_PATH.name}...",
    ]
  elif tool.name == "gateway_agent":
    lines = [
        f"  🧭 [Transport: A2A PROTOCOL] Calling remote Gateway Agent"
        f" ({GATEWAY_URL}) with Google Cloud OIDC Bearer token",
        f"  🌐 [A2A Discovery] Fetching Agent Card:"
        f" {GATEWAY_URL}{AGENT_CARD_WELL_KNOWN_PATH}",
        f"  📤 [A2A Request -> {GATEWAY_URL}] {redact_log(args.get('request'))}",
    ]
  elif tool.name == "contact_discovered_a2a_agent":
    target_url = args.get("agent_url", "").rstrip("/")
    access_token = args.get("access_token", "")
    request_text = args.get("request", "")
    cached_token = tool_context.state.get("last_gateway_token", "")

    user_text = ""
    ic = getattr(tool_context, "_invocation_context", None)
    if ic and getattr(ic, "user_content", None) and ic.user_content.parts:
      user_text = " ".join(
          p.text for p in ic.user_content.parts if getattr(p, "text", None)
      )

    if any(
        phrase in (request_text + " " + user_text).lower()
        for phrase in ("without", "no token", "unauthenticated")
    ):
      access_token = ""
      args["access_token"] = ""
    elif cached_token and (
        not access_token or "." not in access_token or "*" in access_token
    ):
      access_token = cached_token
      args["access_token"] = cached_token

    auth_line = (
        f"  🔐 [A2A Auth] Attaching signed Gateway JWT ('X-Gateway-Token':"
        f" {mask_secret(access_token)})"
        if access_token
        else "  ⚠️  [A2A Auth] No access_token provided (sending unauthenticated request)"
    )
    lines = [
        f"  🧭 [Transport: A2A PROTOCOL] Calling external Remote A2A Agent"
        f" ({target_url}) via Agent-to-Agent (A2A)",
        f"  🌐 [A2A Discovery] Fetching Agent Card:"
        f" {target_url}{AGENT_CARD_WELL_KNOWN_PATH}",
        auth_line,
        f"  📤 [A2A Request -> {target_url}] {redact_log(request_text)}",
    ]

  await _emit_trace(tool_context, lines, leading_newline=True)


async def log_after_tool(tool, args, tool_context, tool_response):
  """Prints and streams the response trace after every tool call.

  Caches any newly minted `X-Gateway-Token` JWT returned by `gateway_agent` into
  `tool_context.state['last_gateway_token']` so subsequent dynamic A2A calls can
  attach it automatically.
  """
  lines: list[str] = []
  if tool.name == "analyze_name_in_bigquery":
    lines = [
        f"  📈 [BigQuery Result] Completed US Names analytics for"
        f" '{args.get('first_name', '')}'."
    ]
  elif tool.name == "query_local_mcp_kb":
    lines = [
        f"  📄 [Local MCP Read] Completed read for '{args.get('category', '')}'."
    ]
  elif tool.name == "write_local_mcp_kb":
    lines = [
        f"  ✅ [Local MCP Write] Saved '{args.get('note_key', '')}' to"
        f" {LOCAL_KB_PATH.name}."
    ]
  elif tool.name == "gateway_agent":
    jwt_token = extract_jwt(str(tool_response))
    if jwt_token:
      tool_context.state["last_gateway_token"] = jwt_token
    card_line = format_agent_card(getattr(tool, "agent", None))
    if card_line:
      lines.append(card_line)
    lines.append(
        f"  📥 [A2A Response <- {GATEWAY_URL}] {redact_log(tool_response)}"
    )
  elif tool.name == "contact_discovered_a2a_agent":
    target_url = args.get("agent_url", "").rstrip("/")
    card_line = tool_context.state.get("temp:last_agent_card_line")
    if card_line:
      lines.append(card_line)
      tool_context.state["temp:last_agent_card_line"] = ""
    if "401 Unauthorized" in str(tool_response):
      lines.append(
          f"  🚫 [A2A Auth Rejected <- {target_url}] 401 Unauthorized (Missing"
          " or invalid X-Gateway-Token JWT)!"
      )
    else:
      lines.append(
          f"  📥 [A2A Response <- {target_url}] {redact_log(tool_response)}"
      )

  await _emit_trace(tool_context, lines, leading_newline=False)


# --- Full A2A Task Lifecycle Middleware for `to_a2a()` ---
_MAX_STORED_TASKS = 256
_LIFECYCLE_TASK_STORE: dict[str, dict] = {}


def _store_lifecycle_task(task_id: str, task_obj: dict) -> None:
  """Stores an A2A Task in `_LIFECYCLE_TASK_STORE` with FIFO eviction at `_MAX_STORED_TASKS`."""
  if (
      task_id not in _LIFECYCLE_TASK_STORE
      and len(_LIFECYCLE_TASK_STORE) >= _MAX_STORED_TASKS
  ):
    _LIFECYCLE_TASK_STORE.pop(next(iter(_LIFECYCLE_TASK_STORE)), None)
  _LIFECYCLE_TASK_STORE[task_id] = task_obj


def _public_task(task_obj: dict) -> dict:
  """Returns a copy of `task_obj` excluding private internal keys (`_*`)."""
  return {k: v for k, v in task_obj.items() if not k.startswith("_")}


def _make_agent_msg(text: str, task_id: str, context_id: str) -> dict:
  """Constructs a standard A2A agent `Message` dictionary."""
  return {
      "role": "agent",
      "messageId": uuid.uuid4().hex,
      "taskId": task_id,
      "contextId": context_id,
      "parts": [{"kind": "text", "text": text}],
  }


def _status_event(
    task_id: str,
    context_id: str,
    state: str,
    *,
    message: dict | None = None,
    final: bool = False,
) -> dict:
  """Constructs an A2A `status-update` SSE event dictionary."""
  status: dict = {"state": state}
  if message is not None:
    status["message"] = message
  return {
      "kind": "status-update",
      "taskId": task_id,
      "contextId": context_id,
      "status": status,
      "final": final,
  }


def _sse_response(rpc_id, events: list[dict]) -> StreamingResponse:
  """Returns an SSE `StreamingResponse` yielding JSON-RPC 2.0 event payloads."""

  async def _stream():
    """Yields each event formatted as a JSON-RPC 2.0 SSE `data:` frame."""
    for ev in events:
      yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': rpc_id, 'result': ev})}\n\n"

  return StreamingResponse(_stream(), media_type="text/event-stream")


def _extract_result_text(result: dict) -> str:
  """Extracts text from an A2A Task or Message result dictionary.

  Inspects `artifacts`, `status.message`, direct `parts`, and `history` in
  priority order to return the plain-text response content.
  """
  texts: list[str] = []
  for artifact in result.get("artifacts") or []:
    for part in artifact.get("parts") or []:
      if isinstance(part, dict) and part.get("text"):
        texts.append(part["text"])
  status_msg = (result.get("status") or {}).get("message") or {}
  for part in status_msg.get("parts") or []:
    if isinstance(part, dict) and part.get("text"):
      texts.append(part["text"])
  for part in result.get("parts") or []:
    if isinstance(part, dict) and part.get("text"):
      texts.append(part["text"])
  if not texts:
    for msg in reversed(result.get("history") or []):
      if msg.get("role") in ("agent", "model", "assistant"):
        for part in msg.get("parts") or []:
          if isinstance(part, dict) and part.get("text"):
            texts.append(part["text"])
        if texts:
          break
  return "\n".join(texts).strip() if texts else json.dumps(result)


async def _a2a_task_lifecycle_dispatch(request: Request, call_next):
  """Implements the stateful A2A Task Lifecycle (`submitted` -> `working` -> `input-required` -> `completed`, SSE `message/stream`, `tasks/get`, and `tasks/cancel`) on the local ADK `to_a2a()` server."""
  if request.method != "POST":
    return await call_next(request)

  raw_body = await request.body()
  try:
    rpc_req = json.loads(raw_body)
  except Exception:
    return await call_next(request)

  if not isinstance(rpc_req, dict):
    return await call_next(request)

  rpc_id = rpc_req.get("id")
  method = rpc_req.get("method", "")
  params = rpc_req.get("params") or {}

  # 1. Handle `tasks/get` (Task State Inspection & Polling)
  if method == "tasks/get":
    query_task_id = params.get("id", "")
    if query_task_id in _LIFECYCLE_TASK_STORE:
      stored = _public_task(_LIFECYCLE_TASK_STORE[query_task_id])
      state = (stored.get("status") or {}).get("state", "unknown")
      print(
          f"  🔍 [A2A Server tasks/get] Returning persisted Task"
          f" taskId={query_task_id} -> state='{state}'"
          f" (history={len(stored.get('history') or [])} messages,"
          f" artifacts={len(stored.get('artifacts') or [])})",
          flush=True,
      )
      return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": stored})
    return await call_next(request)

  # 2. Handle `tasks/cancel` (Task Cancellation)
  if method == "tasks/cancel":
    cancel_task_id = params.get("id", "")
    if cancel_task_id in _LIFECYCLE_TASK_STORE:
      task_entry = _LIFECYCLE_TASK_STORE[cancel_task_id]
      cancel_msg = _make_agent_msg(
          f"Task {cancel_task_id} was canceled by the client.",
          cancel_task_id,
          task_entry.get("contextId", ""),
      )
      task_entry["status"] = {"state": "canceled", "message": cancel_msg}
      print(
          f"  🛑 [A2A Server tasks/cancel] Canceled Task"
          f" taskId={cancel_task_id} -> state='canceled'",
          flush=True,
      )
      return JSONResponse(
          {"jsonrpc": "2.0", "id": rpc_id, "result": _public_task(task_entry)}
      )
    return await call_next(request)

  # 3. Handle `message/send` and `message/stream`
  if method in ("message/send", "message/stream"):
    msg = params.get("message") or {}
    user_text = " ".join(
        p.get("text", "")
        for p in (msg.get("parts") or [])
        if isinstance(p, dict)
    ).strip()
    incoming_task_id = msg.get("taskId") or ""
    incoming_context_id = msg.get("contextId") or ""

    resumed_entry = None
    if incoming_task_id and incoming_task_id in _LIFECYCLE_TASK_STORE:
      candidate = _LIFECYCLE_TASK_STORE[incoming_task_id]
      if (candidate.get("status") or {}).get("state") == "input-required":
        resumed_entry = candidate
    elif "confirmed" in user_text.lower():
      for candidate in reversed(list(_LIFECYCLE_TASK_STORE.values())):
        if (candidate.get("status") or {}).get("state") == "input-required":
          resumed_entry = candidate
          break

    wants_local_write = any(
        kw in user_text.lower()
        for kw in (
            "save",
            "write",
            "local_kb",
            "ebba_a2a_report",
            "elena_a2a_report",
            "local knowledge base",
        )
    )

    # Turn 1: Pause in `input-required` before mutating local MCP storage
    if (
        wants_local_write
        and resumed_entry is None
        and "confirmed" not in user_text.lower()
    ):
      task_id = incoming_task_id or f"task-{uuid.uuid4().hex[:12]}"
      context_id = incoming_context_id or f"ctx-{uuid.uuid4().hex[:12]}"
      pause_text = (
          f"A2A Task '{task_id}' paused in state 'input-required': "
          "Writing to local MCP storage ('local_kb.json') requires caller "
          f"confirmation. Please resume taskId='{task_id}' "
          f"(contextId='{context_id}') with 'CONFIRMED' to execute the "
          "analytics workflow and commit the local MCP write."
      )
      agent_pause_msg = _make_agent_msg(pause_text, task_id, context_id)
      task_obj = {
          "kind": "task",
          "id": task_id,
          "contextId": context_id,
          "status": {"state": "input-required", "message": agent_pause_msg},
          "history": [msg, agent_pause_msg],
          "artifacts": [],
          "_original_request": user_text,
      }
      _store_lifecycle_task(task_id, task_obj)

      print(
          f"\n  🔄 [A2A Server Task Lifecycle] Created taskId={task_id} |"
          f" contextId={context_id}\n"
          f"     ├─ State: submitted      (Task accepted via {method})\n"
          "     ├─ State: working        (Evaluating local MCP write"
          " policy...)\n"
          "     └─ State: input-required (Paused: awaiting caller confirmation"
          " before writing to local_kb.json)\n",
          flush=True,
      )

      if method == "message/stream":
        return _sse_response(
            rpc_id,
            [
                _status_event(task_id, context_id, "submitted"),
                _status_event(task_id, context_id, "working"),
                _status_event(
                    task_id,
                    context_id,
                    "input-required",
                    message=agent_pause_msg,
                    final=True,
                ),
            ],
        )
      return JSONResponse(
          {"jsonrpc": "2.0", "id": rpc_id, "result": _public_task(task_obj)}
      )

    # Turn 2: Resume the paused `taskId` (`input-required` -> `working` -> `completed`)
    if resumed_entry is not None:
      task_id = resumed_entry["id"]
      context_id = resumed_entry["contextId"]
      original_request = resumed_entry.get("_original_request", user_text)
      print(
          f"\n  ▶️  [A2A Server Task Lifecycle] Resuming paused"
          f" taskId={task_id} | contextId={context_id}\n"
          "     ├─ State: input-required -> working (Caller confirmation"
          " received; executing ADK root_agent...)",
          flush=True,
      )

      forwarded_rpc = {
          "jsonrpc": "2.0",
          "id": rpc_id,
          "method": "message/send",
          "params": {
              "message": {
                  "role": "user",
                  "messageId": uuid.uuid4().hex,
                  "parts": [{"kind": "text", "text": original_request}],
              }
          },
      }
      new_body = json.dumps(forwarded_rpc).encode("utf-8")
      request._body = new_body
      request.scope["headers"] = [
          (
              k,
              str(len(new_body)).encode("latin-1")
              if k.lower() == b"content-length"
              else v,
          )
          for k, v in request.scope.get("headers", [])
      ]

      downstream_resp = await call_next(request)
      resp_bytes = b"".join(
          [chunk async for chunk in downstream_resp.body_iterator]
      )
      try:
        downstream_result = (json.loads(resp_bytes)).get("result") or {}
      except Exception:
        downstream_result = {}

      final_text = _extract_result_text(downstream_result)
      completed_artifacts = downstream_result.get("artifacts") or [
          {
              "artifactId": f"artifact-{uuid.uuid4().hex[:8]}",
              "name": "analytics_report",
              "parts": [{"kind": "text", "text": final_text}],
          }
      ]
      agent_final_msg = _make_agent_msg(final_text, task_id, context_id)
      completed_task = {
          "kind": "task",
          "id": task_id,
          "contextId": context_id,
          "status": {"state": "completed", "message": agent_final_msg},
          "history": (
              resumed_entry.get("history", []) + [msg, agent_final_msg]
          ),
          "artifacts": completed_artifacts,
      }
      _store_lifecycle_task(task_id, completed_task)
      print(
          f"     └─ State: working -> completed      (Task {task_id} finished;"
          f" {len(completed_artifacts)} artifact(s) persisted in TaskStore)\n",
          flush=True,
      )

      if method == "message/stream":
        return _sse_response(
            rpc_id,
            [
                _status_event(task_id, context_id, "working"),
                {
                    "kind": "artifact-update",
                    "taskId": task_id,
                    "contextId": context_id,
                    "artifact": completed_artifacts[0],
                    "lastChunk": True,
                },
                _status_event(
                    task_id,
                    context_id,
                    "completed",
                    message=agent_final_msg,
                    final=True,
                ),
            ],
        )
      return JSONResponse(
          {"jsonrpc": "2.0", "id": rpc_id, "result": completed_task}
      )

  return await call_next(request)


def _install_a2a_lifecycle_wrapper() -> None:
  """Wraps `google.adk.a2a.utils.agent_to_a2a.to_a2a` so `a2a_app = to_a2a(root_agent, ...)` automatically includes the A2A Task Lifecycle middleware."""
  try:
    from google.adk.a2a.utils import agent_to_a2a

    orig_to_a2a = agent_to_a2a.to_a2a
    if getattr(orig_to_a2a, "_has_lifecycle_wrapper", False):
      return

    def _wrapped_to_a2a(agent, *args, **kwargs):
      """Builds the Starlette A2A ASGI app via `orig_to_a2a` and attaches `_a2a_task_lifecycle_dispatch`."""
      app = orig_to_a2a(agent, *args, **kwargs)
      app.add_middleware(
          BaseHTTPMiddleware, dispatch=_a2a_task_lifecycle_dispatch
      )
      return app

    _wrapped_to_a2a._has_lifecycle_wrapper = True  # type: ignore[attr-defined]
    agent_to_a2a.to_a2a = _wrapped_to_a2a
  except Exception:
    pass


_install_a2a_lifecycle_wrapper()
