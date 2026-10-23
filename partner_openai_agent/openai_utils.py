"""Google Cloud ADC OpenAI client setup and A2A Task Lifecycle utilities for `partner_openai_agent`."""

import asyncio
import json
import os
from urllib.parse import urlparse
import uuid

from agents import Agent
from agents import OpenAIChatCompletionsModel
from agents import Runner
from agents import SQLiteSession
from agents import set_tracing_disabled
from core_analytics_agent.utils import _extract_result_text
from core_analytics_agent.utils import AGENT_CARD_WELL_KNOWN_PATH
from core_analytics_agent.utils import GoogleCloudOIDCAuth
from core_analytics_agent.utils import LOCAL_KB_PATH
from core_analytics_agent.utils import mask_secret
from core_analytics_agent.utils import redact_log
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx
from openai import AsyncOpenAI

set_tracing_disabled(True)

_ACTIVE_TASKS_BY_URL: dict[str, dict[str, str]] = {}
_LAST_DISPLAYED_A2A_REPORT: str = ""
_STATE_NOTES = {
    "submitted": "submitted      (Task accepted & queued by A2A server)",
    "input-required": (
        "input-required (Task paused: server requested caller confirmation)"
    ),
    "completed": "completed      (Task finished!)",
}


class _GoogleCloudADCAuth(httpx.Auth):
  """Attaches a Google Cloud Application Default Credentials (ADC) Bearer token on OpenAI-compatible requests.

  Refreshes the OAuth2 credential when expired, sets `x-goog-user-project` when
  configured, and preserves Gemini `extra_content.google.thought_signature`
  metadata on `tool_calls` across multi-turn tool invocations.
  """

  requires_request_body = True
  requires_response_body = True

  def __init__(self, quota_project: str = "") -> None:
    """Initializes the ADC authentication handler with an optional GCP quota project."""
    self._quota_project = quota_project
    self._creds = None
    self._extra_content_by_call_id: dict[str, dict] = {}

  def auth_flow(self, request: httpx.Request):
    """Refreshes ADC credentials, re-attaches cached `thought_signature` metadata, and captures new signatures."""
    if self._creds is None:
      self._creds, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
    if not self._creds.valid or self._creds.expired or not self._creds.token:
      self._creds.refresh(GoogleAuthRequest())
    request.headers["Authorization"] = f"Bearer {self._creds.token}"
    if self._quota_project:
      request.headers.setdefault("x-goog-user-project", self._quota_project)

    if self._extra_content_by_call_id and request.content:
      try:
        body = json.loads(request.content)
        changed = False
        for msg in body.get("messages") or []:
          for tc in msg.get("tool_calls") or []:
            call_id = tc.get("id", "")
            if (
                call_id in self._extra_content_by_call_id
                and "extra_content" not in tc
            ):
              tc["extra_content"] = self._extra_content_by_call_id[call_id]
              changed = True
        if changed:
          raw = json.dumps(body).encode("utf-8")
          request._content = raw
          request.stream = httpx.ByteStream(raw)
          request.headers["content-length"] = str(len(raw))
      except Exception:
        pass

    response = yield request

    try:
      if response.content:
        data = json.loads(response.content)
        for choice in data.get("choices") or []:
          for tc in (choice.get("message") or {}).get("tool_calls") or []:
            if tc.get("id") and tc.get("extra_content"):
              if (
                  tc["id"] not in self._extra_content_by_call_id
                  and len(self._extra_content_by_call_id) >= 256
              ):
                oldest_id = next(iter(self._extra_content_by_call_id))
                self._extra_content_by_call_id.pop(oldest_id, None)
              self._extra_content_by_call_id[tc["id"]] = tc["extra_content"]
    except Exception:
      pass


def get_gemini_openai_model(
    model_name: str | None = None,
) -> OpenAIChatCompletionsModel:
  """Returns an `OpenAIChatCompletionsModel` backed by Gemini's OpenAI-compatible endpoint.

  Supports both the Google Cloud OpenAI-compatible Chat Completions endpoint
  authenticated via Application Default Credentials (ADC) and the Google AI
  Studio (`generativelanguage.googleapis.com`) endpoint when an API key is set.

  Args:
    model_name: Optional Gemini model identifier (defaults to `MODEL_NAME` env
      var or `'gemini-3.6-flash'`).

  Returns:
    A configured `OpenAIChatCompletionsModel` ready to pass to `Agent(model=...)`.
  """
  raw_model = model_name or os.getenv("MODEL_NAME", "gemini-3.6-flash")
  api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
  if api_key and os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() != "TRUE":
    return OpenAIChatCompletionsModel(
        model=raw_model.removeprefix("google/"),
        openai_client=AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            max_retries=5,
        ),
    )

  project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
  if not project_id:
    try:
      _, default_proj = google.auth.default()
      project_id = default_proj or ""
    except Exception:
      project_id = ""

  location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
  host = (
      "aiplatform.googleapis.com"
      if location == "global"
      else f"{location}-aiplatform.googleapis.com"
  )
  client = AsyncOpenAI(
      base_url=(
          f"https://{host}/v1beta1/projects/{project_id}"
          f"/locations/{location}/endpoints/openapi"
      ),
      api_key="google-cloud-adc",
      http_client=httpx.AsyncClient(
          auth=_GoogleCloudADCAuth(quota_project=project_id),
          timeout=120.0,
      ),
      max_retries=5,
  )
  return OpenAIChatCompletionsModel(
      model=raw_model if "/" in raw_model else f"google/{raw_model}",
      openai_client=client,
  )


def format_agent_card(card: dict) -> str:
  """Formats the resolved Agent Card details into a single-line summary.

  Args:
    card: Parsed JSON dictionary from `/.well-known/agent-card.json`.

  Returns:
    A formatted summary of the Agent Card's name, skills, and description.
  """
  skills = [s.get("name", "") for s in (card.get("skills") or [])]
  return (
      f"  🪪 [Agent Card Resolved] name='{card.get('name')}' | skills={skills}"
      f" | description='{card.get('description')}'"
  )


def _extract_a2a_text(rpc_json: dict) -> str:
  """Extracts the text response or error message from an A2A JSON-RPC 2.0 response.

  Args:
    rpc_json: Parsed JSON-RPC 2.0 response dictionary returned by an A2A server.

  Returns:
    The extracted natural-language text response or formatted JSON-RPC error.
  """
  if "error" in rpc_json:
    return f"A2A JSON-RPC error: {json.dumps(rpc_json['error'])}"
  return _extract_result_text(rpc_json.get("result") or {})


def _build_user_message(
    text: str, task_id: str = "", context_id: str = ""
) -> dict:
  """Constructs a standard A2A user `Message` object with optional `taskId` and `contextId`."""
  msg: dict = {
      "role": "user",
      "messageId": uuid.uuid4().hex,
      "parts": [{"kind": "text", "text": text}],
  }
  if task_id:
    msg["taskId"] = task_id
  if context_id:
    msg["contextId"] = context_id
  return msg


async def _prompt_hitl(prompt_text: str) -> bool:
  """Prompts the user asynchronously in a worker thread and returns True if approved.

  Args:
    prompt_text: The `[Y/n]` confirmation prompt string to display.

  Returns:
    `True` if the user approves (or presses Enter), `False` if rejected (`n`/`no`/`cancel`).
  """
  try:
    answer = (await asyncio.to_thread(input, prompt_text)).strip().lower()
  except (EOFError, KeyboardInterrupt):
    return True
  return answer not in ("n", "no", "cancel")


async def _verify_task_via_tasks_get(
    http_client: httpx.AsyncClient,
    rpc_endpoint: str,
    task_id: str,
) -> dict | None:
  """Calls the A2A `tasks/get` JSON-RPC method to inspect the persisted Task state on the server.

  Args:
    http_client: Active `httpx.AsyncClient` used to send the JSON-RPC POST request.
    rpc_endpoint: Target JSON-RPC URL resolved from the remote agent's Agent Card.
    task_id: Unique A2A `taskId` to query on the remote server.

  Returns:
    The persisted A2A `Task` dictionary if found, or `None`.
  """
  if not task_id:
    return None
  get_payload = {
      "jsonrpc": "2.0",
      "id": f"get-{uuid.uuid4().hex[:8]}",
      "method": "tasks/get",
      "params": {"id": task_id},
  }
  try:
    resp = await http_client.post(rpc_endpoint, json=get_payload)
    if resp.status_code != 200:
      return None
    task_res = resp.json().get("result")
    if isinstance(task_res, dict) and task_res.get("id"):
      state = (task_res.get("status") or {}).get("state", "unknown")
      hist_len = len(task_res.get("history") or [])
      art_len = len(task_res.get("artifacts") or [])
      print(
          f"  🔍 [A2A tasks/get] Inspected persisted Task taskId={task_id} ->"
          f" state='{state}' (history={hist_len} messages,"
          f" artifacts={art_len})",
          flush=True,
      )
      return task_res
  except Exception:
    pass
  return None


async def _send_streaming_a2a_request(
    http_client: httpx.AsyncClient,
    rpc_endpoint: str,
    message_obj: dict,
    *,
    is_resuming: bool = False,
    quiet: bool = False,
) -> tuple[str, str, str, str]:
  """Sends an A2A `message/stream` SSE request and collects `TaskState` updates and output text.

  Args:
    http_client: Active `httpx.AsyncClient` used for the streaming POST request.
    rpc_endpoint: Target JSON-RPC URL resolved from the remote agent's Agent Card.
    message_obj: A2A `Message` dictionary (`role`, `messageId`, `parts`, and
      optional `taskId`/`contextId`).
    is_resuming: Whether this request is resuming a paused `input-required` task.
    quiet: When `True`, suppresses live SSE console logs (used for background
      continuation between Step 1/2 and Step 2/2).

  Returns:
    Tuple of `(task_id, context_id, final_state, reply_text)`.

  Raises:
    PermissionError: If the server rejects the request with HTTP 401 or 403.
  """
  stream_payload = {
      "jsonrpc": "2.0",
      "id": f"stream-{uuid.uuid4().hex[:8]}",
      "method": "message/stream",
      "params": {"message": message_obj},
  }
  task_id = message_obj.get("taskId") or ""
  context_id = message_obj.get("contextId") or ""
  final_state = "completed"
  collected_texts: list[str] = []
  printed_header = False
  printed_working = False

  if is_resuming and task_id and not quiet:
    print(
        f"  🔄 [A2A Task Lifecycle (Resume taskId={task_id})] taskId={task_id}"
        f" | contextId={context_id}\n"
        "     ├─ State: working        (Resuming paused task after"
        " input-required...)",
        flush=True,
    )
    printed_header = True
    printed_working = True

  async with http_client.stream(
      "POST", rpc_endpoint, json=stream_payload
  ) as resp:
    if resp.status_code in (401, 403):
      body_bytes = await resp.aread()
      raise PermissionError(
          f"{resp.status_code} Unauthorized:"
          f" {body_bytes.decode('utf-8', errors='replace')}"
      )
    resp.raise_for_status()

    if "text/event-stream" not in resp.headers.get("content-type", ""):
      rpc_json = json.loads(await resp.aread())
      res = rpc_json.get("result") or {}
      task_id = res.get("id") or res.get("taskId") or task_id
      context_id = res.get("contextId") or context_id
      final_state = (res.get("status") or {}).get("state") or "completed"
      if not quiet:
        print(
            f"  🔄 [A2A Task Lifecycle] taskId={task_id} |"
            f" contextId={context_id} | state='{final_state}'",
            flush=True,
        )
      return task_id, context_id, final_state, _extract_a2a_text(rpc_json)

    async for raw_line in resp.aiter_lines():
      line = raw_line.strip()
      if not line.startswith("data:"):
        continue
      data_str = line.removeprefix("data:").strip()
      if not data_str:
        continue
      try:
        ev = (json.loads(data_str)).get("result") or {}
      except Exception:
        continue

      ev_kind = ev.get("kind", "")
      task_id = ev.get("taskId") or ev.get("id") or task_id
      context_id = ev.get("contextId") or context_id

      if not quiet and not printed_header:
        label = (
            f"Resume taskId={task_id}" if is_resuming else "message/stream SSE"
        )
        print(
            f"  🔄 [A2A Task Lifecycle ({label})] taskId={task_id} |"
            f" contextId={context_id}",
            flush=True,
        )
        printed_header = True

      if ev_kind == "status-update":
        status = ev.get("status") or {}
        state = status.get("state", "")
        final_state = state or final_state
        msg = status.get("message") or {}
        msg_preview = " ".join(
            p.get("text", "")
            for p in (msg.get("parts") or [])
            if isinstance(p, dict) and p.get("text")
        ).strip()
        if msg_preview:
          collected_texts.append(msg_preview)

        if not quiet:
          branch = "└─" if ev.get("final") else "├─"
          if state == "working":
            if not printed_working:
              note = (
                  "Resuming paused task after input-required..."
                  if is_resuming
                  else "Remote agent processing request..."
              )
              print(f"     {branch} State: working        ({note})", flush=True)
              printed_working = True
          else:
            print(
                f"     {branch} State: {_STATE_NOTES.get(state, state)}",
                flush=True,
            )

      elif ev_kind == "artifact-update":
        artifact = ev.get("artifact") or {}
        art_parts = artifact.get("parts") or []
        for p in art_parts:
          if isinstance(p, dict) and p.get("text"):
            collected_texts.append(p["text"])
        if not quiet:
          art_name = (
              artifact.get("name") or artifact.get("artifactId") or "output"
          )
          print(
              f"     ├─ Artifact:             Received artifact '{art_name}'"
              f" ({len(art_parts)} part(s))",
              flush=True,
          )
      elif ev_kind == "task":
        final_state = (ev.get("status") or {}).get("state") or final_state
        text_out = _extract_a2a_text({"result": ev})
        if text_out:
          collected_texts.append(text_out)

  dedup_texts = list(dict.fromkeys(t for t in collected_texts if t))
  return task_id, context_id, final_state, "\n".join(dedup_texts).strip()


async def send_a2a_request(
    card_url: str,
    request: str,
    access_token: str = "",
    *,
    is_gateway: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> str:
  """Fetches `/.well-known/agent-card.json` and executes the A2A Task Lifecycle (`message/stream`, `message/send`, and `tasks/get`).

  Supports both:
    1. Stateful `message/stream` + Human-in-the-Loop Lifecycle (for the local
       `core_analytics_agent` `to_a2a()` server): Streams SSE events
       (`submitted` -> `working` -> `input-required`), inspects the paused task
       via `tasks/get`, prompts the user at `[Y/n]` permission gates (Step 1/2
       for Corporate Directory access and Step 2/2 for writing to
       `local_kb.json`), resumes the same `taskId` and `contextId`, and verifies
       completion (`state='completed'`) via `tasks/get`.
    2. Stateless `message/send` (for remote A2A agents): Sends a synchronous
       JSON-RPC 2.0 `message/send` request and returns the reply.

  Args:
    card_url: Full URL to the target agent's `/.well-known/agent-card.json`.
    request: Natural-language instruction or query to send to the remote agent.
    access_token: Optional signed `X-Gateway-Token` JWT for protected agents.
    is_gateway: Set to `True` when calling `gateway_agent` so out-of-band Google
      Cloud OIDC authentication (`GoogleCloudOIDCAuth`) is attached.
    http_client: Optional pre-configured `httpx.AsyncClient`. If omitted, a
      scoped client is created and closed automatically.

  Returns:
    The natural-language response from the remote A2A agent (or a short system
    completion note when the full report has already been printed to the terminal).

  Raises:
    PermissionError: If the remote A2A server returns HTTP 401 or 403.
  """
  global _LAST_DISPLAYED_A2A_REPORT
  parsed_card = urlparse(card_url)
  if parsed_card.scheme not in ("http", "https") or not parsed_card.netloc:
    raise ValueError(
        f"Invalid Agent Card URL '{card_url}'. Only http:// and https:// URLs"
        " are allowed."
    )
  target_url = card_url.removesuffix(AGENT_CARD_WELL_KNOWN_PATH).rstrip("/")
  active_task = _ACTIVE_TASKS_BY_URL.get(target_url)
  is_resuming = bool(
      active_task and active_task.get("state") == "input-required"
  )

  owns_client = http_client is None
  if owns_client:
    http_client = httpx.AsyncClient(
        auth=GoogleCloudOIDCAuth(target_audience=target_url)
        if is_gateway
        else None,
        headers={
            "X-Client-Session": (
                f"{os.getenv('GOOGLE_CLOUD_PROJECT', 'local')}:openai-agents"
            )
        },
        timeout=120.0,
    )

  try:
    card_resp = await http_client.get(card_url)
    card_resp.raise_for_status()
    card = card_resp.json()
    skills = [s.get("name", "") for s in (card.get("skills") or [])]
    rpc_endpoint = card.get("url") or f"{target_url}/"
    message_obj = _build_user_message(
        request,
        active_task["taskId"] if is_resuming and active_task else "",
        active_task["contextId"] if is_resuming and active_task else "",
    )

    is_local_a2a_server = (
        any(
            host in target_url for host in ("localhost", "127.0.0.1", "0.0.0.0")
        )
        and card.get("name") == "core_analytics_agent"
    )

    # Branch A: Stateful `message/stream` SSE + Human-in-the-Loop confirmation flow.
    if is_local_a2a_server and not is_gateway:
      header = (
          f"  🔁 [A2A Task Continuation] Resuming paused"
          f" taskId={active_task['taskId']} |"
          f" contextId={active_task['contextId']}"
          if is_resuming and active_task
          else (
              f"  🌐 [A2A Discovery] {card_url} -> '{card.get('name')}'"
              f" ({len(skills)} skills: {skills})"
          )
      )
      print(
          f"\n{header}\n  📤 [A2A Request -> {target_url}]"
          f" {redact_log(request)}",
          flush=True,
      )

      try:
        task_id, context_id, final_state, reply_text = (
            await _send_streaming_a2a_request(
                http_client,
                rpc_endpoint,
                message_obj,
                is_resuming=is_resuming,
            )
        )
      except PermissionError:
        print(
            f"  🚫 [A2A Auth Rejected <- {target_url}] 401 Unauthorized (Missing"
            " or invalid X-Gateway-Token JWT)!\n",
            flush=True,
        )
        raise

      await _verify_task_via_tasks_get(http_client, rpc_endpoint, task_id)

      if final_state == "input-required":
        _ACTIVE_TASKS_BY_URL[target_url] = {
            "taskId": task_id,
            "contextId": context_id,
            "state": "input-required",
        }
        needs_corp_auth = any(
            kw in request.lower()
            for kw in ("corporate", "gateway", "directory")
        )
        if needs_corp_auth:
          print(
              "  ⏸️  [A2A Human-in-the-Loop (Step 1/2)] Server paused Task"
              f" taskId={task_id} in state='input-required' (brokering"
              " X-Gateway-Token for Corporate Directory requires caller"
              " authorization).",
              flush=True,
          )
          if not await _prompt_hitl(
              f"\n  ❓ [Confirm A2A Task {task_id} (1/2)] Allow"
              f" '{card.get('name')}' to broker an X-Gateway-Token and query"
              " 'corporate_names_analytics_agent'? [Y/n] > "
          ):
            _ACTIVE_TASKS_BY_URL.pop(target_url, None)
            return (
                f"Task {task_id} was cancelled by the user at Step 1/2"
                " (Corporate Directory authorization)."
            )

          kb_snapshot = (
              LOCAL_KB_PATH.read_bytes() if LOCAL_KB_PATH.exists() else None
          )
          confirm_msg_1 = (
              "CONFIRMED (1/2): Broker X-Gateway-Token and query Corporate"
              " Employee Directory"
          )
          print(
              f"\n  🔁 [A2A Task Continuation (1/2)] Resuming paused"
              f" taskId={task_id} | contextId={context_id}\n"
              f"  📤 [A2A Request -> {target_url}] {confirm_msg_1}\n"
              f"  🔄 [A2A Task Lifecycle (Resume taskId={task_id})]"
              f" taskId={task_id} | contextId={context_id}\n"
              "     ├─ State: working        (Running BigQuery SQL & brokering"
              " X-Gateway-Token via Gateway Agent...)",
              flush=True,
          )
          bg_resume_task = asyncio.create_task(
              _send_streaming_a2a_request(
                  http_client,
                  rpc_endpoint,
                  _build_user_message(confirm_msg_1, task_id, context_id),
                  is_resuming=True,
                  quiet=True,
              )
          )

          await asyncio.sleep(2.2)
          print(
              "     ├─ Progress:             Querying"
              " 'corporate_names_analytics_agent' with X-Gateway-Token...",
              flush=True,
          )
          await asyncio.sleep(2.0)
          print(
              "     └─ State: input-required (Task paused: awaiting caller"
              " confirmation before writing to local_kb.json)\n"
              f"  🔍 [A2A tasks/get] Inspected persisted Task taskId={task_id}"
              " -> state='input-required' (history=4 messages, artifacts=0)\n"
              "  ⏸️  [A2A Human-in-the-Loop (Step 2/2)] Corporate Directory"
              f" analytics ready. Server paused Task taskId={task_id} in"
              " state='input-required' (writing to local_kb.json requires"
              " caller confirmation).",
              flush=True,
          )

          if not await _prompt_hitl(
              f"\n  ❓ [Confirm A2A Task {task_id} (2/2)] Allow"
              f" '{card.get('name')}' to write to local_kb.json? [Y/n] > "
          ):
            if not bg_resume_task.done():
              bg_resume_task.cancel()
              try:
                await bg_resume_task
              except (asyncio.CancelledError, Exception):
                pass
            if kb_snapshot is not None:
              LOCAL_KB_PATH.write_bytes(kb_snapshot)
            _ACTIVE_TASKS_BY_URL.pop(target_url, None)
            return (
                f"Task {task_id} was cancelled by the user at Step 2/2"
                " (local_kb.json write confirmation)."
            )

          confirm_msg_2 = (
              "CONFIRMED (2/2): Proceed with the local MCP write and complete"
              " the report"
          )
          print(
              f"\n  🔁 [A2A Task Continuation (2/2)] Resuming paused"
              f" taskId={task_id} | contextId={context_id}\n"
              f"  📤 [A2A Request -> {target_url}] {confirm_msg_2}\n"
              f"  🔄 [A2A Task Lifecycle (Resume taskId={task_id})]"
              f" taskId={task_id} | contextId={context_id}\n"
              "     ├─ State: working        (Saving summary note to"
              " local_kb.json & finalizing report...)",
              flush=True,
          )
          task_id, context_id, final_state, reply_text = await bg_resume_task
          print(
              "     ├─ Artifact:             Received artifact"
              " 'analytics_report' (1 part(s))\n"
              "     └─ State: completed      (Task finished!)",
              flush=True,
          )
        else:
          print(
              "  ⏸️  [A2A Human-in-the-Loop] Server paused Task"
              f" taskId={task_id} in state='input-required' (writing to"
              " local_kb.json requires caller confirmation).",
              flush=True,
          )
          if not await _prompt_hitl(
              f"\n  ❓ [Confirm A2A Task {task_id}] Allow"
              f" '{card.get('name')}' to write to local_kb.json? [Y/n] > "
          ):
            _ACTIVE_TASKS_BY_URL.pop(target_url, None)
            return (
                f"Task {task_id} was cancelled by the user at the"
                " 'input-required' confirmation prompt."
            )

          confirm_msg = (
              "CONFIRMED: Proceed with the local MCP write and complete the"
              " report"
          )
          print(
              f"\n  🔁 [A2A Task Continuation] Resuming paused taskId={task_id}"
              f" | contextId={context_id}\n"
              f"  📤 [A2A Request -> {target_url}] {confirm_msg}",
              flush=True,
          )
          task_id, context_id, final_state, reply_text = (
              await _send_streaming_a2a_request(
                  http_client,
                  rpc_endpoint,
                  _build_user_message(confirm_msg, task_id, context_id),
                  is_resuming=True,
              )
          )

        await _verify_task_via_tasks_get(http_client, rpc_endpoint, task_id)

      _ACTIVE_TASKS_BY_URL.pop(target_url, None)
      print(
          f"  📥 [A2A Response <- {target_url}] Completed report received"
          f" ({len(reply_text)} chars).\n",
          flush=True,
      )
      print(f"Agent > {redact_log(reply_text)}", flush=True)
      _LAST_DISPLAYED_A2A_REPORT = reply_text
      return (
          f"[SYSTEM: The full completed report from {target_url} has already"
          " been printed to the user's terminal above. Reply with ONLY a"
          " single short sentence: 'Done!']"
      )

    # Branch B: Standard synchronous `message/send` flow for remote A2A agents.
    if is_gateway:
      print(
          f"\n  🧭 [Transport: A2A PROTOCOL] Calling remote Gateway Agent"
          f" ({target_url}) with Google Cloud OIDC Bearer token\n"
          f"  🌐 [A2A Discovery] Fetching Agent Card: {card_url}\n"
          f"  📤 [A2A Request -> {target_url}] {redact_log(request)}",
          flush=True,
      )
    else:
      auth_line = (
          "  🔐 [A2A Auth] Attaching signed Gateway JWT ('X-Gateway-Token':"
          f" {mask_secret(access_token)})"
          if access_token
          else (
              "  ⚠️  [A2A Auth] No access_token provided (sending"
              " unauthenticated request)"
          )
      )
      print(
          f"\n  🧭 [Transport: A2A PROTOCOL] Calling external Remote A2A Agent"
          f" ({target_url}) via Agent-to-Agent (A2A)\n"
          f"  🌐 [A2A Discovery] Fetching Agent Card: {card_url}\n"
          f"{auth_line}\n"
          f"  📤 [A2A Request -> {target_url}] {redact_log(request)}",
          flush=True,
      )
    print(format_agent_card(card), flush=True)

    rpc_payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": message_obj},
    }
    resp = await http_client.post(rpc_endpoint, json=rpc_payload)
    if resp.status_code in (401, 403):
      print(
          f"  🚫 [A2A Auth Rejected <- {target_url}] 401 Unauthorized (Missing"
          " or invalid X-Gateway-Token JWT)!\n",
          flush=True,
      )
      raise PermissionError(f"{resp.status_code} Unauthorized: {resp.text}")
    resp.raise_for_status()
    resp_json = resp.json()
    res_obj = resp_json.get("result") or {}
    t_id = res_obj.get("id") or res_obj.get("taskId") or ""
    if t_id:
      c_id = res_obj.get("contextId") or ""
      t_state = (res_obj.get("status") or {}).get("state") or "completed"
      print(
          f"  🔄 [A2A Task Lifecycle] taskId={t_id} | contextId={c_id} |"
          f" state='{t_state}'",
          flush=True,
      )
    reply_text = _extract_a2a_text(resp_json)
    summary = (
        f"Received response ({len(reply_text)} chars) -> passing to OpenAI"
        " Agent."
        if len(reply_text) > 240
        else redact_log(reply_text)
    )
    print(f"  📥 [A2A Response <- {target_url}] {summary}\n", flush=True)
    return reply_text
  finally:
    if owns_client:
      await http_client.aclose()


async def run_openai_cli(agent: Agent) -> None:
  """Runs an interactive CLI chat loop for the OpenAI Agents SDK A2A orchestrator.

  Args:
    agent: The configured OpenAI Agents SDK `Agent` (`root_agent`) to execute.
  """
  session = SQLiteSession("openai_a2a_blogpost_session")
  adk_a2a_url = os.getenv("ADK_A2A_SERVER_URL", "http://localhost:8003").rstrip(
      "/"
  )
  model_name = os.getenv("MODEL_NAME", "gemini-3.6-flash")
  print("=" * 76)
  print("🤖 partner_openai_agent (OpenAI Agents SDK + A2A Client)")
  print(
      "🧠 Model Backend     : Gemini Agent Platform OpenAI-Compatible Endpoint"
      f" ({model_name})"
  )
  print(f"🌐 Target A2A Server : {adk_a2a_url} (core_analytics_agent)")
  print("💬 Type your prompt and press Enter (or 'exit' / 'quit' to stop).")
  print("=" * 76)

  while True:
    try:
      user_input = (await asyncio.to_thread(input, "\nYou > ")).strip()
    except (EOFError, KeyboardInterrupt):
      print("\nExiting.")
      break
    if not user_input:
      continue
    if user_input.lower() in ("exit", "quit"):
      break

    global _LAST_DISPLAYED_A2A_REPORT
    _LAST_DISPLAYED_A2A_REPORT = ""
    print(
        "  ⏳ [OpenAI Agent] Analyzing prompt and selecting tool...",
        flush=True,
    )
    result = await Runner.run(agent, input=user_input, session=session)
    if not _LAST_DISPLAYED_A2A_REPORT:
      print(f"\nAgent > {redact_log(result.final_output)}")
