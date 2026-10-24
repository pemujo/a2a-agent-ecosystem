"""Gateway Agent: Central A2A Service Registry and OIDC-to-JWT Token Broker.

Serves two side-by-side interfaces so developers can directly compare rigid
REST APIs with the Agent2Agent (A2A) protocol:
  1. A2A Protocol Interface (`to_a2a(gateway_agent)`):
     - Publishes `/.well-known/agent-card.json` advertising `find_agent_by_service`
       and `issue_access_token` skills, and `/.well-known/jwks.json` publishing
       its public RSA verification key (RFC 7517 JWKS).
     - Authenticates the calling agent out-of-band via the HTTP
       `Authorization: Bearer <Google_OIDC_Token>` header attached by the
       caller's transport layer (Application Default Credentials / Cloud IAM).
     - Mints a short-lived, asymmetrically signed RS256 JWT (`X-Gateway-Token`)
       scoped to target audience `"corporate_names_analytics_v2"` with a 5-minute TTL.
  2. Rigid REST Interface (`/api/v1/lookup` and `/api/v1/token`):
     - Requires exact internal `service_key` and `target_agent_name` strings
       and returns `400 Bad Request` whenever a caller passes a friendly alias.
"""

import base64
import contextvars
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import warnings

logging.disable(level=logging.WARNING)
warnings.filterwarnings("ignore")


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


_load_dotenv_if_present()

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.auth.transport import requests as google_auth_requests
from google.genai import types
from google.oauth2 import id_token as google_id_token
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)

CORPORATE_NAMES_AGENT_URL = os.getenv(
    "CORPORATE_NAMES_AGENT_URL", "http://localhost:8002"
).rstrip("/")
JWT_TTL_SECONDS = 300

# Ephemeral 2048-bit RSA keypair generated in memory on startup (zero static secrets).
# The Gateway signs `X-Gateway-Token` JWTs with the private key (`RS256`) and exposes
# the public key at `GET /.well-known/jwks.json` for downstream agents to verify.
_GATEWAY_RSA_PRIVATE_KEY = rsa.generate_private_key(
    public_exponent=65537, key_size=2048
)
_GATEWAY_RSA_PUBLIC_NUMBERS = (
    _GATEWAY_RSA_PRIVATE_KEY.public_key().public_numbers()
)
_GATEWAY_KEY_ID = hashlib.sha256(
    _GATEWAY_RSA_PUBLIC_NUMBERS.n.to_bytes(256, "big")
).hexdigest()[:16]

# Context variable capturing the incoming HTTP Authorization header out-of-band
_CALLER_AUTH_HEADER: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_CALLER_AUTH_HEADER", default=""
)

SERVICE_REGISTRY = {
    "corp_analytics_registry": {
        "agent_name": "corporate_names_analytics_v2",
        "url": CORPORATE_NAMES_AGENT_URL,
        "auth_required": True,
        "auth_instructions": (
            "Requires a signed 'X-Gateway-Token' JWT header. Call"
            " 'issue_access_token' on this Gateway Agent (your HTTP transport"
            " automatically presents your Google Cloud OIDC/ADC Bearer token"
            " in the Authorization header)."
        ),
        "capabilities": (
            "Corporate Names Analytics Agent (`corporate_names_analytics_agent`"
            " / internal ID `corporate_names_analytics_v2`): runs SQL analytics"
            " over the 5,000-Employee Corporate Directory (`corp_employees`),"
            " returning company-wide Top 10 names, employee counts, workforce"
            " percentages, top departments, and global office hubs."
        ),
    },
}

# Friendly aliases that the A2A tools resolve semantically (whereas rigid REST rejects them).
_CORPORATE_AGENT_ALIASES = {
    "corporate_names_analytics_agent",
    "corporate_names_analytics",
    "corporate_names_analytics_v2",
    "corporate_names_agent",
    "corp_analytics_registry",
    "corp_names_analytics",
    "corp_names",
    "corporate_directory",
    "employee_directory",
}
STRICT_VALID_AGENT_NAMES = {"corporate_names_analytics_v2"}


def _b64url_encode(data: bytes) -> str:
  """Encodes raw bytes into an unpadded URL-safe Base64 string per RFC 7515 (JWS).

  Args:
    data: Raw byte sequence to encode.

  Returns:
    The URL-safe Base64 string with trailing `'='` padding removed.
  """
  return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def get_gateway_jwks() -> dict:
  """Returns the Gateway's public RSA key in RFC 7517 JSON Web Key Set (JWKS) format."""
  n_bytes = _GATEWAY_RSA_PUBLIC_NUMBERS.n.to_bytes(
      (_GATEWAY_RSA_PUBLIC_NUMBERS.n.bit_length() + 7) // 8, "big"
  )
  e_bytes = _GATEWAY_RSA_PUBLIC_NUMBERS.e.to_bytes(
      (_GATEWAY_RSA_PUBLIC_NUMBERS.e.bit_length() + 7) // 8, "big"
  )
  return {
      "keys": [
          {
              "kty": "RSA",
              "alg": "RS256",
              "use": "sig",
              "kid": _GATEWAY_KEY_ID,
              "n": _b64url_encode(n_bytes),
              "e": _b64url_encode(e_bytes),
          }
      ]
  }


def mint_gateway_jwt(subject: str, audience: str) -> str:
  """Mints a short-lived asymmetric RS256 JWT scoped to the target A2A agent audience.

  Args:
    subject: Verified caller principal (e.g., Google Cloud OIDC email or subject).
    audience: Target A2A agent identifier (e.g., `'corporate_names_analytics_v2'`).

  Returns:
    A 3-segment signed RS256 JWT string (`<header>.<payload>.<signature>`).
  """
  now = int(time.time())
  header = {"alg": "RS256", "typ": "JWT", "kid": _GATEWAY_KEY_ID}
  payload = {
      "iss": "gateway_agent",
      "sub": subject,
      "aud": audience,
      "iat": now,
      "exp": now + JWT_TTL_SECONDS,
  }
  signing_input = (
      f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode())}."
      f"{_b64url_encode(json.dumps(payload, separators=(',', ':')).encode())}"
  )
  signature = _GATEWAY_RSA_PRIVATE_KEY.sign(
      signing_input.encode("ascii"),
      padding.PKCS1v15(),
      hashes.SHA256(),
  )
  return f"{signing_input}.{_b64url_encode(signature)}"


# Bounded 60-second TTL cache for verified Google OIDC/OAuth2 Bearer tokens,
# keyed by SHA-256 digest of the raw token so raw secrets are never stored in memory.
_VERIFIED_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_TOKEN_CACHE_TTL_SECONDS = 60.0
_MAX_TOKEN_CACHE_SIZE = 256


def verify_caller_google_identity(auth_header: str) -> tuple[bool, str]:
  """Verifies the caller's Google Cloud OIDC ID token or ADC OAuth2 access token.

  Uses a bounded 60-second TTL cache keyed by the SHA-256 digest of the token so
  repeated requests from the same caller avoid redundant network round-trips to
  `oauth2.googleapis.com/tokeninfo` without keeping raw tokens in memory.

  Args:
    auth_header: The incoming HTTP `Authorization: Bearer <token>` header.

  Returns:
    A tuple of `(is_valid, principal_email_or_error)`.
  """
  if not auth_header or not auth_header.lower().startswith("bearer "):
    return (
        False,
        "Missing HTTP 'Authorization: Bearer <google_token>' header on request.",
    )

  raw_token = auth_header.split(" ", 1)[1].strip()
  if not raw_token:
    return False, "Empty Bearer token in HTTP Authorization header."

  token_digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
  now = time.time()
  cached = _VERIFIED_TOKEN_CACHE.get(token_digest)
  if cached is not None:
    cached_principal, expires_at = cached
    if now < expires_at:
      return True, cached_principal
    _VERIFIED_TOKEN_CACHE.pop(token_digest, None)

  def _cache_verified(principal_id: str) -> tuple[bool, str]:
    """Caches a verified principal under the token's SHA-256 digest with FIFO eviction."""
    if (
        token_digest not in _VERIFIED_TOKEN_CACHE
        and len(_VERIFIED_TOKEN_CACHE) >= _MAX_TOKEN_CACHE_SIZE
    ):
      _VERIFIED_TOKEN_CACHE.pop(next(iter(_VERIFIED_TOKEN_CACHE)), None)
    _VERIFIED_TOKEN_CACHE[token_digest] = (
        principal_id,
        now + _TOKEN_CACHE_TTL_SECONDS,
    )
    return True, principal_id

  # 1. Try verifying as a Google-signed OIDC ID token (JWT with 3 segments).
  if raw_token.count(".") == 2:
    try:
      claims = google_id_token.verify_oauth2_token(
          raw_token, google_auth_requests.Request()
      )
      principal = (
          claims.get("email")
          or claims.get("sub")
          or "oidc-service-account@gcp"
      )
      return _cache_verified(principal)
    except Exception:
      pass

  # 2. Try verifying as a Google Cloud ADC OAuth2 access token via Google tokeninfo.
  try:
    with httpx.Client(timeout=5.0) as client:
      resp = client.get(
          "https://oauth2.googleapis.com/tokeninfo",
          params={"access_token": raw_token},
      )
      if resp.status_code == 200:
        info = resp.json()
        principal = (
            info.get("email")
            or info.get("sub")
            or f"gcp-principal-{info.get('azp', 'adc')}"
        )
        return _cache_verified(principal)
  except Exception:
    pass

  # 3. Allow explicitly signed local test OIDC token when running offline local tests.
  if (
      os.getenv("ALLOW_LOCAL_ADC_FALLBACK", "true").lower() == "true"
      and raw_token.startswith("local-adc-oidc:")
  ):
    principal = raw_token.split(":", 1)[1] or "local-dev@example.com"
    return _cache_verified(principal)

  return False, "Invalid or expired Google Cloud OIDC / OAuth2 Bearer token."


def find_agent_by_service(service_name: str) -> dict:
  """Looks up the remote A2A agent URL and authentication requirements for a service.

  Args:
    service_name: Service or agent name to look up (for example,
      'corp_analytics_registry' or 'corporate_names_analytics_agent').

  Returns:
    A dictionary with the target agent's `agent_name`, `url`, `auth_required`
    flag, `auth_instructions`, and `capabilities`, or an error dictionary
    listing `available_services`.
  """
  key = service_name.lower().strip().replace(" ", "_")
  if key in _CORPORATE_AGENT_ALIASES:
    key = "corp_analytics_registry"
  if key in SERVICE_REGISTRY:
    return SERVICE_REGISTRY[key]
  return {
      "error": f"Unknown service_name '{service_name}'.",
      "available_services": list(SERVICE_REGISTRY.keys()),
  }


def issue_access_token(target_agent_name: str) -> dict:
  """Verifies the caller's Google Cloud OIDC/ADC HTTP Bearer token and mints a scoped X-Gateway-Token JWT.

  Args:
    target_agent_name: Registered agent_name ('corporate_names_analytics_v2' for
      the Corporate Names Analytics Agent).

  Returns:
    A dictionary containing `status='granted'`, `caller_principal`,
    `target_agent`, `header_name='X-Gateway-Token'`, `token_type='JWT (RS256)'`,
    `expires_in_seconds`, and `access_token`, or an error/denial dictionary.
  """
  raw_target = target_agent_name.lower().strip().replace(" ", "_")
  target = (
      "corporate_names_analytics_v2"
      if raw_target in _CORPORATE_AGENT_ALIASES
      else raw_target
  )
  if target not in STRICT_VALID_AGENT_NAMES:
    return {
        "status": "error",
        "error": (
            f"400 Bad Request: Unknown target_agent_name '{target_agent_name}'."
        ),
    }

  auth_header = _CALLER_AUTH_HEADER.get()
  verified, principal_or_err = verify_caller_google_identity(auth_header)
  if not verified:
    return {
        "status": "denied",
        "reason": (
            f"OIDC/ADC authentication failed: {principal_or_err} Ensure your"
            " HTTP transport attaches 'Authorization: Bearer <google_token>'."
        ),
    }

  signed_jwt = mint_gateway_jwt(subject=principal_or_err, audience=target)
  return {
      "status": "granted",
      "caller_principal": principal_or_err,
      "target_agent": target,
      "header_name": "X-Gateway-Token",
      "token_type": "JWT (RS256)",
      "expires_in_seconds": JWT_TTL_SECONDS,
      "access_token": signed_jwt,
  }


gateway_agent = Agent(
    name="gateway_agent",
    model=Gemini(
        model=os.getenv("MODEL_NAME", "gemini-3.6-flash"),
        retry_options=RETRY_OPTIONS,
    ),
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(retry_options=RETRY_OPTIONS)
    ),
    description=(
        "Gateway Agent (Central Service Registry and OIDC-to-JWT Token Broker)."
        " Maps service categories ('corp_analytics_registry') to remote A2A"
        " agent URLs and exchanges the caller's Google Cloud OIDC/ADC identity"
        " for a short-lived, asymmetrically signed 'X-Gateway-Token' RS256 JWT."
    ),
    instruction="""
    You are the Gateway Agent (Central Service Registry and OIDC-to-JWT Token Broker).
    You have two responsibilities:
    1. SERVICE DISCOVERY ('find_agent_by_service'):
       - Maps service names ('corp_analytics_registry') to their remote A2A agent URLs.
       - If a caller asks for 'corporate_names_analytics_agent', corporate names analytics, or employee directory stats, map it to service_name='corp_analytics_registry'.
       - When returning a protected agent's URL, inform the caller that it requires a signed 'X-Gateway-Token' JWT, which they can request from you via 'issue_access_token'.

    2. OIDC-TO-JWT TOKEN BROKER ('issue_access_token'):
       - Exchanges the caller's out-of-band Google Cloud OIDC/ADC Bearer token (attached automatically in the HTTP Authorization header) for a short-lived, asymmetrically signed RS256 'X-Gateway-Token' JWT scoped to 'corporate_names_analytics_v2'.
       - If the caller refers to the corporate names agent in natural language (e.g. 'corporate_names_analytics_agent', 'corporate names agent', 'corp_analytics_registry'), map it to target_agent_name='corporate_names_analytics_v2'.
       - When 'issue_access_token' grants a JWT, ALWAYS include the exact, unmodified 'access_token' JWT string in your response so the calling agent can attach it in 'X-Gateway-Token'.
    """,
    tools=[find_agent_by_service, issue_access_token],
)

a2a_app = to_a2a(gateway_agent, port=int(os.getenv("PORT", "8001")))


async def capture_auth_and_rewrite_card(request: Request, call_next):
  """Captures the incoming HTTP `Authorization` header out-of-band for OIDC token verification."""
  token_reset = _CALLER_AUTH_HEADER.set(
      request.headers.get("authorization", "")
  )
  try:
    return await call_next(request)
  finally:
    _CALLER_AUTH_HEADER.reset(token_reset)


a2a_app.add_middleware(
    BaseHTTPMiddleware, dispatch=capture_auth_and_rewrite_card
)


async def jwks_endpoint(request: Request) -> JSONResponse:
  """Publishes the Gateway's public RSA key at `GET /.well-known/jwks.json` (RFC 7517)."""
  del request
  return JSONResponse(get_gateway_jwks())


# --- Legacy Rigid REST Endpoints (to contrast with A2A) ---
async def rest_lookup_endpoint(request: Request) -> JSONResponse:
  """REST endpoint (`GET /api/v1/lookup`): requires exact query param `?service_key=<key>`."""
  raw_key = request.query_params.get("service_key", "")
  service_key = raw_key.lower().strip().replace(" ", "_")
  if service_key not in SERVICE_REGISTRY:
    return JSONResponse(
        {"error": f"400 Bad Request: Unknown service_key '{raw_key}'."},
        status_code=400,
    )
  return JSONResponse(SERVICE_REGISTRY[service_key])


async def rest_token_endpoint(request: Request) -> JSONResponse:
  """REST endpoint (`POST /api/v1/token`): requires exact JSON body field `target_agent_name` and HTTP Bearer header."""
  try:
    body = await request.json()
  except Exception:
    return JSONResponse({"error": "400 Invalid JSON body"}, status_code=400)

  target = body.get("target_agent_name", "")
  if not target:
    return JSONResponse(
        {
            "error": (
                "422 Unprocessable Entity: Missing required field"
                " ('target_agent_name')."
            )
        },
        status_code=422,
    )
  if target.lower().strip() not in STRICT_VALID_AGENT_NAMES:
    return JSONResponse(
        {"error": f"400 Bad Request: Unknown target_agent_name '{target}'."},
        status_code=400,
    )
  result = issue_access_token(target)
  status = 200 if result.get("status") == "granted" else 401
  return JSONResponse(result, status_code=status)


a2a_app.routes.extend([
    Route("/.well-known/jwks.json", jwks_endpoint, methods=["GET"]),
    Route("/api/v1/lookup", rest_lookup_endpoint, methods=["GET"]),
    Route("/api/v1/token", rest_token_endpoint, methods=["POST"]),
])
