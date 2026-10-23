"""Corporate Names Analytics Agent (`corporate_names_analytics_agent` / `corporate_names_analytics_v2`).

Represents a protected internal enterprise A2A service running on port 8002:
  1. Public Discovery (`GET /.well-known/agent-card.json`): Allows any A2A
     client to inspect the agent's skills without authentication.
  2. Zero-Trust Cryptographic JWT Enforcement (`POST /`): The Starlette
     middleware `verify_gateway_jwt_and_rewrite_card` verifies the HMAC-SHA256
     signature, issuer (`"gateway_agent"`), target audience
     (`"corporate_names_analytics_v2"`), and 5-minute expiration timestamp of
     the `X-Gateway-Token` JWT before executing any JSON-RPC `POST` request.
  3. Specialist Capability:
     - `query_corporate_name_analytics`: Queries the 5,000-employee Corporate
       Employee Directory (`corp_employees`) backed by in-memory SQLite.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import random
import sqlite3
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

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)

GATEWAY_JWT_SECRET = os.getenv(
    "GATEWAY_JWT_SECRET", "a2a-enterprise-gateway-hs256-signing-secret-2026"
)
EXPECTED_AUDIENCE = "corporate_names_analytics_v2"
TOTAL_EMPLOYEES = 5000


def _b64url_decode(segment: str) -> bytes:
  """Decodes an unpadded URL-safe Base64 JWT segment into raw bytes.

  Args:
    segment: URL-safe Base64 string (header, payload, or signature segment).

  Returns:
    The decoded raw byte sequence.
  """
  padding = "=" * (-len(segment) % 4)
  return base64.urlsafe_b64decode(segment + padding)


def verify_gateway_jwt(token: str) -> tuple[bool, dict | str]:
  """Cryptographically verifies an HS256 X-Gateway-Token JWT.

  Validates the 3-segment JWT format and JOSE header (`alg="HS256"`,
  `typ="JWT"`), recomputes the HMAC-SHA256 signature using `GATEWAY_JWT_SECRET`,
  and checks `iss="gateway_agent"`, `sub`, `aud="corporate_names_analytics_v2"`,
  `iat`, and `exp` against the current timestamp.

  Args:
    token: Raw `X-Gateway-Token` header value received on an incoming HTTP request.

  Returns:
    A tuple of `(True, claims_dict)` if valid, or `(False, error_message)` if
    missing, tampered with, wrong audience/issuer, or expired.
  """
  if not token or token.count(".") != 2:
    return False, "Missing or malformed X-Gateway-Token JWT."

  # Split the JWT into `<header>.<payload>.<signature>` and verify JOSE header (`alg`, `typ`) per RFC 8725.
  header_b64, payload_b64, signature_b64 = token.split(".")
  try:
    header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
  except Exception:
    return False, "Invalid JSON header in X-Gateway-Token."

  if header.get("alg") != "HS256" or header.get("typ") != "JWT":
    return (
        False,
        f"Unsupported JWT header alg/typ in X-Gateway-Token: alg={header.get('alg')!r}.",
    )

  signing_input = f"{header_b64}.{payload_b64}"
  expected_sig = hmac.new(
      GATEWAY_JWT_SECRET.encode("utf-8"),
      signing_input.encode("ascii"),
      hashlib.sha256,
  ).digest()

  try:
    actual_sig = _b64url_decode(signature_b64)
  except Exception:
    return False, "Invalid base64url signature encoding in X-Gateway-Token."

  # Compare signatures in constant time to prevent timing side-channel attacks.
  if not hmac.compare_digest(expected_sig, actual_sig):
    return False, "Invalid JWT signature on X-Gateway-Token."

  try:
    claims = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
  except Exception:
    return False, "Invalid JSON payload in X-Gateway-Token."

  # Verify standard JWT claims: issuer (`iss`), subject (`sub`), audience (`aud`), issued-at (`iat`), and expiration (`exp`).
  if claims.get("iss") != "gateway_agent":
    return False, f"Invalid JWT issuer: {claims.get('iss')}"
  if not claims.get("sub"):
    return False, "Missing JWT subject ('sub') claim in X-Gateway-Token."
  if claims.get("aud") != EXPECTED_AUDIENCE:
    return (
        False,
        f"Invalid JWT audience '{claims.get('aud')}' (expected '{EXPECTED_AUDIENCE}').",
    )
  now = int(time.time())
  if int(claims.get("iat", 0)) > now + 30:
    return False, "X-Gateway-Token JWT 'iat' timestamp is in the future."
  if int(claims.get("exp", 0)) < now:
    return False, "X-Gateway-Token JWT has expired."

  return True, claims


_WEIGHTED_FIRST_NAMES = [
    ("Alex", 145), ("Michael", 132), ("David", 124), ("Sarah", 118), ("Elena", 112),
    ("Daniel", 106), ("Jessica", 98), ("Marcus", 94), ("Emily", 90), ("Priya", 86),
    ("John", 82), ("Sophia", 78), ("Liam", 74), ("Olivia", 71), ("Noah", 68),
    ("Ethan", 64), ("Matthew", 61), ("Ava", 58), ("Lucas", 55), ("Isabella", 52),
    ("Sergio", 46), ("Pedro", 42), ("Mia", 40), ("Charlotte", 38), ("Mason", 36),
    ("Logan", 35), ("Amelia", 34), ("Harper", 33), ("Aiden", 32), ("Oliver", 31),
    ("Jacob", 30), ("Abigail", 29), ("Elizabeth", 28), ("Sofia", 27), ("Sebastian", 26),
    ("Avery", 25), ("Ella", 24), ("Samuel", 23), ("Scarlett", 22), ("Grace", 22),
    ("Joseph", 21), ("Chloe", 21), ("Victoria", 20), ("Riley", 20), ("Aria", 19),
    ("Lily", 19), ("Jack", 18), ("Hannah", 18), ("Ryan", 18), ("Nathan", 17),
    ("Lavi", 16), ("Sirisha", 14), ("Kevin", 28), ("Rachel", 26), ("Jason", 25),
    ("Megan", 24), ("Justin", 23), ("Lauren", 23), ("Brandon", 22), ("Amanda", 22),
    ("Tyler", 21), ("Kayla", 21), ("Zachary", 20), ("Natalie", 20), ("Kyle", 19),
    ("Brianna", 19), ("Aaron", 19), ("Haley", 18), ("Jeremy", 18), ("Alexis", 18),
    ("Eric", 24), ("Michelle", 23), ("Stephen", 20), ("Amber", 17), ("Brian", 25),
    ("Melissa", 22), ("Timothy", 19), ("Rebecca", 21), ("Carlos", 24), ("Diego", 21),
    ("Mateo", 20), ("Camila", 22), ("Valentina", 19), ("Lucia", 18), ("Arjun", 24),
    ("Rohan", 22), ("Ananya", 21), ("Divya", 19), ("Vikram", 18), ("Neha", 18),
    ("Wei", 23), ("Mei", 21), ("Kenji", 17), ("Yuki", 17), ("Min", 18),
    ("Hana", 18), ("Tariq", 16), ("Fatima", 17), ("Nadia", 16), ("Omar", 17),
]

_LAST_NAMES = [
    "Rivera", "Melendez", "Villani", "Nigam", "Karra", "Vance", "Sterling",
    "Patel", "Chen", "Kim", "Silva", "Nguyen", "Garcia", "Martinez", "Robinson",
    "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young",
    "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Adams", "Baker",
    "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner",
    "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart",
]

_DEPARTMENTS = [
    "AI Research & Foundation Models",
    "Cloud Platform Engineering",
    "Security & Zero-Trust Infrastructure",
    "Data & Analytics Platform",
    "Developer Relations & Ecosystem",
    "Product Management & UX",
    "Site Reliability Engineering",
    "Global Enterprise Solutions",
]

_OFFICE_HUBS = [
    "Sunnyvale", "Mountain View", "New York", "Seattle", "Austin",
    "London", "Zurich", "Bangalore", "Tokyo", "Sao Paulo",
]

_ROLES = [
    "Software Engineer",
    "Senior Software Engineer",
    "Staff AI Systems Engineer",
    "Principal Architect",
    "Data Scientist",
    "AI Developer Relations Engineer",
    "Security Engineer",
    "Product Manager",
    "Site Reliability Engineer",
]


def _init_corporate_sqlite_db() -> sqlite3.Connection:
  """Initializes an in-memory SQLite table (`corp_employees`) with 5,000 employees.

  Seeds a deterministic pseudo-random generator (`random.Random(2026)`) so the
  5,000 employee records, department distributions, and office hubs are 100%
  reproducible across runs.

  Returns:
    An open in-memory `sqlite3.Connection` populated with `corp_employees`.
  """
  conn = sqlite3.connect(":memory:", check_same_thread=False)
  conn.row_factory = sqlite3.Row
  cur = conn.cursor()
  cur.execute(
      """
      CREATE TABLE corp_employees (
          employee_id TEXT PRIMARY KEY,
          first_name TEXT NOT NULL,
          last_name TEXT NOT NULL,
          full_name TEXT NOT NULL,
          department TEXT NOT NULL,
          office_hub TEXT NOT NULL,
          role TEXT NOT NULL,
          hire_year INTEGER NOT NULL
      )
      """
  )
  cur.execute(
      "CREATE INDEX idx_corp_first_name ON corp_employees(LOWER(first_name))"
  )

  # Deterministically generate 5,000 employee rows using weighted first-name frequencies.
  rng = random.Random(2026)
  names_pool = [n.capitalize() for n, _ in _WEIGHTED_FIRST_NAMES]
  weights_pool = [w for _, w in _WEIGHTED_FIRST_NAMES]
  rows = []
  for idx in range(1, TOTAL_EMPLOYEES + 1):
    fname = rng.choices(names_pool, weights=weights_pool, k=1)[0]
    lname = rng.choice(_LAST_NAMES)
    rows.append((
        f"EMP-{idx:05d}",
        fname,
        lname,
        f"{fname} {lname}",
        rng.choice(_DEPARTMENTS),
        rng.choice(_OFFICE_HUBS),
        rng.choice(_ROLES),
        rng.randint(2014, 2026),
    ))

  cur.executemany(
      "INSERT INTO corp_employees VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      rows,
  )
  conn.commit()
  return conn


_DB_CONN = _init_corporate_sqlite_db()


def query_corporate_name_analytics(first_name: str = "") -> dict:
  """Runs SQL analytics against the Corporate Employee Directory (`corp_employees`).

  Args:
    first_name: Optional first name to analyze inside the corporate directory
      (e.g. 'Alex', 'John', 'Elena'). Leave empty to retrieve the company-wide
      Top 10 names.

  Returns:
    A dictionary containing `total_corporate_employees`, `top_10_corporate_names`,
    and (if `first_name` was provided) `queried_first_name` breakdown stats.
  """
  cur = _DB_CONN.cursor()
  cur.execute("SELECT COUNT(*) AS total FROM corp_employees")
  total_employees = int(cur.fetchone()["total"])

  cur.execute(
      """
      SELECT first_name, COUNT(*) AS employee_count
      FROM corp_employees
      GROUP BY first_name
      ORDER BY employee_count DESC, first_name ASC
      """
  )
  all_ranked = cur.fetchall()
  top_10_corporate_names = [
      {
          "corporate_rank": idx,
          "first_name": row["first_name"],
          "employee_count": int(row["employee_count"]),
          "percent_of_workforce": round(
              (int(row["employee_count"]) / total_employees) * 100.0, 2
          ),
      }
      for idx, row in enumerate(all_ranked[:10], start=1)
  ]

  result: dict = {
      "agent": "corporate_names_analytics_agent (Internal ID: corporate_names_analytics_v2)",
      "total_corporate_employees": total_employees,
      "top_10_corporate_names": top_10_corporate_names,
  }

  # If a specific `first_name` was requested, compute its rank, departments, and office hubs.
  clean_first = (
      first_name.strip().split()[0].capitalize() if first_name.strip() else ""
  )
  if clean_first:
    corp_rank = None
    emp_count = 0
    for idx, row in enumerate(all_ranked, start=1):
      if row["first_name"].lower() == clean_first.lower():
        corp_rank = idx
        emp_count = int(row["employee_count"])
        break

    cur.execute(
        """
        SELECT department, COUNT(*) AS cnt FROM corp_employees
        WHERE LOWER(first_name) = LOWER(?)
        GROUP BY department ORDER BY cnt DESC, department ASC LIMIT 3
        """,
        (clean_first,),
    )
    top_departments = [
        {"department": r["department"], "count": int(r["cnt"])}
        for r in cur.fetchall()
    ]

    cur.execute(
        """
        SELECT office_hub, COUNT(*) AS cnt FROM corp_employees
        WHERE LOWER(first_name) = LOWER(?)
        GROUP BY office_hub ORDER BY cnt DESC, office_hub ASC LIMIT 3
        """,
        (clean_first,),
    )
    top_office_hubs = [
        {"office_hub": r["office_hub"], "count": int(r["cnt"])}
        for r in cur.fetchall()
    ]

    result["queried_first_name"] = {
        "first_name": clean_first,
        "corporate_employee_count": emp_count,
        "total_corporate_employees": total_employees,
        "corporate_rank": (
            corp_rank if corp_rank is not None else "Not in directory (0 matches)"
        ),
        "percent_of_workforce": round((emp_count / total_employees) * 100.0, 2),
        "top_departments": top_departments,
        "top_office_hubs": top_office_hubs,
    }

  return result


corporate_names_analytics_agent = Agent(
    name="corporate_names_analytics_v2",
    model=Gemini(
        model=os.getenv("MODEL_NAME", "gemini-3.6-flash"),
        retry_options=RETRY_OPTIONS,
    ),
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(retry_options=RETRY_OPTIONS)
    ),
    description=(
        "Corporate Names Analytics Agent (`corporate_names_analytics_agent` /"
        " internal ID `corporate_names_analytics_v2`): runs SQL analytics over"
        " the 5,000-employee Corporate Employee Directory (`corp_employees`)."
    ),
    instruction="""
    You are the Corporate Names Analytics Agent (`corporate_names_analytics_agent`, internal registry ID `corporate_names_analytics_v2`).
    You own the Corporate Employee Directory across global offices and engineering departments.

    - Use `query_corporate_name_analytics(first_name=...)` whenever a caller asks for corporate name analytics, how many employees share a given first name in the company, or the Top 10 most common names in the corporate directory.
    """,
    tools=[query_corporate_name_analytics],
)

a2a_app = to_a2a(
    corporate_names_analytics_agent, port=int(os.getenv("PORT", "8002"))
)


async def verify_gateway_jwt_and_rewrite_card(request: Request, call_next):
  """Enforces Zero-Trust `X-Gateway-Token` JWT verification on all JSON-RPC `POST` requests.

  Allows unauthenticated `GET /.well-known/agent-card.json` discovery requests
  while enforcing strict HMAC-SHA256 `X-Gateway-Token` verification on every
  JSON-RPC 2.0 `POST` invocation.

  Args:
    request: Incoming Starlette HTTP `Request`.
    call_next: Next ASGI middleware or route handler in the Starlette pipeline.

  Returns:
    A `401 Unauthorized` `JSONResponse` if `X-Gateway-Token` is missing or
    invalid on a `POST` request, or the downstream HTTP `Response`.
  """
  if request.method == "POST":
    token = request.headers.get("x-gateway-token", "")
    is_valid, claims_or_err = verify_gateway_jwt(token)
    if not is_valid:
      return JSONResponse(
          {
              "error": f"401 Unauthorized: {claims_or_err}",
              "hint": (
                  "Request a signed X-Gateway-Token JWT from the Gateway Agent"
                  " using target_agent_name='corporate_names_analytics_v2'."
              ),
          },
          status_code=401,
      )
  return await call_next(request)


a2a_app.add_middleware(
    BaseHTTPMiddleware, dispatch=verify_gateway_jwt_and_rewrite_card
)
