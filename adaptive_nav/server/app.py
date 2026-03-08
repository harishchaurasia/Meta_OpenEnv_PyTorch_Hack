"""FastAPI application for the OpenEnv Adaptive Navigation Environment.

Launch locally with:
    uvicorn adaptive_nav.server.app:app --host 0.0.0.0 --port 8000

Or via the OpenEnv CLI:
    openenv serve
"""

from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

log = logging.getLogger("adaptive_nav.server")

# ---------------------------------------------------------------------------
# Landing page served at GET /
# ---------------------------------------------------------------------------

_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive Navigation Environment</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9;
         display: flex; justify-content: center; padding: 3rem 1rem; }
  .card { max-width: 560px; background: #161b22; border-radius: 12px;
          padding: 2rem 2.5rem; box-shadow: 0 4px 24px rgba(0,0,0,.4); }
  h1 { margin: 0 0 .25rem; font-size: 1.5rem; color: #58a6ff; }
  .tag { display: inline-block; font-size: .75rem; background: #1f6feb;
         color: #fff; padding: 2px 8px; border-radius: 4px; margin-bottom: 1rem; }
  p  { line-height: 1.6; font-size: .95rem; }
  table { width: 100%; border-collapse: collapse; margin-top: 1.25rem; font-size: .9rem; }
  th, td { text-align: left; padding: .5rem .75rem; }
  th { color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
  td { border-bottom: 1px solid #21262d; }
  code { background: #0d1117; padding: 2px 6px; border-radius: 4px; font-size: .85rem; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="card">
  <h1>Adaptive Navigation Environment</h1>
  <span class="tag">OpenEnv 0.2.1</span>
  <p>
    A partially observable 2D grid-world for evaluating LLM&nbsp;/&nbsp;RL agents
    on long-horizon planning, key-door-goal mechanics, and dynamic replanning.
  </p>
  <table>
    <tr><th>Endpoint</th><th>Description</th></tr>
    <tr><td><code><a href="/health">/health</a></code></td><td>Server health check</td></tr>
    <tr><td><code><a href="/schema">/schema</a></code></td><td>Action / observation JSON schema</td></tr>
    <tr><td><code><a href="/metadata">/metadata</a></code></td><td>Environment metadata</td></tr>
    <tr><td><code>/ws</code></td><td>WebSocket session (reset / step / state)</td></tr>
  </table>
  <p style="margin-top:1.5rem;font-size:.85rem;color:#8b949e;">
    Source:
    <a href="https://github.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack">GitHub</a>
  </p>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Build the app -- try OpenEnv first, fall back to plain FastAPI.
# Catches ALL exceptions (not just ImportError) so the container always
# starts and the landing page / health check remain reachable.
# ---------------------------------------------------------------------------
_startup_error: str | None = None

try:
    from openenv.core.env_server import create_app as _create_app

    from adaptive_nav.models import NavAction, NavObservation
    from adaptive_nav.server.nav_environment import NavEnvironment

    app = _create_app(
        NavEnvironment,
        NavAction,
        NavObservation,
        env_name="adaptive_nav",
    )
    log.info("OpenEnv app created successfully")

except Exception as exc:
    _startup_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    log.warning("OpenEnv create_app failed, using plain FastAPI fallback: %s", exc)

    app = FastAPI(title="Adaptive Nav Environment")


# ---------------------------------------------------------------------------
# Routes that are always available regardless of OpenEnv status
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    return _LANDING_HTML


@app.get("/health")
def health():
    return {"status": "ok", "openenv": _startup_error is None}


if _startup_error:
    @app.get("/debug/startup-error")
    def startup_error():
        return {"error": _startup_error}
