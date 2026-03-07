"""FastAPI application for the OpenEnv Adaptive Navigation Environment.

Launch locally with:
    uvicorn adaptive_nav.server.app:app --host 0.0.0.0 --port 8000

Or via the OpenEnv CLI:
    openenv serve
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Try the real OpenEnv create_app; fall back to a plain FastAPI stub so the
# project still *imports* without openenv-core installed.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import create_app

    from adaptive_nav.models import NavAction, NavObservation
    from adaptive_nav.server.nav_environment import NavEnvironment

    app = create_app(
        NavEnvironment,
        NavAction,
        NavObservation,
        env_name="adaptive_nav",
    )

except ImportError:
    # openenv-core not installed -- provide a minimal FastAPI app so the
    # local demo path (demo.py / streamlit app.py) keeps working and so
    # ``uvicorn adaptive_nav.server.app:app`` gives a useful error.
    from fastapi import FastAPI

    app = FastAPI(title="Adaptive Nav (openenv-core not installed)")

    @app.get("/")
    def _root():
        return {
            "error": "openenv-core is not installed. "
                     "Run: pip install 'openenv-core[core]>=0.2.1'"
        }
