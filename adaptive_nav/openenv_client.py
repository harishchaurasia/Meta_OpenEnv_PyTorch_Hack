"""OpenEnv client for the Adaptive Navigation Environment.

Usage (remote / Docker / HF Spaces)::

    from adaptive_nav.openenv_client import NavEnvClient
    from adaptive_nav.models import NavAction

    # Connect to a running server (local, Docker, or HF Space)
    client = NavEnvClient(base_url="http://localhost:8000")

    with client:
        result = client.reset()
        print(result.observation.local_view)

        result = client.step(NavAction(action_id=0))  # move up
        print(result.observation.energy, result.reward)
"""

from __future__ import annotations

from typing import Any

from adaptive_nav.models import NavAction, NavObservation, NavState

# ---------------------------------------------------------------------------
# OpenEnv EnvClient (with fallback stub)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class NavEnvClient(EnvClient[NavAction, NavObservation, NavState]):
        """Typed WebSocket client for the Adaptive Nav environment."""

        def _step_payload(self, action: NavAction) -> dict[str, Any]:
            return {"action_id": action.action_id}

        def _parse_result(self, payload: dict[str, Any]) -> StepResult[NavObservation]:
            obs_data = payload.get("observation", {})
            obs = NavObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict[str, Any]) -> NavState:
            return NavState(**payload)

except ImportError:
    # When openenv-core is not installed, provide a thin HTTP-based client
    # that works with the plain FastAPI fallback.  This keeps the project
    # functional for local experimentation without Docker.
    import json
    import urllib.request

    class _StepResult:
        """Minimal stand-in for openenv.core.client_types.StepResult."""
        def __init__(self, observation: NavObservation, reward: float | None, done: bool):
            self.observation = observation
            self.reward = reward
            self.done = done

    class NavEnvClient:  # type: ignore[no-redef]
        """Lightweight HTTP client that talks to the FastAPI fallback server."""

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        # context manager
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def close(self):
            pass

        def reset(self, seed: int | None = None) -> _StepResult:
            body = json.dumps({"seed": seed}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/reset", data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            obs = NavObservation(**resp.get("observation", resp))
            return _StepResult(obs, resp.get("reward", 0.0), resp.get("done", False))

        def step(self, action: NavAction) -> _StepResult:
            body = json.dumps({"action": {"action_id": action.action_id}}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/step", data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            obs = NavObservation(**resp.get("observation", resp))
            return _StepResult(obs, resp.get("reward", 0.0), resp.get("done", False))

        def state(self) -> NavState:
            resp = json.loads(urllib.request.urlopen(f"{self.base_url}/state").read())
            return NavState(**resp)

        @classmethod
        def from_hub(cls, repo_id: str):
            raise NotImplementedError(
                "from_hub requires openenv-core. "
                "Install it with: pip install 'openenv-core[core]>=0.2.1'"
            )
