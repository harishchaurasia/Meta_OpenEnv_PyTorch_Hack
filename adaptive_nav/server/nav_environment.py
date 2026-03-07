"""OpenEnv Environment wrapper around the local AdaptiveNavEnv.

This module bridges the existing grid-world mechanics (adaptive_nav.env)
with the OpenEnv server interface so the environment can be deployed as an
HTTP/WebSocket service via ``openenv serve`` or ``openenv push``.

The local AdaptiveNavEnv is **not** modified -- this wrapper simply
translates between OpenEnv's Pydantic types and the local dict-based API.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from adaptive_nav.env import ACTION_NAMES, AdaptiveNavEnv
from adaptive_nav.models import NavAction, NavObservation, NavState

# ---------------------------------------------------------------------------
# OpenEnv base class (with fallback if openenv-core is not installed)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from abc import ABC, abstractmethod

    class Environment(ABC):  # type: ignore[no-redef]
        """Minimal stub matching the openenv.core.env_server.interfaces.Environment contract."""
        def __init__(self, transform=None, rubric=None):
            self.rubric = rubric

        @abstractmethod
        def reset(self, seed=None, episode_id=None, **kw): ...

        @abstractmethod
        def step(self, action, timeout_s=None, **kw): ...

        @property
        @abstractmethod
        def state(self): ...


class NavEnvironment(Environment):
    """OpenEnv-compatible wrapper for the Adaptive Navigation grid-world.

    Each WebSocket session gets its own instance (via the factory passed to
    ``create_app``), so state is fully isolated between clients.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        grid_size: int = 11,
        max_energy: int = 80,
        dynamic_changes: bool = True,
        dynamic_step: int = 25,
        view_radius: int = 2,
    ) -> None:
        super().__init__()
        self._env = AdaptiveNavEnv(
            grid_size=grid_size,
            max_energy=max_energy,
            dynamic_changes=dynamic_changes,
            dynamic_step=dynamic_step,
            view_radius=view_radius,
        )
        self._episode_id: str = ""
        self._seed: Optional[int] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> NavObservation:
        """Reset the grid-world and return the initial observation."""
        self._seed = seed
        self._episode_id = episode_id or str(uuid4())
        raw_obs = self._env.reset(seed=seed)
        return self._to_observation(raw_obs, reward=0.0)

    def step(
        self,
        action: NavAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> NavObservation:
        """Execute one action and return the resulting observation."""
        raw_obs, reward, done, info = self._env.step(action.action_id)
        return self._to_observation(raw_obs, reward=reward, info=info)

    @property
    def state(self) -> NavState:
        """Return a serialisable snapshot of internal state."""
        return NavState(
            episode_id=self._episode_id,
            step_count=self._env.step_count,
            has_key=self._env.has_key,
            door_unlocked=self._env.door_unlocked,
            energy=self._env.energy,
            won=self._env.won,
            done_reason=self._env.done_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_observation(
        self,
        raw: dict[str, Any],
        reward: float = 0.0,
        info: dict[str, Any] | None = None,
    ) -> NavObservation:
        """Convert the local dict observation to a Pydantic NavObservation."""
        info = info or {}
        local_view = raw["local_view"].tolist()
        return NavObservation(
            local_view=local_view,
            energy=raw["energy"],
            has_key=raw["has_key"],
            door_unlocked=raw["door_unlocked"],
            step_count=raw["step_count"],
            mission=raw.get("mission", {}),
            done=self._env.done,
            reward=reward,
            done_reason=self._env.done_reason,
            grid_text=self._env.render(mode="text"),
        )
