"""Pydantic models for the OpenEnv interface.

These dataclasses define the wire format that flows between the OpenEnv
server and any EnvClient (local, Docker, or HF Spaces).  They replace the
raw dicts used by the local-only AdaptiveNavEnv API.

Mapping to local env.py concepts
---------------------------------
NavAction.action_id  ->  the integer 0-5 passed to AdaptiveNavEnv.step()
NavObservation       ->  the dict returned by AdaptiveNavEnv.get_observation()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

# ---------------------------------------------------------------------------
# OpenEnv base types  (imported from openenv-core when available, otherwise
# we define compatible stubs so the project still runs without the package)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback stubs -- keep the same field contract so the rest of the code
    # works identically whether openenv-core is installed or not.
    from pydantic import BaseModel

    class Action(BaseModel):  # type: ignore[no-redef]
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class NavAction(Action):
    """An agent action in the Adaptive Navigation Environment.

    action_id mapping:
        0=up, 1=down, 2=left, 3=right, 4=interact, 5=wait
    """
    action_id: int = Field(
        ..., ge=0, le=5,
        description="Action index: 0=up 1=down 2=left 3=right 4=interact 5=wait",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class NavObservation(Observation):
    """What the agent sees after each step (or on reset)."""

    local_view: List[List[str]] = Field(
        ..., description="2-D partial-observability window around the agent",
    )
    energy: int = Field(..., description="Remaining energy budget")
    has_key: bool = Field(..., description="Whether the agent holds the key")
    door_unlocked: bool = Field(..., description="Whether the door is open")
    step_count: int = Field(0, description="Steps taken so far")
    mission: Dict[str, bool] = Field(
        default_factory=dict,
        description="Mission checklist: get_key / unlock_door / reach_goal",
    )
    done_reason: str = Field(
        "", description="Why the episode ended (reached_goal / out_of_energy / '')",
    )
    grid_text: str = Field(
        "", description="Full text render of the grid (included for readability)",
    )


# ---------------------------------------------------------------------------
# State  (internal server state -- not sent to agent by default)
# ---------------------------------------------------------------------------

class NavState(State):
    """Snapshot of server-side episode state."""
    has_key: bool = False
    door_unlocked: bool = False
    energy: int = 0
    won: bool = False
    done_reason: str = ""
