"""Core environment for the Adaptive Navigation Environment."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from adaptive_nav.generator import (
    AGENT, DOOR, EMPTY, GOAL, HAZARD, KEY, WALL,
    apply_dynamic_change, generate_grid, generate_easy_grid,
    _bfs_path,
)
from adaptive_nav.renderer import render_rgb, render_text

# ---------------------------------------------------------------------------
# Reward configuration -- edit these to tune behaviour
# ---------------------------------------------------------------------------
REWARD_CONFIG: dict[str, float] = {
    "goal": 100.0,
    "pick_key": 20.0,
    "unlock_door": 30.0,
    "step": -1.0,
    "wall_bump": -2.0,
    "useless_interact": -2.0,
    "hazard": -20.0,
    "energy_bonus_factor": 0.5,
    "speed_bonus_factor": 0.3,
}

# Action constants
UP, DOWN, LEFT, RIGHT, INTERACT, WAIT = range(6)
ACTION_NAMES = ["up", "down", "left", "right", "interact", "wait"]
_MOVE_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class AdaptiveNavEnv:
    """Partially-observable 2D grid-world with key-door-goal mechanics.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (will be rounded up to odd).
    view_radius : int
        Radius of the agent's local observation window.
    max_energy : int
        Starting energy budget; the episode ends when it hits 0.
    dynamic_changes : bool
        Whether to apply a world mutation mid-episode.
    dynamic_step : int
        The step number at which the dynamic change triggers.
    seed : int or None
        RNG seed for reproducible grid generation.
    debug : bool
        When True, observations include the full grid and path hints.
    easy : bool
        When True, use a small hand-crafted easy map instead of random gen.
    """

    def __init__(
        self,
        grid_size: int = 10,
        view_radius: int = 2,
        max_energy: int = 80,
        dynamic_changes: bool = True,
        dynamic_step: int = 25,
        seed: Optional[int] = None,
        debug: bool = False,
        easy: bool = False,
    ) -> None:
        self.grid_size = grid_size
        self.view_radius = view_radius
        self.max_energy = max_energy
        self.dynamic_changes = dynamic_changes
        self.dynamic_step = dynamic_step
        self._init_seed = seed
        self.debug = debug
        self.easy = easy

        # State (populated by reset)
        self.grid: np.ndarray = np.array([])
        self.agent_pos: tuple[int, int] = (0, 0)
        self.has_key: bool = False
        self.door_unlocked: bool = False
        self.energy: int = max_energy
        self.step_count: int = 0
        self.done: bool = False
        self.won: bool = False
        self.done_reason: str = ""
        self.dynamic_applied: bool = False
        self.dynamic_cells: list[tuple[int, int]] = []
        self.total_reward: float = 0.0
        self._last_dynamic_msg: str = ""
        self._key_pos: tuple[int, int] = (0, 0)
        self._door_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset environment to a fresh episode and return initial observation."""
        use_seed = seed if seed is not None else self._init_seed
        if self.easy:
            data = generate_easy_grid()
        else:
            data = generate_grid(self.grid_size, seed=use_seed)

        self.grid = data["grid"]
        self.agent_pos = data["agent_pos"]
        self._key_pos = data["key_pos"]
        self._door_pos = data["door_pos"]
        self._goal_pos = data["goal_pos"]
        self.has_key = False
        self.door_unlocked = False
        self.energy = self.max_energy
        self.step_count = 0
        self.done = False
        self.won = False
        self.done_reason = ""
        self.dynamic_applied = False
        self.dynamic_cells = data["dynamic_cells"]
        self.total_reward = 0.0
        self._last_dynamic_msg = ""

        return self.get_observation()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute *action* and return (observation, reward, done, info)."""
        if self.done:
            return self.get_observation(), 0.0, True, {
                "msg": "Episode already finished.",
                "done_reason": self.done_reason,
            }

        reward = REWARD_CONFIG["step"]
        info: dict[str, Any] = {"action": ACTION_NAMES[action]}

        # --- Dynamic world change ---
        if (self.dynamic_changes
                and not self.dynamic_applied
                and self.step_count == self.dynamic_step):
            msg = apply_dynamic_change(self.grid, self.dynamic_cells)
            self.dynamic_applied = True
            self._last_dynamic_msg = msg
            info["dynamic_event"] = msg

        # --- Process action ---
        if action in _MOVE_DELTAS:
            reward += self._handle_move(action, info)
        elif action == INTERACT:
            reward += self._handle_interact(info)
        elif action == WAIT:
            info["msg"] = "Waited."
        else:
            info["msg"] = f"Unknown action {action}"

        self.energy -= 1
        self.step_count += 1

        # --- Check termination ---
        if self.energy <= 0 and not self.done:
            self.done = True
            self.done_reason = "out_of_energy"
            info["msg"] = info.get("msg", "") + " Out of energy!"

        info["done_reason"] = self.done_reason
        self.total_reward += reward
        return self.get_observation(), reward, self.done, info

    def get_observation(self, full: bool = False) -> dict[str, Any]:
        """Return the agent's observation.

        By default only a local window is visible (partial observability).
        Set *full=True* or enable *debug* mode to also include the full grid.
        """
        r, c = self.agent_pos
        rad = self.view_radius
        rows, cols = self.grid.shape

        # Extract local view, padding out-of-bounds with walls
        local = np.full((2 * rad + 1, 2 * rad + 1), WALL, dtype="U1")
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    local[dr + rad, dc + rad] = self.grid[nr, nc]

        obs: dict[str, Any] = {
            "local_view": local,
            "energy": self.energy,
            "has_key": self.has_key,
            "door_unlocked": self.door_unlocked,
            "step_count": self.step_count,
            "mission": self.mission_status(),
        }
        if full or self.debug:
            obs["full_grid"] = self.grid.copy()
            obs["path_hints"] = self.get_path_hints()
        return obs

    def mission_status(self) -> dict[str, bool]:
        """Return a checklist of the three mission objectives."""
        return {
            "get_key": self.has_key,
            "unlock_door": self.door_unlocked,
            "reach_goal": self.won,
        }

    def get_path_hints(self) -> dict[str, list[tuple[int, int]]]:
        """Compute shortest paths for debug display.

        Returns paths from agent to the next logical objective.
        """
        passable = {EMPTY, KEY, DOOR, GOAL, AGENT}
        hints: dict[str, list[tuple[int, int]]] = {}
        if not self.has_key:
            hints["to_key"] = _bfs_path(self.grid, self.agent_pos, self._key_pos, passable)
        elif not self.door_unlocked:
            hints["to_door"] = _bfs_path(self.grid, self.agent_pos, self._door_pos, passable)
        else:
            hints["to_goal"] = _bfs_path(self.grid, self.agent_pos, self._goal_pos, passable)
        return hints

    def render(self, mode: str = "text") -> Any:
        """Render the environment.

        mode='text'      -> returns a multi-line string.
        mode='rgb_array' -> returns an (H, W, 3) uint8 numpy array.
        """
        if mode == "text":
            return render_text(self.grid, self.agent_pos)
        if mode == "rgb_array":
            return render_rgb(self.grid, self.agent_pos)
        raise ValueError(f"Unsupported render mode: {mode!r}")

    def is_done(self) -> bool:
        """Return whether the episode has ended."""
        return self.done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_move(self, action: int, info: dict[str, Any]) -> float:
        dr, dc = _MOVE_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        rows, cols = self.grid.shape

        if not (0 <= nr < rows and 0 <= nc < cols):
            info["msg"] = "Bumped boundary."
            return REWARD_CONFIG["wall_bump"]

        target = self.grid[nr, nc]

        if target == WALL:
            info["msg"] = "Bumped wall."
            return REWARD_CONFIG["wall_bump"]

        if target == HAZARD:
            info["msg"] = "Hit hazard!"
            return REWARD_CONFIG["hazard"]

        if target == DOOR and not self.door_unlocked:
            info["msg"] = "Door is locked."
            return REWARD_CONFIG["wall_bump"]

        # Valid move
        self.grid[self.agent_pos] = EMPTY
        self.agent_pos = (nr, nc)

        if target == GOAL:
            self.done = True
            self.won = True
            self.done_reason = "reached_goal"
            bonus = (self.energy * REWARD_CONFIG["energy_bonus_factor"]
                     + (self.max_energy - self.step_count) * REWARD_CONFIG["speed_bonus_factor"])
            info["msg"] = "Reached the goal!"
            self.grid[self.agent_pos] = AGENT
            return REWARD_CONFIG["goal"] + bonus

        if target == KEY:
            self.has_key = True
            info["msg"] = "Picked up key (walked over)."
            self.grid[self.agent_pos] = AGENT
            return REWARD_CONFIG["pick_key"]

        self.grid[self.agent_pos] = AGENT
        return 0.0

    def _handle_interact(self, info: dict[str, Any]) -> float:
        r, c = self.agent_pos

        # Standing on key -> pick up
        if self.grid[r, c] == KEY:
            self.has_key = True
            self.grid[r, c] = AGENT
            info["msg"] = "Picked up key!"
            return REWARD_CONFIG["pick_key"]

        # Adjacent to locked door and has key -> unlock
        if self.has_key:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]
                        and self.grid[nr, nc] == DOOR and not self.door_unlocked):
                    self.door_unlocked = True
                    self.grid[nr, nc] = EMPTY
                    info["msg"] = "Unlocked the door!"
                    return REWARD_CONFIG["unlock_door"]

        info["msg"] = "Nothing to interact with."
        return REWARD_CONFIG["useless_interact"]
