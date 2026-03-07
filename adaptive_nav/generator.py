"""Maze generation and dynamic world mutation for the Adaptive Navigation Environment."""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Cell symbols
# ---------------------------------------------------------------------------
WALL = "#"
EMPTY = "."
AGENT = "A"
KEY = "K"
DOOR = "D"
GOAL = "G"
HAZARD = "X"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bfs_reachable(grid: np.ndarray, start: tuple[int, int],
                   target: tuple[int, int],
                   passable: set[str] | None = None) -> bool:
    """Return True if *target* is reachable from *start* on *grid*.

    *passable* defaults to {EMPTY, KEY, DOOR, GOAL, AGENT}.
    """
    if passable is None:
        passable = {EMPTY, KEY, DOOR, GOAL, AGENT}
    rows, cols = grid.shape
    visited = set()
    queue: deque[tuple[int, int]] = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == target:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr, nc] in passable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def _bfs_path(grid: np.ndarray, start: tuple[int, int],
              target: tuple[int, int],
              passable: set[str] | None = None) -> list[tuple[int, int]]:
    """Return shortest path (list of cells) from *start* to *target*, or []."""
    if passable is None:
        passable = {EMPTY, KEY, DOOR, GOAL, AGENT}
    rows, cols = grid.shape
    visited: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == target:
            path = []
            cur: tuple[int, int] | None = (r, c)
            while cur is not None:
                path.append(cur)
                cur = visited[cur]
            return path[::-1]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr, nc] in passable:
                    visited[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
    return []


def _open_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return coordinates of all EMPTY cells."""
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == EMPTY]


# ---------------------------------------------------------------------------
# Maze carving  (randomized DFS on odd-indexed sub-grid)
# ---------------------------------------------------------------------------

def _carve_maze(grid: np.ndarray, rng: random.Random) -> None:
    """In-place carve corridors through an all-wall grid using randomized DFS.

    Works on odd-indexed rows/cols to guarantee walls between corridors.
    """
    rows, cols = grid.shape
    start_r, start_c = 1, 1
    grid[start_r, start_c] = EMPTY
    stack = [(start_r, start_c)]

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr, nc] == WALL:
                neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
        if neighbors:
            nr, nc, wr, wc = rng.choice(neighbors)
            grid[wr, wc] = EMPTY
            grid[nr, nc] = EMPTY
            stack.append((nr, nc))
        else:
            stack.pop()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_grid(
    size: int = 10,
    seed: Optional[int] = None,
    max_retries: int = 50,
) -> dict:
    """Generate a solvable grid with agent, key, door, goal, and dynamic cells.

    Returns a dict with keys:
        grid, agent_pos, key_pos, door_pos, goal_pos, dynamic_cells
    """
    # Ensure odd internal dimensions for clean maze carving
    if size % 2 == 0:
        size += 1

    rng = random.Random(seed)

    for _attempt in range(max_retries):
        grid = np.full((size, size), WALL, dtype="U1")
        _carve_maze(grid, rng)

        opens = _open_cells(grid)
        if len(opens) < 10:
            continue

        # --- place agent in top-left quadrant ---
        tl = [(r, c) for r, c in opens if r < size // 2 and c < size // 2]
        if not tl:
            tl = opens[:len(opens) // 4] or opens
        agent_pos = rng.choice(tl)
        grid[agent_pos] = AGENT

        # --- place key roughly mid-grid, away from agent ---
        mid = [(r, c) for r, c in opens
               if abs(r - agent_pos[0]) + abs(c - agent_pos[1]) > size // 3
               and grid[r, c] == EMPTY]
        if not mid:
            mid = [p for p in opens if grid[p[0], p[1]] == EMPTY]
        if not mid:
            continue
        key_pos = rng.choice(mid)
        grid[key_pos] = KEY

        # --- place door: on a corridor cell that separates key-region from goal-region ---
        br = [(r, c) for r, c in opens
              if r > size // 2 and c > size // 2 and grid[r, c] == EMPTY]
        if not br:
            br = [(r, c) for r, c in opens
                  if r >= size // 2 and grid[r, c] == EMPTY]
        if not br:
            continue

        # Pick door as a bottleneck between key-area and goal-area
        door_candidates = [(r, c) for r, c in opens
                           if size // 3 <= r <= 2 * size // 3
                           and grid[r, c] == EMPTY]
        if not door_candidates:
            door_candidates = [p for p in opens if grid[p[0], p[1]] == EMPTY]
        if not door_candidates:
            continue
        door_pos = rng.choice(door_candidates)
        grid[door_pos] = DOOR

        # --- place goal in bottom-right quadrant ---
        goal_candidates = [(r, c) for r, c in br if grid[r, c] == EMPTY]
        if not goal_candidates:
            continue
        goal_pos = rng.choice(goal_candidates)
        grid[goal_pos] = GOAL

        # --- validate reachability ---
        passable_all = {EMPTY, KEY, DOOR, GOAL, AGENT}
        if not _bfs_reachable(grid, agent_pos, key_pos, passable_all):
            continue
        if not _bfs_reachable(grid, key_pos, door_pos, passable_all):
            continue
        if not _bfs_reachable(grid, door_pos, goal_pos, passable_all):
            continue

        # --- designate dynamic cells (corridor cells near the door) ---
        dynamic_cells: list[tuple[int, int]] = []
        dr, dc = door_pos
        for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                  (-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = dr + delta_r, dc + delta_c
            if (0 < nr < size - 1 and 0 < nc < size - 1
                    and grid[nr, nc] == EMPTY
                    and (nr, nc) != agent_pos):
                dynamic_cells.append((nr, nc))
                if len(dynamic_cells) >= 2:
                    break

        return {
            "grid": grid,
            "agent_pos": agent_pos,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "goal_pos": goal_pos,
            "dynamic_cells": dynamic_cells,
        }

    # Fallback: hand-crafted small grid (should rarely trigger)
    return _fallback_grid(size)


def _fallback_grid(size: int) -> dict:
    """A guaranteed-solvable hand-crafted layout used if generation retries exhaust."""
    grid = np.full((size, size), WALL, dtype="U1")
    # Carve an L-shaped corridor
    for c in range(1, size - 1):
        grid[1, c] = EMPTY
    for r in range(1, size - 1):
        grid[r, size - 2] = EMPTY
    for c in range(1, size - 1):
        grid[size - 2, c] = EMPTY

    agent_pos = (1, 1)
    key_pos = (1, size - 3)
    door_pos = (size // 2, size - 2)
    goal_pos = (size - 2, size - 2)
    dynamic_cells = [(size // 2 - 1, size - 2)]

    grid[agent_pos] = AGENT
    grid[key_pos] = KEY
    grid[door_pos] = DOOR
    grid[goal_pos] = GOAL

    return {
        "grid": grid,
        "agent_pos": agent_pos,
        "key_pos": key_pos,
        "door_pos": door_pos,
        "goal_pos": goal_pos,
        "dynamic_cells": dynamic_cells,
    }


def generate_easy_grid() -> dict:
    """A small 7x7 hand-crafted map with an obvious solvable path.

    Layout (read top-to-bottom, left-to-right):
        # # # # # # #
        # A . . . . #
        # # # # . # #
        # . . K . . #
        # . # D # . #
        # . . . . G #
        # # # # # # #
    """
    grid = np.array([
        ["#", "#", "#", "#", "#", "#", "#"],
        ["#", "A", ".", ".", ".", ".", "#"],
        ["#", "#", "#", "#", ".", "#", "#"],
        ["#", ".", ".", "K", ".", ".", "#"],
        ["#", ".", "#", "D", "#", ".", "#"],
        ["#", ".", ".", ".", ".", "G", "#"],
        ["#", "#", "#", "#", "#", "#", "#"],
    ], dtype="U1")

    return {
        "grid": grid,
        "agent_pos": (1, 1),
        "key_pos": (3, 3),
        "door_pos": (4, 3),
        "goal_pos": (5, 5),
        "dynamic_cells": [(2, 4)],
    }


def apply_dynamic_change(grid: np.ndarray,
                         dynamic_cells: list[tuple[int, int]]) -> str:
    """Toggle dynamic cells between EMPTY and WALL.

    Returns a human-readable description of what changed.
    """
    changes: list[str] = []
    for r, c in dynamic_cells:
        if grid[r, c] == EMPTY:
            grid[r, c] = WALL
            changes.append(f"({r},{c}) blocked")
        elif grid[r, c] == WALL:
            grid[r, c] = EMPTY
            changes.append(f"({r},{c}) opened")
    return "Dynamic change: " + ", ".join(changes) if changes else "No dynamic change"
