"""Text and image rendering for the Adaptive Navigation Environment."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Colour palette  (R, G, B)  0-255
# ---------------------------------------------------------------------------
CELL_COLORS: dict[str, tuple[int, int, int]] = {
    "#": (60, 60, 60),       # wall   – dark grey
    ".": (240, 240, 240),    # empty  – near-white
    "A": (50, 120, 220),     # agent  – blue
    "K": (230, 190, 40),     # key    – gold
    "D": (160, 100, 50),     # door   – brown
    "G": (40, 200, 80),      # goal   – green
    "X": (220, 50, 50),      # hazard – red
}
_DEFAULT_COLOR = (180, 180, 180)
_PATH_HINT_COLOR = (140, 200, 255)  # light blue dot for path hints

LEGEND = """Adaptive Nav Environment — Legend

  G (green)   — Goal: your objective; reach this to win
  A (blue)    — Agent: the player you control
  K (gold)    — Key: collect this to unlock the door
  D (brown)   — Door: locked until you have the key
  # (grey)    — Wall: impassable obstacle
  . (white)   — Empty: walkable floor
  X (red)     — Hazard: avoid; costs energy or harms

  In debug mode: light blue dots mark path hints."""


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

def render_text(
    grid: np.ndarray,
    agent_pos: tuple[int, int],
    path_hints: dict[str, list[tuple[int, int]]] | None = None,
) -> str:
    """Return a multiline string representation of the grid.

    The agent symbol is overlaid at *agent_pos*.  When *path_hints* is
    provided, path cells are shown as '*' (debug breadcrumbs).
    """
    hint_cells: set[tuple[int, int]] = set()
    if path_hints:
        for cells in path_hints.values():
            hint_cells.update(cells)

    rows, cols = grid.shape
    lines: list[str] = []
    for r in range(rows):
        row_chars: list[str] = []
        for c in range(cols):
            if (r, c) == agent_pos:
                row_chars.append("A")
            elif (r, c) in hint_cells and grid[r, c] == ".":
                row_chars.append("*")
            else:
                row_chars.append(grid[r, c])
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RGB renderer (matplotlib-free, pure numpy for speed)
# ---------------------------------------------------------------------------

def render_rgb(
    grid: np.ndarray,
    agent_pos: tuple[int, int],
    cell_size: int = 40,
    path_hints: dict[str, list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Return an (H, W, 3) uint8 image of the grid.

    Each grid cell is rendered as a *cell_size* x *cell_size* coloured square
    with a 1-pixel dark border.  When *path_hints* is provided, hint cells
    get a small coloured dot in the centre.
    """
    hint_cells: set[tuple[int, int]] = set()
    if path_hints:
        for cells in path_hints.values():
            hint_cells.update(cells)

    rows, cols = grid.shape
    img_h = rows * cell_size
    img_w = cols * cell_size
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            sym = grid[r, c]
            if (r, c) == agent_pos:
                sym = "A"
            color = CELL_COLORS.get(sym, _DEFAULT_COLOR)

            y0 = r * cell_size
            x0 = c * cell_size

            img[y0:y0 + cell_size, x0:x0 + cell_size] = color

            # Draw path-hint dot on empty cells
            if (r, c) in hint_cells and sym == ".":
                dot_r = cell_size // 4
                cy, cx = y0 + cell_size // 2, x0 + cell_size // 2
                img[cy - dot_r:cy + dot_r, cx - dot_r:cx + dot_r] = _PATH_HINT_COLOR

            # 1-pixel border
            img[y0, x0:x0 + cell_size] = (30, 30, 30)
            img[y0 + cell_size - 1, x0:x0 + cell_size] = (30, 30, 30)
            img[y0:y0 + cell_size, x0] = (30, 30, 30)
            img[y0:y0 + cell_size, x0 + cell_size - 1] = (30, 30, 30)

    return img


def render_observation_rgb(
    local_view: np.ndarray,
    cell_size: int = 48,
) -> np.ndarray:
    """Render just the agent's local observation window as an RGB image."""
    rows, cols = local_view.shape
    img_h = rows * cell_size
    img_w = cols * cell_size
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    center = (rows // 2, cols // 2)

    for r in range(rows):
        for c in range(cols):
            sym = local_view[r, c]
            if (r, c) == center and sym != "A":
                sym = "A"
            color = CELL_COLORS.get(sym, _DEFAULT_COLOR)

            y0 = r * cell_size
            x0 = c * cell_size
            img[y0:y0 + cell_size, x0:x0 + cell_size] = color
            img[y0, x0:x0 + cell_size] = (30, 30, 30)
            img[y0 + cell_size - 1, x0:x0 + cell_size] = (30, 30, 30)
            img[y0:y0 + cell_size, x0] = (30, 30, 30)
            img[y0:y0 + cell_size, x0 + cell_size - 1] = (30, 30, 30)

    return img
