# Adaptive Navigation Environment

A partially observable 2D grid-world environment designed for training and evaluating LLM / RL agents on **long-horizon planning** and **world modeling**.

Built for the Meta OpenEnv PyTorch Hackathon.

---

## Problem Framing

Real-world navigation requires agents to:

1. **Explore** an unknown environment under partial observability.
2. **Plan** multi-step sequences (find key → unlock door → reach goal).
3. **Adapt** when the world changes mid-episode (corridors open/close).
4. **Manage resources** (limited energy budget).

This environment distils those challenges into a clean, configurable grid-world that is simple enough to iterate on quickly but rich enough to expose genuine planning failures.

---

## Environment Mechanics

| Element | Symbol | Description |
|---------|--------|-------------|
| Wall | `#` | Impassable terrain |
| Empty | `.` | Walkable cell |
| Agent | `A` | The player |
| Key | `K` | Must be collected to unlock the door |
| Door | `D` | Blocks the path to the goal until unlocked |
| Goal | `G` | Episode ends successfully when reached |
| Hazard | `X` | Penalty on contact (optional) |

The grid is generated with a randomized DFS maze algorithm, then key, door, and goal are placed with BFS-validated reachability guarantees.

---

## Action Space

| Index | Action | Effect |
|-------|--------|--------|
| 0 | Up | Move one cell up |
| 1 | Down | Move one cell down |
| 2 | Left | Move one cell left |
| 3 | Right | Move one cell right |
| 4 | Interact | Pick up key (if on key) / unlock door (if adjacent and key owned) |
| 5 | Wait | Do nothing (still costs energy and a step) |

---

## Observation Space

By default the agent receives a **partial observation**:

```python
{
    "local_view": np.ndarray,   # (2*r+1, 2*r+1) character grid centred on agent
    "energy": int,              # remaining energy
    "has_key": bool,            # whether the key has been collected
    "door_unlocked": bool,      # whether the door has been opened
    "step_count": int,          # steps taken so far
}
```

Pass `full=True` to `get_observation()` to also receive the entire grid (useful for debugging / oracle baselines).

---

## Reward Design

| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Pick up key | +20 |
| Unlock door | +30 |
| Each step | -1 |
| Bump into wall | -3 |
| Useless interact | -5 |
| Hit hazard | -20 |
| Completion bonus | +remaining_energy × 0.5 + (max_energy − steps) × 0.3 |

All values are defined in `REWARD_CONFIG` at the top of `adaptive_nav/env.py` for easy tuning.

---

## Dynamic World Changes

When `dynamic_changes=True` (default), the environment mutates the grid at a configurable step (default 25):

- One or two corridor cells near the door toggle between wall and empty.
- This can block a previously valid route or open a new shortcut.
- The agent must detect the change through its local observation and re-plan.

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the CLI demo

```bash
# Random-action episode
python demo.py

# Manual play (WASD + E to interact + Q to wait)
python demo.py --manual

# Reproducible seed
python demo.py --seed 42
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## Project Structure

```
adaptive_nav/
    __init__.py        # Package exports
    env.py             # AdaptiveNavEnv class + reward config
    generator.py       # Maze generation + dynamic changes
    renderer.py        # Text and RGB renderers
app.py                 # Streamlit interactive demo
demo.py                # CLI demo script
requirements.txt
README.md
```

---

## Environment API

```python
from adaptive_nav import AdaptiveNavEnv

env = AdaptiveNavEnv(grid_size=11, dynamic_changes=True)
obs = env.reset(seed=42)

obs, reward, done, info = env.step(action=0)  # move up
text = env.render(mode="text")                 # terminal string
img  = env.render(mode="rgb_array")            # numpy RGB image
```

---

## Tech Stack

- **Python 3.10+**
- **NumPy** – grid representation and rendering
- **Matplotlib** – optional visualisation
- **Streamlit** – interactive web demo

---

## License

MIT
