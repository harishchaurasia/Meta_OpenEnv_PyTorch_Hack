# Adaptive Navigation Environment

A partially observable 2D grid-world environment designed for training and evaluating LLM / RL agents on **long-horizon planning** and **world modeling**.

Built for the **Meta OpenEnv PyTorch Hackathon** -- fully compatible with [OpenEnv 0.2.1](https://github.com/meta-pytorch/OpenEnv).

---

## Problem Framing

Real-world navigation requires agents to:

1. **Explore** an unknown environment under partial observability.
2. **Plan** multi-step sequences (find key -> unlock door -> reach goal).
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
    "local_view": list[list[str]],  # (2*r+1, 2*r+1) character grid centred on agent
    "energy": int,                  # remaining energy
    "has_key": bool,                # whether the key has been collected
    "door_unlocked": bool,          # whether the door has been opened
    "step_count": int,              # steps taken so far
    "mission": {"get_key": bool, "unlock_door": bool, "reach_goal": bool},
}
```

Pass `full=True` to `get_observation()` or enable `debug=True` to also receive the full grid and path hints.

---

## Reward Design

| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Pick up key | +20 |
| Unlock door | +30 |
| Each step | -1 |
| Bump into wall | -2 |
| Useless interact | -2 |
| Hit hazard | -20 |
| Completion bonus | +remaining_energy x 0.5 + (max_energy - steps) x 0.3 |

All values are defined in `REWARD_CONFIG` at the top of `adaptive_nav/env.py` for easy tuning.

---

## Dynamic World Changes

When `dynamic_changes=True` (default), the environment mutates the grid at a configurable step (default 25):

- One or two corridor cells near the door toggle between wall and empty.
- This can block a previously valid route or open a new shortcut.
- The agent must detect the change through its local observation and re-plan.

---

## Quick Start (Local)

```bash
pip install -r requirements.txt

# Random-action episode
python demo.py

# Manual play (WASD + E to interact + Q to wait)
python demo.py --manual

# Easy 7x7 map with debug path hints
python demo.py --easy --debug

# Streamlit interactive app
streamlit run app.py
```

---

## OpenEnv Deployment

This project is structured as an [OpenEnv](https://meta-pytorch.org/OpenEnv/) environment and can be served as an HTTP/WebSocket service or deployed to Hugging Face Spaces.

### Install OpenEnv deps

```bash
pip install "openenv-core[core]>=0.2.1" uvicorn fastapi
```

### Run the OpenEnv server locally

```bash
uvicorn adaptive_nav.server.app:app --host 0.0.0.0 --port 8000
```

### Connect with the client

```python
from adaptive_nav.openenv_client import NavEnvClient
from adaptive_nav.models import NavAction

client = NavEnvClient(base_url="http://localhost:8000")

with client:
    result = client.reset()
    print(result.observation.local_view)

    result = client.step(NavAction(action_id=0))   # move up
    print(result.observation.energy, result.reward)
```

### Build Docker image

```bash
docker build -f adaptive_nav/server/Dockerfile -t adaptive-nav:latest .
docker run -p 8000:8000 adaptive-nav:latest
```

### Deploy to Hugging Face Spaces

```bash
openenv validate
openenv push --repo-id YOUR_USERNAME/adaptive-nav
```

---

## TRL Training Skeleton

A minimal training script using Hugging Face TRL's `GRPOTrainer` is provided at `train_skeleton.py`. It demonstrates:

1. Connecting to the OpenEnv server as a reward source.
2. Custom `rollout_func` that generates completions and steps through the env.
3. Extracting environment rewards via `kwargs` in the reward function.

```bash
# Terminal 1: Start the environment server
uvicorn adaptive_nav.server.app:app --host 0.0.0.0 --port 8001

# Terminal 2: Run training (requires GPU + pip install trl transformers datasets accelerate)
python train_skeleton.py --env-url http://localhost:8001
```

This skeleton is **not** expected to converge -- it shows the wiring so you can iterate on prompting, multi-turn interaction, and reward shaping.

---

## Project Structure

```
adaptive_nav/
    __init__.py              # Package exports (local + OpenEnv)
    env.py                   # AdaptiveNavEnv -- local environment class
    generator.py             # Maze generation + dynamic changes
    renderer.py              # Text and RGB renderers
    models.py                # Pydantic Action/Observation/State (OpenEnv wire format)
    openenv_client.py        # EnvClient for remote access
    server/
        __init__.py
        nav_environment.py   # OpenEnv Environment wrapper around AdaptiveNavEnv
        app.py               # FastAPI app (openenv create_app)
        requirements.txt     # Server-specific deps
        Dockerfile           # For Docker / HF Spaces deployment
openenv.yaml                 # OpenEnv CLI config
app.py                       # Streamlit interactive demo (local)
demo.py                      # CLI demo script (local)
train_skeleton.py            # Minimal TRL training example
requirements.txt             # Core + local demo deps
README.md
```

### Architecture

```
+------------------+       +------------------------+
|  Local path      |       |  OpenEnv path          |
|  (demo.py,       |       |  (server/app.py,       |
|   app.py)        |       |   Docker, HF Spaces)   |
+--------+---------+       +----------+-------------+
         |                             |
         v                             v
   AdaptiveNavEnv              NavEnvironment
   (env.py)                    (wraps AdaptiveNavEnv)
         |                             |
         v                             v
   generator.py +              NavAction / NavObservation
   renderer.py                 (models.py -- Pydantic)
                                       |
                                       v
                               NavEnvClient
                               (openenv_client.py)
```

---

## Environment API (Local)

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
- **NumPy** -- grid representation and rendering
- **Pydantic** -- OpenEnv wire format
- **OpenEnv 0.2.1** -- environment server framework
- **FastAPI + Uvicorn** -- HTTP/WebSocket server
- **Streamlit** -- local interactive web demo
- **TRL** -- training skeleton (optional)

---

## License

MIT
