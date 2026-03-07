"""Adaptive Navigation Environment -- a partially observable 2D grid-world.

Two ways to use this package:

1. **Local** (original path -- no extra deps):
       from adaptive_nav import AdaptiveNavEnv
       env = AdaptiveNavEnv(); env.reset()

2. **OpenEnv** (hackathon / deployment path -- needs openenv-core):
       from adaptive_nav.openenv_client import NavEnvClient
       from adaptive_nav.models import NavAction
       client = NavEnvClient(base_url="http://localhost:8000")
       client.reset(); client.step(NavAction(action_id=0))
"""

from adaptive_nav.env import AdaptiveNavEnv
from adaptive_nav.models import NavAction, NavObservation, NavState

__all__ = [
    "AdaptiveNavEnv",
    "NavAction",
    "NavObservation",
    "NavState",
]
