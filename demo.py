#!/usr/bin/env python3
"""CLI demo for the Adaptive Navigation Environment.

Usage
-----
  python demo.py                # random-action episode
  python demo.py --manual       # keyboard-controlled episode
  python demo.py --seed 42      # reproducible random episode
  python demo.py --easy         # small hand-crafted easy map
  python demo.py --debug        # shows full grid + path hints
"""

from __future__ import annotations

import argparse
import random
import sys

from adaptive_nav.env import ACTION_NAMES, AdaptiveNavEnv
from adaptive_nav.renderer import LEGEND

MANUAL_KEYS = {
    "w": 0, "s": 1, "a": 2, "d": 3,
    "e": 4,
    "q": 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive Navigation Environment demo")
    p.add_argument("--manual", action="store_true", help="Play manually with WASD+E+Q")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--grid-size", type=int, default=10, help="Grid side length")
    p.add_argument("--max-energy", type=int, default=80, help="Starting energy")
    p.add_argument("--no-dynamic", action="store_true", help="Disable dynamic changes")
    p.add_argument("--debug", action="store_true", help="Show full grid + path hints")
    p.add_argument("--easy", action="store_true", help="Use small easy test map")
    return p.parse_args()


def print_legend() -> None:
    print(f"\n  Legend: {LEGEND}")


def print_mission(obs: dict) -> None:
    m = obs["mission"]
    check = lambda v: "[x]" if v else "[ ]"
    print(f"  Mission: {check(m['get_key'])} Get key  "
          f"{check(m['unlock_door'])} Unlock door  "
          f"{check(m['reach_goal'])} Reach goal")


def print_obs(obs: dict, debug: bool = False) -> None:
    lv = obs["local_view"]
    print("\n--- Local Observation ---")
    for row in lv:
        print(" ".join(row))
    print(f"Energy: {obs['energy']}  |  Has key: {obs['has_key']}  |  "
          f"Door unlocked: {obs['door_unlocked']}  |  Step: {obs['step_count']}")
    print_mission(obs)

    if debug and "path_hints" in obs:
        hints = obs["path_hints"]
        for label, path in hints.items():
            print(f"  Path hint ({label}): {len(path)} steps")


def main() -> None:
    args = parse_args()
    env = AdaptiveNavEnv(
        grid_size=args.grid_size,
        max_energy=args.max_energy,
        dynamic_changes=not args.no_dynamic,
        seed=args.seed,
        debug=args.debug,
        easy=args.easy,
    )
    obs = env.reset()

    print("=" * 50)
    print("  ADAPTIVE NAVIGATION ENVIRONMENT")
    print("=" * 50)
    print_legend()

    if args.debug:
        hints = obs.get("path_hints")
        print(env.render(mode="text") if not hints
              else _render_with_hints(env, hints))
    else:
        print(env.render(mode="text"))
    print_obs(obs, debug=args.debug)

    if args.manual:
        print("\nControls: W=up  S=down  A=left  D=right  E=interact  Q=wait")

    total_reward = 0.0
    while not env.is_done():
        if args.manual:
            raw = input("\nAction> ").strip().lower()
            if raw in ("exit", "quit"):
                print("Quitting.")
                sys.exit(0)
            if raw not in MANUAL_KEYS:
                print(f"Unknown key '{raw}'. Use: {list(MANUAL_KEYS.keys())}")
                continue
            action = MANUAL_KEYS[raw]
        else:
            action = random.randint(0, 5)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\n>> Action: {ACTION_NAMES[action]}  |  Reward: {reward:+.1f}  |  "
              f"Info: {info.get('msg', '')}")
        if "dynamic_event" in info:
            print(f"   *** {info['dynamic_event']} ***")

        if args.debug:
            hints = obs.get("path_hints")
            print(_render_with_hints(env, hints) if hints
                  else env.render(mode="text"))
        else:
            print(env.render(mode="text"))
        print_obs(obs, debug=args.debug)

    done_reason = info.get("done_reason", "unknown") if 'info' in dir() else env.done_reason

    print("\n" + "=" * 50)
    print("  EPISODE FINISHED")
    print("=" * 50)
    print(f"  Result : {'WIN' if env.won else 'LOSS'}")
    print(f"  Reason : {env.done_reason}")
    print(f"  Steps  : {env.step_count}")
    print(f"  Energy : {env.energy}")
    print(f"  Reward : {total_reward:+.1f}")
    print("=" * 50)


def _render_with_hints(env: AdaptiveNavEnv, hints: dict | None) -> str:
    """Render text grid with path-hint overlay."""
    from adaptive_nav.renderer import render_text
    return render_text(env.grid, env.agent_pos, path_hints=hints)


if __name__ == "__main__":
    main()
