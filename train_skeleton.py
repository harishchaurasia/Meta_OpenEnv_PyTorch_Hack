#!/usr/bin/env python3
"""Minimal TRL training scaffold for the Adaptive Navigation Environment.

What this does:
- builds text prompts from real environment observations
- lets a small LLM choose one of 6 discrete actions
- runs short multi-step rollouts
- accumulates environment reward
- feeds that reward back into TRL GRPOTrainer

This is intentionally hackathon-friendly:
- simple
- easy to explain
- works in local mode by default
- can optionally use the deployed OpenEnv server / HF Space

Colab quick start:
    !pip install trl transformers datasets accelerate pydantic numpy
    !git clone https://github.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack.git
    %cd Meta_OpenEnv_PyTorch_Hack
    !python train_skeleton.py --episodes 8 --epochs 1 --mode local

Remote / OpenEnv mode:
    !python train_skeleton.py --mode remote --env-url https://YOUR-SPACE.hf.space --episodes 8 --epochs 1
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRL training scaffold for Adaptive Nav")
    p.add_argument(
        "--mode",
        choices=["local", "remote"],
        default="local",
        help="Use local AdaptiveNavEnv directly or remote OpenEnv server via NavEnvClient",
    )
    p.add_argument(
        "--env-url",
        type=str,
        default="https://harishchaurasia-adaptive-nav-openenv.hf.space",
        help="Base URL of the deployed OpenEnv environment (used in remote mode)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id",
    )
    p.add_argument("--episodes", type=int, default=8, help="Number of initial prompts")
    p.add_argument("--rollout-steps", type=int, default=8, help="Steps per mini-episode")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=8,
        help="Max new tokens for each action generation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Only used in local mode",
    )
    p.add_argument(
        "--max-energy",
        type=int,
        default=50,
        help="Only used in local mode",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Prompting
# ──────────────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are controlling an autonomous exploration robot in a partially observable 2D environment.

# Mission:
# 1. Find the access token (K)
# 2. Unlock the secured checkpoint (D)
# 3. Reach the target zone (G)

# Action space:
# 0 = up
# 1 = down
# 2 = left
# 3 = right
# 4 = interact
# 5 = wait

# Rules:
# - Respond with EXACTLY ONE digit from 0 to 5
# - Do not explain
# - Do not output any other text
# """

SYSTEM_PROMPT = """You are controlling an autonomous exploration robot.

Mission:
1. Find the access token (K)
2. Unlock the secured checkpoint (D)
3. Reach the target zone (G)

Valid actions:
up
down
left
right
interact
wait

Respond with EXACTLY ONE valid action word from the list above.
Do not explain.
Do not output anything else.
"""


def _get_value(obs: Any, key: str, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def obs_to_prompt(obs: Any) -> str:
    """Convert a local or remote observation into an LLM prompt."""
    local_view = _get_value(obs, "local_view", [])
    energy = _get_value(obs, "energy", 0)
    has_key = _get_value(obs, "has_key", False)
    door_unlocked = _get_value(obs, "door_unlocked", False)
    step_count = _get_value(obs, "step_count", 0)
    mission = _get_value(obs, "mission", {}) or {}

    grid_lines = "\n".join(" ".join(map(str, row)) for row in local_view)

    mission_str = (
        f"get_key: {'DONE' if mission.get('get_key') else 'TODO'}\n"
        f"unlock_door: {'DONE' if mission.get('unlock_door') else 'TODO'}\n"
        f"reach_goal: {'DONE' if mission.get('reach_goal') else 'TODO'}"
    )

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Current sensor window:\n{grid_lines}\n\n"
        f"Battery: {energy}\n"
        f"Has access token: {has_key}\n"
        f"Checkpoint unlocked: {door_unlocked}\n"
        f"Step: {step_count}\n"
        f"Mission status:\n{mission_str}\n\n"
        f"Output one action digit (0-5):"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Action parsing
# ──────────────────────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"^\s*([0-5])\s*$")


# def parse_action(text: str) -> int:
#     """Strict parse: accept only a clean single digit 0-5, else wait."""
#     text = text.strip()
#     m = _ACTION_RE.match(text)
#     return int(m.group(1)) if m else 5


def parse_action(text: str) -> int:
    text = text.strip().lower()

    mapping = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
        "interact": 4,
        "wait": 5,
    }

    for word, action_id in mapping.items():
        if word in text:
            return action_id

    return 5


def action_to_name(action_id: int) -> str:
    mapping = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
        4: "interact",
        5: "wait",
    }
    return mapping.get(action_id, "wait")


# ──────────────────────────────────────────────────────────────────────────────
# Environment adapters
# ──────────────────────────────────────────────────────────────────────────────

class LocalEnvAdapter:
    """Thin adapter around AdaptiveNavEnv so local and remote modes look similar."""

    def __init__(self, grid_size: int, max_energy: int, dynamic_changes: bool = True):
        from adaptive_nav.env import AdaptiveNavEnv
        self.env = AdaptiveNavEnv(
            grid_size=grid_size,
            max_energy=max_energy,
            dynamic_changes=dynamic_changes,
        )

    def reset(self, seed: int | None = None):
        return self.env.reset(seed=seed)

    def step(self, action_id: int):
        return self.env.step(action_id)


class RemoteEnvAdapter:
    """Thin adapter around NavEnvClient for deployed OpenEnv environments."""

    def __init__(self, base_url: str):
        from adaptive_nav.models import NavAction
        from adaptive_nav.openenv_client import NavEnvClient

        self.NavAction = NavAction
        self.client = NavEnvClient(base_url=base_url)

    def reset(self, seed: int | None = None):
        # Current client/server path may ignore seed; that's okay for scaffold use.
        result = self.client.reset()
        return result.observation if hasattr(result, "observation") else result

    def step(self, action_id: int):
        result = self.client.step(self.NavAction(action_id=action_id))
        obs = result.observation if hasattr(result, "observation") else result
        reward = float(getattr(result, "reward", 0.0) or 0.0)
        done = bool(getattr(result, "done", False))
        info = {}
        return obs, reward, done, info


def make_env_factory(args: argparse.Namespace):
    if args.mode == "local":
        def factory():
            return LocalEnvAdapter(
                grid_size=args.grid_size,
                max_energy=args.max_energy,
                dynamic_changes=True,
            )
        return factory

    def factory():
        return RemoteEnvAdapter(base_url=args.env_url)

    return factory


# ──────────────────────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────────────────────

def generate_action_completion(prompt: str, tokenizer, model, max_tokens: int) -> str:
    """Generate one short completion that should contain exactly one action digit."""
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        # output_ids = model.generate(
        #     **inputs,
        #     max_new_tokens=3,
        #     do_sample=False,
        #     temperature=0.7,
        #     top_p=0.9,
        #     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        # )

        output_ids = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_episode(env, tokenizer, model, max_steps: int, max_tokens: int, seed: int | None = None):
    """Run a short rollout and return total reward plus generated action texts."""
    obs = env.reset(seed=seed)
    total_reward = 0.0
    completions: list[str] = []
    action_trace: list[int] = []

    for _ in range(max_steps):
        prompt = obs_to_prompt(obs)
        completion = generate_action_completion(prompt, tokenizer, model, max_tokens)
        action_id = parse_action(completion)

        completions.append(completion)
        action_trace.append(action_id)

        obs, reward, done, _info = env.step(action_id)
        total_reward += float(reward)

        if done:
            break

    return total_reward, completions, action_trace


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install missing packages first:\n"
            "pip install trl transformers datasets accelerate pydantic numpy"
        ) from exc

    env_factory = make_env_factory(args)

    # Build a small prompt dataset from initial observations.
    prompts: list[str] = []
    for i in range(args.episodes):
        env = env_factory()
        obs = env.reset(seed=args.seed + i)
        prompts.append(obs_to_prompt(obs))

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Built dataset with {len(dataset)} prompts.")
    print(f"Mode: {args.mode}")
    if args.mode == "remote":
        print(f"Remote env: {args.env_url}")

    def rollout_func(prompts_batch: list[str], trainer: GRPOTrainer):
        """For each prompt, run a short episode and return accumulated reward."""
        tokenizer = trainer.processing_class
        model = trainer.model

        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_rewards = []

        for idx, prompt_text in enumerate(prompts_batch):
            env = env_factory()

            ep_reward, completions, _actions = run_episode(
                env=env,
                tokenizer=tokenizer,
                model=model,
                max_steps=args.rollout_steps,
                max_tokens=args.max_completion_length,
                seed=args.seed + idx,
            )

            # We pack the action outputs into a short completion string.
            # Example: "3 3 1 4"
            packed_completion = " ".join(completions).strip()
            if not packed_completion:
                packed_completion = "5"

            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )["input_ids"][0]

            completion_ids = tokenizer(
                packed_completion,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            )["input_ids"][0]

            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            all_logprobs.append([0.0] * len(completion_ids))
            all_rewards.append(float(ep_reward))

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_rewards,
        }

    def reward_from_env(completions, **kwargs):
        rewards = kwargs.get("env_reward", [])
        return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=GRPOConfig(
            use_vllm=False,
            num_train_epochs=args.epochs,
            num_generations=2,
            max_completion_length=64,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            logging_steps=1,
            output_dir="./nav_training_output",
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # Tiny post-training sanity rollout
    print("\nRunning one post-training sanity rollout...")
    tokenizer = trainer.processing_class
    model = trainer.model
    env = env_factory()
    total_reward, completions, actions = run_episode(
        env=env,
        tokenizer=tokenizer,
        model=model,
        max_steps=args.rollout_steps,
        max_tokens=args.max_completion_length,
        seed=args.seed + 999,
    )

    print("Action trace:", [action_to_name(a) for a in actions])
    print("Raw completions:", completions)
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()