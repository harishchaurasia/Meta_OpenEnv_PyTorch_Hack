#!/usr/bin/env python3
"""TRL training scaffold for the Adaptive Navigation Environment.

**Multiple-choice action selection**: instead of generating free-form text and
parsing it, we score each of the 6 candidate action words by computing the
model's log-probability of each word given the observation prompt, then pick
the highest-scoring action.  This is deterministic, never produces invalid
actions, and gives a clean gradient signal for GRPO.

Colab quick start:
    !pip install trl transformers datasets accelerate pydantic numpy
    !git clone https://github.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack.git
    %cd Meta_OpenEnv_PyTorch_Hack
    !python train_skeleton.py --episodes 8 --epochs 1 --mode local

Remote / OpenEnv mode:
    !python train_skeleton.py --mode remote --env-url https://YOUR-SPACE.hf.space
"""

from __future__ import annotations

import argparse
import random
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Action vocabulary (the 6 legal moves)
# ──────────────────────────────────────────────────────────────────────────────

ACTION_WORDS = ["up", "down", "left", "right", "interact", "wait"]
ACTION_TO_ID = {w: i for i, w in enumerate(ACTION_WORDS)}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRL training scaffold for Adaptive Nav")
    p.add_argument("--mode", choices=["local", "remote"], default="local",
                   help="local AdaptiveNavEnv or remote OpenEnv server")
    p.add_argument("--env-url", type=str,
                   default="https://harishchaurasia-adaptive-nav-openenv.hf.space")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--max-energy", type=int, default=50)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Observation → prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are controlling an exploration robot in a partially observable 2D grid.

Mission:
1. Find the access token (K)
2. Unlock the secured checkpoint (D)
3. Reach the target zone (G)

You must choose exactly one action from: up, down, left, right, interact, wait"""


def _get(obs: Any, key: str, default=None):
    """Read a field from a dict or a Pydantic model uniformly."""
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def obs_to_prompt(obs: Any) -> str:
    """Build a text prompt from the current environment observation."""
    local_view = _get(obs, "local_view", [])
    energy = _get(obs, "energy", 0)
    has_key = _get(obs, "has_key", False)
    door_unlocked = _get(obs, "door_unlocked", False)
    step_count = _get(obs, "step_count", 0)
    mission = _get(obs, "mission", {}) or {}

    grid_lines = "\n".join(" ".join(map(str, row)) for row in local_view)
    mission_str = (
        f"  get_key: {'DONE' if mission.get('get_key') else 'TODO'}\n"
        f"  unlock_door: {'DONE' if mission.get('unlock_door') else 'TODO'}\n"
        f"  reach_goal: {'DONE' if mission.get('reach_goal') else 'TODO'}"
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Sensor window:\n{grid_lines}\n\n"
        f"Battery: {energy}\n"
        f"Has access token: {has_key}\n"
        f"Checkpoint unlocked: {door_unlocked}\n"
        f"Step: {step_count}\n"
        f"Mission:\n{mission_str}\n\n"
        f"Best action:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multiple-choice action scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_actions(prompt: str, tokenizer, model) -> list[float]:
    """Score each candidate action by its log-probability under the model.

    For each action word we concatenate ``prompt + " " + word``, run a single
    forward pass, and sum the log-probs of the action-word tokens.  This is
    the standard "completion scoring" trick for multiple-choice with causal LMs.

    Returns a list of 6 floats (one score per action).
    """
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024)["input_ids"]
    prompt_len = prompt_ids.shape[1]
    device = next(model.parameters()).device

    scores: list[float] = []
    for word in ACTION_WORDS:
        full_text = f"{prompt} {word}"
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=1024 + 8)["input_ids"].to(device)
        # Number of tokens belonging to " <word>"
        action_len = full_ids.shape[1] - prompt_len
        if action_len <= 0:
            scores.append(float("-inf"))
            continue

        with torch.no_grad():
            logits = model(full_ids).logits  # (1, seq_len, vocab)

        # Log-probs of each token in the action suffix
        # logits[:, t, :] predicts token t+1, so we look at positions
        # [prompt_len-1 .. -2] to score tokens [prompt_len .. -1].
        log_probs = F.log_softmax(logits[0, prompt_len - 1: -1, :], dim=-1)
        target_ids = full_ids[0, prompt_len:]
        token_scores = log_probs[range(action_len), target_ids]
        scores.append(token_scores.sum().item())

    return scores


def choose_action(prompt: str, tokenizer, model) -> tuple[int, str, list[float]]:
    """Pick the best action via log-prob scoring.

    Returns (action_id, action_word, all_scores).
    """
    scores = score_actions(prompt, tokenizer, model)
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return best, ACTION_WORDS[best], scores


# ──────────────────────────────────────────────────────────────────────────────
# Environment adapters (same as before — local vs. remote)
# ──────────────────────────────────────────────────────────────────────────────

class LocalEnvAdapter:
    def __init__(self, grid_size: int, max_energy: int, dynamic_changes: bool = True):
        from adaptive_nav.env import AdaptiveNavEnv
        self.env = AdaptiveNavEnv(
            grid_size=grid_size, max_energy=max_energy,
            dynamic_changes=dynamic_changes,
        )

    def reset(self, seed: int | None = None):
        return self.env.reset(seed=seed)

    def step(self, action_id: int):
        return self.env.step(action_id)


class RemoteEnvAdapter:
    def __init__(self, base_url: str):
        from adaptive_nav.models import NavAction
        from adaptive_nav.openenv_client import NavEnvClient
        self.NavAction = NavAction
        self.client = NavEnvClient(base_url=base_url)

    def reset(self, seed: int | None = None):
        result = self.client.reset()
        return result.observation if hasattr(result, "observation") else result

    def step(self, action_id: int):
        result = self.client.step(self.NavAction(action_id=action_id))
        obs = result.observation if hasattr(result, "observation") else result
        reward = float(getattr(result, "reward", 0.0) or 0.0)
        done = bool(getattr(result, "done", False))
        return obs, reward, done, {}


def make_env_factory(args: argparse.Namespace):
    if args.mode == "local":
        return lambda: LocalEnvAdapter(args.grid_size, args.max_energy)
    return lambda: RemoteEnvAdapter(args.env_url)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step episode rollout (with multiple-choice scoring)
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    tokenizer,
    model,
    max_steps: int,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[float, list[str], list[int]]:
    """Play up to *max_steps* using log-prob action scoring each turn.

    Returns (total_reward, chosen_action_words, action_ids).
    """
    obs = env.reset(seed=seed)
    total_reward = 0.0
    words: list[str] = []
    ids: list[int] = []

    for t in range(max_steps):
        prompt = obs_to_prompt(obs)
        action_id, word, scores = choose_action(prompt, tokenizer, model)

        if verbose:
            ranked = sorted(zip(ACTION_WORDS, scores), key=lambda x: x[1], reverse=True)
            top3 = " | ".join(f"{w}={s:+.2f}" for w, s in ranked[:3])
            print(f"  step {t}: chose {word:>8s}  ({top3})")

        words.append(word)
        ids.append(action_id)

        obs, reward, done, _ = env.step(action_id)
        total_reward += float(reward)
        if done:
            break

    return total_reward, words, ids


# ──────────────────────────────────────────────────────────────────────────────
# Main (TRL GRPO training loop)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install missing packages:\n"
            "  pip install trl transformers datasets accelerate pydantic numpy"
        ) from exc

    env_factory = make_env_factory(args)

    # ── Build prompt dataset from real initial observations ───────────────
    prompts: list[str] = []
    for i in range(args.episodes):
        env = env_factory()
        obs = env.reset(seed=args.seed + i)
        prompts.append(obs_to_prompt(obs))

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset: {len(dataset)} prompts  |  mode={args.mode}")

    # ── Rollout function: run a scored episode per prompt ─────────────────
    def rollout_func(prompts_batch: list[str], trainer: GRPOTrainer):
        tokenizer = trainer.processing_class
        model = trainer.model

        all_prompt_ids, all_completion_ids = [], []
        all_logprobs, all_rewards = [], []

        for idx, prompt_text in enumerate(prompts_batch):
            env = env_factory()
            ep_reward, action_words, _ = run_episode(
                env, tokenizer, model,
                max_steps=args.rollout_steps,
                seed=args.seed + idx,
            )

            # The "completion" for GRPO is the sequence of chosen action words
            completion_text = " ".join(action_words) or "wait"
            prompt_ids = tokenizer(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=1024,
            )["input_ids"][0]
            completion_ids = tokenizer(
                completion_text, return_tensors="pt",
                truncation=True, max_length=64,
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

    # ── Reward function just passes through environment reward ────────────
    def reward_from_env(completions, **kwargs):
        rewards = kwargs.get("env_reward", [])
        return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

    # ── Trainer (no vLLM — pure PyTorch for Colab compatibility) ──────────
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

    print("Starting GRPO training...")
    trainer.train()
    print("Training complete.\n")

    # ── Post-training sanity rollout (verbose) ────────────────────────────
    print("Post-training rollout:")
    tokenizer = trainer.processing_class
    model = trainer.model
    env = env_factory()
    total_reward, words, ids = run_episode(
        env, tokenizer, model,
        max_steps=args.rollout_steps,
        seed=args.seed + 999,
        verbose=True,
    )
    print(f"\nChosen actions: {words}")
    print(f"Action ids:     {ids}")
    print(f"Total reward:   {total_reward:.2f}")


if __name__ == "__main__":
    main()
