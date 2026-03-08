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
# Action masking from the local observation window
# ──────────────────────────────────────────────────────────────────────────────

# Grid symbols (must match adaptive_nav.generator)
_WALL = "#"
_KEY = "K"
_DOOR = "D"

# Movement deltas indexed by action id: up=0, down=1, left=2, right=3
_MOVE_DR_DC = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def compute_action_mask(obs: Any) -> tuple[list[bool], list[str]]:
    """Inspect the local view and decide which actions are clearly useless.

    Returns ``(mask, reasons)`` where ``mask[i]`` is True when action *i* is
    allowed.  ``reasons`` collects human-readable strings for masked actions.
    """
    local_view = _get(obs, "local_view", [])
    has_key = _get(obs, "has_key", False)

    # Convert to plain list-of-lists if numpy
    if hasattr(local_view, "tolist"):
        local_view = local_view.tolist()

    rows = len(local_view)
    cols = len(local_view[0]) if rows else 0
    cr, cc = rows // 2, cols // 2  # agent is always at the center

    mask = [True] * 6
    reasons: list[str] = []

    # -- Mask movement into walls ------------------------------------------
    for action_id, (dr, dc) in enumerate(_MOVE_DR_DC):
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cell = str(local_view[nr][nc])
            if cell == _WALL:
                mask[action_id] = False
                reasons.append(f"{ACTION_WORDS[action_id]}→wall")
        else:
            # Out of the view window means out-of-bounds wall
            mask[action_id] = False
            reasons.append(f"{ACTION_WORDS[action_id]}→edge")

    # -- Mask interact unless it would actually do something -----------------
    agent_cell = str(local_view[cr][cc]) if rows and cols else ""
    standing_on_key = agent_cell == _KEY
    door_unlocked = bool(_get(obs, "door_unlocked", False))

    door_adjacent = False
    for dr, dc in _MOVE_DR_DC:
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if str(local_view[nr][nc]) == _DOOR:
                door_adjacent = True
                break

    can_interact = standing_on_key or (door_adjacent and has_key and not door_unlocked)

    if not can_interact:
        mask[4] = False
        if door_adjacent and not has_key:
            reasons.append("interact→door but no key")
        elif door_adjacent and door_unlocked:
            reasons.append("interact→door already open")
        else:
            reasons.append("interact→nothing nearby")

    # -- wait is always valid (index 5) ------------------------------------
    # (already True)

    return mask, reasons


# ──────────────────────────────────────────────────────────────────────────────
# Multiple-choice action scoring (with masking)
# ──────────────────────────────────────────────────────────────────────────────

_NEG_INF = float("-inf")


def score_actions(prompt: str, tokenizer, model, mask: list[bool]) -> list[float]:
    """Score each candidate action by its log-probability under the model.

    Masked actions (``mask[i] == False``) get ``-inf`` and are never chosen.
    """
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024)["input_ids"]
    prompt_len = prompt_ids.shape[1]
    device = next(model.parameters()).device

    scores: list[float] = []
    for idx, word in enumerate(ACTION_WORDS):
        if not mask[idx]:
            scores.append(_NEG_INF)
            continue

        full_text = f"{prompt} {word}"
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=1024 + 8)["input_ids"].to(device)
        action_len = full_ids.shape[1] - prompt_len
        if action_len <= 0:
            scores.append(_NEG_INF)
            continue

        with torch.no_grad():
            logits = model(full_ids).logits  # (1, seq_len, vocab)

        # logits[:, t, :] predicts token t+1, so positions
        # [prompt_len-1 .. -2] score tokens [prompt_len .. -1].
        log_probs = F.log_softmax(logits[0, prompt_len - 1: -1, :], dim=-1)
        target_ids = full_ids[0, prompt_len:]
        token_scores = log_probs[range(action_len), target_ids]
        scores.append(token_scores.sum().item())

    return scores


def choose_action(
    prompt: str,
    obs: Any,
    tokenizer,
    model,
) -> tuple[int, str, list[float], list[bool], list[str]]:
    """Pick the best unmasked action via log-prob scoring.

    Returns (action_id, action_word, scores, mask, mask_reasons).
    """
    mask, reasons = compute_action_mask(obs)
    scores = score_actions(prompt, tokenizer, model, mask)

    # Pick the highest-scoring *allowed* action (fallback: wait)
    best = 5
    best_score = _NEG_INF
    for i, s in enumerate(scores):
        if mask[i] and s > best_score:
            best, best_score = i, s

    return best, ACTION_WORDS[best], scores, mask, reasons


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
    """Plain HTTP adapter for the deployed OpenEnv server.

    Uses the REST endpoints (/api/reset, /api/step) instead of WebSocket.
    HF Spaces aggressively closes idle WebSocket sessions, so HTTP is far
    more reliable for hackathon use.
    """

    def __init__(self, base_url: str):
        import json
        import urllib.request
        self._json = json
        self._urlopen = urllib.request.urlopen
        self._Request = urllib.request.Request
        self._base = base_url.rstrip("/")

    def _post(self, path: str, body: dict) -> dict:
        data = self._json.dumps(body).encode()
        req = self._Request(
            f"{self._base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = self._urlopen(req, timeout=30)
        return self._json.loads(resp.read())

    def reset(self, seed: int | None = None):
        return self._post("/api/reset", {"seed": seed})

    def step(self, action_id: int):
        result = self._post("/api/step", {"action_id": action_id})
        obs = result
        reward = float(result.get("reward", 0.0) or 0.0)
        done = bool(result.get("done", False))
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
    """Play up to *max_steps* using masked log-prob action scoring each turn.

    Returns (total_reward, chosen_action_words, action_ids).
    """
    obs = env.reset(seed=seed)
    total_reward = 0.0
    words: list[str] = []
    ids: list[int] = []

    for t in range(max_steps):
        prompt = obs_to_prompt(obs)
        action_id, word, scores, mask, mask_reasons = choose_action(
            prompt, obs, tokenizer, model,
        )

        if verbose:
            # Show allowed actions with scores, then which ones were masked
            allowed = [
                (ACTION_WORDS[i], scores[i])
                for i in range(6) if mask[i]
            ]
            allowed.sort(key=lambda x: x[1], reverse=True)
            top = " | ".join(f"{w}={s:+.2f}" for w, s in allowed[:4])
            masked_str = ", ".join(mask_reasons) if mask_reasons else "none"
            print(f"  step {t}: chose {word:>8s}  ({top})  masked=[{masked_str}]")

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
