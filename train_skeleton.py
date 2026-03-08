#!/usr/bin/env python3
"""TRL training skeleton for the Adaptive Navigation Environment.

Demonstrates how to wire a grid-world OpenEnv environment into
Hugging Face TRL's GRPOTrainer.  Designed for easy Colab execution
on a single GPU -- no vLLM required.

This is a hackathon scaffold.  It won't converge to a strong policy,
but it shows the full loop: observation -> prompt -> LLM -> parse action
-> step environment -> accumulate reward -> train.

Colab quick-start:
    !pip install trl transformers datasets accelerate pydantic numpy
    !git clone https://github.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack.git
    %cd Meta_OpenEnv_PyTorch_Hack
    !python train_skeleton.py --episodes 16
"""

from __future__ import annotations

import argparse
import random
import re


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRL training skeleton for Adaptive Nav")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes", type=int, default=32,
                   help="Number of prompt episodes in the dataset")
    p.add_argument("--rollout-steps", type=int, default=8,
                   help="Steps per mini-episode during rollout")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=32,
                   help="Max tokens the model can generate per turn")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Observation → text prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an agent in a 2D grid maze.  Your mission:
  1. Find the key (K)
  2. Unlock the door (D)
  3. Reach the goal (G)

You can only see a small window around you.
Respond with EXACTLY ONE digit 0-5:
  0=up  1=down  2=left  3=right  4=interact  5=wait
No other text."""


def obs_to_prompt(obs) -> str:
    """Build a natural-language prompt from a NavObservation.

    The prompt gives the LLM everything the agent can see so it can
    reason about which action to take next.
    """
    # Format the local view as a readable grid
    if hasattr(obs, "local_view"):
        view = obs.local_view
    else:
        view = obs["local_view"]

    grid_lines = "\n".join(" ".join(row) for row in view)

    if hasattr(obs, "energy"):
        energy = obs.energy
        has_key = obs.has_key
        door_unlocked = obs.door_unlocked
        step_count = obs.step_count
        mission = obs.mission if hasattr(obs, "mission") else {}
    else:
        energy = obs["energy"]
        has_key = obs["has_key"]
        door_unlocked = obs["door_unlocked"]
        step_count = obs["step_count"]
        mission = obs.get("mission", {})

    m = mission
    mission_str = (
        f"  get_key: {'DONE' if m.get('get_key') else 'TODO'}\n"
        f"  unlock_door: {'DONE' if m.get('unlock_door') else 'TODO'}\n"
        f"  reach_goal: {'DONE' if m.get('reach_goal') else 'TODO'}"
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Current observation ---\n"
        f"Local view:\n{grid_lines}\n\n"
        f"Energy: {energy}\n"
        f"Has key: {has_key}\n"
        f"Door unlocked: {door_unlocked}\n"
        f"Step: {step_count}\n"
        f"Mission:\n{mission_str}\n\n"
        f"Your action (single digit 0-5):"
    )


# ── Parse model output ──────────────────────────────────────────────────

_ACTION_RE = re.compile(r"[0-5]")


def parse_action(text: str) -> int:
    """Extract the first valid action digit from model output.

    Returns 5 (wait) if nothing valid is found -- safe no-op.
    """
    m = _ACTION_RE.search(text.strip())
    return int(m.group()) if m else 5


# ── Mini-episode rollout ────────────────────────────────────────────────

def run_episode(env, tokenizer, model, max_steps: int, max_tokens: int) -> tuple[float, list[str]]:
    """Play a short episode: observe → generate → parse → step → repeat.

    Returns (total_reward, list_of_completions).
    Uses the local AdaptiveNavEnv directly -- no server needed.
    """
    import torch

    obs = env.reset(seed=random.randint(0, 99999))
    total_reward = 0.0
    completions: list[str] = []

    for _ in range(max_steps):
        prompt = obs_to_prompt(obs)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion)

        action_id = parse_action(completion)
        obs, reward, done, _info = env.step(action_id)
        total_reward += reward

        if done:
            break

    return total_reward, completions


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # -- lazy imports so missing deps give a clear error -------------------
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training requires:  pip install trl transformers datasets accelerate\n"
            f"Missing: {exc.name}"
        ) from exc

    from adaptive_nav.env import AdaptiveNavEnv

    # -- build prompt dataset from real environment observations -----------
    # Each row is a prompt built from a fresh episode's initial observation.
    env = AdaptiveNavEnv(grid_size=10, max_energy=50, dynamic_changes=True)
    prompts: list[str] = []
    for i in range(args.episodes):
        obs = env.reset(seed=args.seed + i)
        prompts.append(obs_to_prompt(obs))

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset: {len(dataset)} episodes, first prompt length: {len(prompts[0])} chars")

    # -- custom rollout: multi-step episode per prompt ---------------------
    def rollout_func(prompts_batch: list[str], trainer: GRPOTrainer):
        """For each prompt, run a short episode and return accumulated reward."""
        from transformers import AutoModelForCausalLM
        tokenizer = trainer.processing_class
        model = trainer.model

        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_rewards = []

        for prompt_text in prompts_batch:
            # Tokenize the prompt
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]

            # Run a mini-episode against the local environment
            ep_env = AdaptiveNavEnv(grid_size=10, max_energy=50, dynamic_changes=True)
            ep_reward, completions = run_episode(
                ep_env, tokenizer, model,
                max_steps=args.rollout_steps,
                max_tokens=args.max_completion_length,
            )

            # Use the concatenated completions as the "completion" for training
            full_completion = " ".join(completions)
            completion_ids = tokenizer(
                full_completion, return_tensors="pt", truncation=True,
                max_length=args.max_completion_length,
            )["input_ids"][0]

            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            # Placeholder logprobs (GRPOTrainer recomputes these internally)
            all_logprobs.append([0.0] * len(completion_ids))
            all_rewards.append(ep_reward)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_rewards,
        }

    # -- reward function ---------------------------------------------------
    def reward_from_env(completions, **kwargs):
        """Pass through environment rewards collected during rollout."""
        rewards = kwargs.get("env_reward", [])
        return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

    # -- trainer -----------------------------------------------------------
    # No vLLM -- plain PyTorch generation for maximum Colab compatibility.
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=GRPOConfig(
            use_vllm=False,
            num_train_epochs=args.epochs,
            num_generations=2,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            logging_steps=1,
            output_dir="./nav_training_output",
        ),
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
