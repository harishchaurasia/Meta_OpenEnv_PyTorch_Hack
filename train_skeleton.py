#!/usr/bin/env python3
"""Minimal TRL + OpenEnv training skeleton for the Adaptive Navigation Environment.

This is a **scaffold only** -- it shows where each piece plugs in so you can
iterate quickly during the hackathon.  It is NOT expected to converge on a
good policy; the goal is to demonstrate the integration wiring.

Requirements (install in a Colab / GPU notebook):
    pip install trl transformers datasets accelerate
    pip install openenv-core[core]>=0.2.1
    pip install -e .            # installs adaptive_nav from this repo

Usage:
    # 1. Start the OpenEnv server (in another terminal or Docker):
    #    uvicorn adaptive_nav.server.app:app --host 0.0.0.0 --port 8001
    #
    # 2. Run training (colocate mode, 1 GPU):
    #    python train_skeleton.py

    # Or connect to a remote HF Space:
    #    python train_skeleton.py --env-url https://YOUR-SPACE.hf.space
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# This skeleton imports TRL / Datasets lazily so the file can be read (and
# linted) even when those heavy libraries are not installed locally.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRL training skeleton for Adaptive Nav")
    p.add_argument("--env-url", type=str, default="http://localhost:8001",
                    help="Base URL of the running OpenEnv server")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model id")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- lazy imports so missing deps give a clear error -------------------
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from trl.experimental.openenv import generate_rollout_completions
    except ImportError as exc:
        raise SystemExit(
            "Training requires:  pip install trl transformers datasets accelerate\n"
            f"Missing: {exc.name}"
        ) from exc

    from adaptive_nav.models import NavAction
    from adaptive_nav.openenv_client import NavEnvClient

    # -- connect to the OpenEnv server -------------------------------------
    client = NavEnvClient(base_url=args.env_url)

    # -- build a toy prompt dataset ----------------------------------------
    SYSTEM_PROMPT = (
        "You are an agent navigating a grid maze. "
        "Respond with a single action id (0-5): "
        "0=up 1=down 2=left 3=right 4=interact 5=wait."
    )
    dataset = Dataset.from_dict({
        "prompt": [SYSTEM_PROMPT] * 32,
    })

    # -- custom rollout function -------------------------------------------
    def rollout_func(prompts: list[str], trainer: GRPOTrainer):
        """Generate completions, step through the env, collect rewards."""
        outputs = generate_rollout_completions(trainer, prompts)
        tokenizer = trainer.processing_class
        completions_text = [
            tokenizer.decode(o["completion_ids"], skip_special_tokens=True)
            for o in outputs
        ]

        # Step through the environment for each completion
        env_rewards: list[float] = []
        for text in completions_text:
            client.reset()
            # Parse the model's output as an action id (best-effort)
            action_id = _parse_action(text)
            result = client.step(NavAction(action_id=action_id))
            env_rewards.append(float(result.reward or 0.0))

        return {
            "prompt_ids": [o["prompt_ids"] for o in outputs],
            "completion_ids": [o["completion_ids"] for o in outputs],
            "logprobs": [o["logprobs"] for o in outputs],
            "env_reward": env_rewards,
        }

    # -- reward function that reads env_reward from kwargs -----------------
    def reward_from_env(completions, **kwargs):
        rewards = kwargs.get("env_reward", [])
        return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

    # -- trainer setup -----------------------------------------------------
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=GRPOConfig(
            use_vllm=True,
            vllm_mode="colocate",
            num_train_epochs=args.epochs,
            num_generations=4,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
        ),
    )
    trainer.train()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> int:
    """Best-effort parse of a model completion into an action id 0-5."""
    text = text.strip()
    for ch in text:
        if ch.isdigit() and 0 <= int(ch) <= 5:
            return int(ch)
    return 5  # default to wait


if __name__ == "__main__":
    main()
