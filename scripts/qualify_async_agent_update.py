#!/usr/bin/env python
"""Run one optimizer update through TRL's original async agent worker."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


class ChoiceEnvironment:
    """Minimal per-rollout tool environment for the native agent interface."""

    def __init__(self) -> None:
        self.choice: str | None = None

    def reset(self, **kwargs: Any) -> None:
        del kwargs

    def record_choice(self, choice: str) -> str:
        """Record an answer choice.

        Args:
            choice: The answer choice to record.

        Returns:
            A confirmation of the recorded choice.
        """
        self.choice = choice
        return f"Recorded {choice}."

    def get_reward(self) -> float:
        return 1.0 if self.choice == "A" else 0.0


class NoopWeightTransfer:
    """Allow a one-GPU learner/serving smoke test without NCCL self-join."""

    def init_weight_transfer(self) -> None:
        return None

    def pause(self) -> None:
        return None

    def send_weights(self, iterator: Any) -> None:
        for _ in iterator:
            pass

    def resume(self) -> None:
        return None

    def destroy(self) -> None:
        return None


def ordinal_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Keep a nonconstant group signal even if the small model skips the tool."""
    del kwargs
    return [float(index) for index, _completion in enumerate(completions)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--vllm-server-base-url", default="http://127.0.0.1:8129")
    parser.add_argument("--output-dir", default="outputs/native-async-agent-update")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": 'Use record_choice with choice="A", then briefly confirm the result.',
                    }
                ]
            }
        ]
    )
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        num_generations=2,
        max_steps=1,
        max_completion_length=32,
        token_budget=-1,
        max_inflight_tasks=2,
        queue_maxsize=4,
        max_tool_calling_iterations=2,
        request_timeout=60,
        vllm_server_base_url=args.vllm_server_base_url,
        weight_sync_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        reward_funcs=ordinal_reward,
        args=config,
        train_dataset=dataset,
        environment_factory=ChoiceEnvironment,
        weight_transfer=NoopWeightTransfer(),
    )
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainer.model.named_parameters()}
    result = trainer.train()
    changed = sum(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in trainer.model.named_parameters()
    )
    if result.global_step != 1 or changed == 0:
        raise RuntimeError("native async agent gate did not perform one parameter-changing optimizer update")
    final_metrics = trainer.state.log_history[-2] if len(trainer.state.log_history) > 1 else {}
    print(
        json.dumps(
            {
                "changed_parameter_tensors": changed,
                "global_step": result.global_step,
                "model_version": trainer.model_version,
                "reward": final_metrics.get("reward"),
                "tool_call_frequency": final_metrics.get("tools/call_frequency"),
                "tool_failure_frequency": final_metrics.get("tools/failure_frequency"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
