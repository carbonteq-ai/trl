#!/usr/bin/env python
"""Exercise TRL's native async agent worker at bounded rollout concurrency."""

from __future__ import annotations

import argparse
import json
import queue
import time
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker


class ChoiceEnvironment:
    """Small tool environment used to verify the native agent loop."""

    def __init__(self) -> None:
        self.choice: str | None = None

    def reset(self, **kwargs: Any) -> None:
        del kwargs

    def record_choice(self, choice: str) -> str:
        """Record one answer choice.

        Args:
            choice: The answer choice to record.

        Returns:
            A confirmation of the recorded choice.
        """
        self.choice = choice
        return f"Recorded {choice}."

    def get_reward(self) -> float:
        return 1.0 if self.choice == "A" else 0.0


def ordinal_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Ensure every group has a finite, nonconstant training signal."""
    del kwargs
    return [float(index) for index, _completion in enumerate(completions)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--vllm-server-url", default="http://127.0.0.1:8129")
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-inflight-tasks", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(args.num_groups, args.num_generations, args.max_inflight_tasks, args.max_tokens) < 1:
        raise ValueError("all async rollout workload bounds must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = [
        {
            "role": "user",
            "content": 'Use record_choice with choice="A", then briefly confirm the result.',
        }
    ]
    dataset = Dataset.from_list([{"prompt": prompt} for _ in range(args.num_groups)])
    expected_samples = args.num_groups * args.num_generations
    worker = AsyncRolloutWorker(
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[ordinal_reward],
        processing_class=tokenizer,
        environment_factory=ChoiceEnvironment,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight_tasks,
        queue_maxsize=expected_samples,
        score_queue_maxsize=max(args.num_groups, 1),
        vllm_server_url=args.vllm_server_url,
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        request_timeout=int(args.timeout),
        max_tool_calling_iterations=2,
        child_ready_timeout=int(args.timeout),
    )
    started = time.monotonic()
    samples = []
    metric_payloads = []
    worker.start()
    try:
        deadline = time.monotonic() + args.timeout
        while len(samples) < expected_samples:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                worker.check_health(args.timeout)
                raise TimeoutError(f"received {len(samples)} of {expected_samples} rollout samples")
            try:
                samples.append(worker.rollout_buffer.get(timeout=min(remaining, 1.0)))
            except queue.Empty:
                worker.check_health(args.timeout)
        while True:
            try:
                metric_payloads.append(worker.metrics_queue.get_nowait())
            except queue.Empty:
                break
    finally:
        worker.stop()

    by_group: dict[int, int] = {}
    for sample in samples:
        by_group[sample.group_id] = by_group.get(sample.group_id, 0) + 1
    if len(by_group) < args.num_groups or any(count != args.num_generations for count in by_group.values()):
        raise RuntimeError(f"native async worker returned incomplete groups: {by_group}")
    tool_calls = sum(float(sample.metrics.get("tools/call_frequency", 0.0)) for sample in samples)
    tool_failures = sum(
        float(sample.metrics.get("tools/failure_frequency", 0.0)) for sample in samples
    )
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "groups": len(by_group),
                "max_inflight_tasks": args.max_inflight_tasks,
                "model": args.model,
                "samples": len(samples),
                "samples_per_second": len(samples) / elapsed,
                "tool_call_frequency_sum": tool_calls,
                "tool_failure_frequency_sum": tool_failures,
                "worker_metric_payloads": len(metric_payloads),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
