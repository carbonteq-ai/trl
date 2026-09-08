#!/usr/bin/env python
"""Run the bounded native-AsyncLLM lifecycle gate used by online rollouts.

This is a qualification probe, not a benchmark or training entry point.  It
checks per-request submission, explicit cancellation, final-token logprobs,
drain, sleep/wake, and a second fixed-policy collection on one real engine.
Actor-to-engine weight synchronization remains a separate parity gate owned by
the trainer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from trl.generation.async_vllm_session import AsyncGenerationRequest, AsyncVllmSession


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write one concise sentence about reliable software.")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _request(request_id: str, prompt_ids: list[int], *, max_tokens: int) -> AsyncGenerationRequest:
    return AsyncGenerationRequest(
        request_id=request_id,
        prompt_token_ids=tuple(prompt_ids),
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=0),
    )


def _completion_evidence(output: Any) -> dict[str, Any]:
    if not getattr(output, "finished", False):
        raise RuntimeError("native AsyncLLM stream ended without a finished output")
    choices = getattr(output, "outputs", None)
    if not choices:
        raise RuntimeError("native AsyncLLM returned no completion choices")
    completion = choices[0]
    token_ids = list(completion.token_ids)
    logprobs = completion.logprobs
    if not token_ids or logprobs is None or len(logprobs) != len(token_ids):
        raise RuntimeError("native AsyncLLM did not return one logprob entry per sampled token")
    return {"completion_tokens": len(token_ids), "finish_reason": completion.finish_reason}


async def _wait_until_admitted(session: AsyncVllmSession, request_id: str) -> None:
    async with asyncio.timeout(30):
        while request_id not in session.active_request_ids:
            await asyncio.sleep(0)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    synchronized_versions: list[str] = []

    async def record_synchronization(version: str) -> None:
        synchronized_versions.append(version)

    session = AsyncVllmSession(engine, record_synchronization, sleep_level=1)
    try:
        await session.synchronize_policy("base-0")
        await session.open_policy("base-0")
        completing = asyncio.create_task(session.generate(_request("complete-0", prompt_ids, max_tokens=4)))
        cancelled = asyncio.create_task(session.generate(_request("cancel-0", prompt_ids, max_tokens=256)))
        await _wait_until_admitted(session, "cancel-0")
        if not await session.abort("cancel-0"):
            raise RuntimeError("request reached a terminal state before the cancellation gate")
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        else:
            raise RuntimeError("cancelled request returned a successful rollout")
        first = _completion_evidence(await completing)
        await session.suspend_for_update()

        await session.synchronize_policy("base-1")
        await session.open_policy("base-1")
        second = _completion_evidence(
            await session.generate(_request("complete-1", prompt_ids, max_tokens=4))
        )
        await session.suspend_for_update()
        return {
            "model": args.model,
            "synchronized_versions": synchronized_versions,
            "completed_rounds": [first, second],
            "cancel_acknowledged": True,
            "sample_logprobs_present": True,
            "weight_update_parity_tested": False,
        }
    finally:
        await session.aclose()


def main() -> None:
    args = _parse_args()
    print(json.dumps(asyncio.run(_run(args)), sort_keys=True))


if __name__ == "__main__":
    main()

