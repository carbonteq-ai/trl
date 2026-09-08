#!/usr/bin/env python
"""Qualify changed-weight parity on one native AsyncLLM engine.

This bounded probe uses a CPU reference actor and one GPU inference engine. It
changes the final normalization weight through vLLM's worker RPC, resets the
prefix cache, and compares an observed-token log probability against the same
changed actor. It is a correctness gate, not a training run or benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest, WeightTransferUpdateRequest
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.engine.async_llm import AsyncLLM

from trl.generation.async_vllm_session import AsyncVllmSession
from trl.generation.vllm_generation import extract_actual_prompt_logprobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reliable software should")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--trainer-device", type=int, default=0)
    parser.add_argument("--inference-device", type=int, default=1)
    parser.add_argument("--max-mean-logp-delta", type=float, default=0.05)
    parser.add_argument("--min-observed-change", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _final_norm(actor: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    matches = [
        (name, parameter)
        for name, parameter in actor.named_parameters()
        if name.endswith("model.norm.weight")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one final normalization weight, found {[name for name, _ in matches]}")
    return matches[0]


def _actor_next_token(actor: torch.nn.Module, prompt_ids: list[int]) -> tuple[int, float]:
    with torch.inference_mode():
        logits = actor(input_ids=torch.tensor([prompt_ids], dtype=torch.long)).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_id = int(torch.argmax(logprobs).item())
        return token_id, float(logprobs[token_id].item())


def _actor_token_logprob(actor: torch.nn.Module, prompt_ids: list[int], token_id: int) -> float:
    with torch.inference_mode():
        logits = actor(input_ids=torch.tensor([prompt_ids], dtype=torch.long)).logits[0, -1].float()
        return float(torch.log_softmax(logits, dim=-1)[token_id].item())


async def _vllm_token_logprob(
    engine: AsyncLLM,
    prompt_ids: list[int],
    token_id: int,
    *,
    request_id: str,
) -> float:
    output = None
    sequence = prompt_ids + [token_id]
    async for candidate in engine.generate(
        {"prompt_token_ids": sequence},
        SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=1, detokenize=False),
        request_id,
    ):
        output = candidate
    if output is None or not output.finished:
        raise RuntimeError("vLLM parity request did not finish")
    [[value]] = extract_actual_prompt_logprobs([output], [len(prompt_ids)])
    return value


async def _init_weight_transfer(engine: AsyncLLM) -> Any:
    master_address = get_ip()
    master_port = get_open_port()
    server_init = asyncio.create_task(
        engine.init_weight_transfer_engine(
            WeightTransferInitRequest(
                init_info={
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": 2,
                }
            )
        )
    )
    try:
        trainer_group = await asyncio.to_thread(
            NCCLWeightTransferEngine.trainer_init,
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": 2,
            },
        )
        await server_init
    except BaseException:
        await asyncio.gather(server_init, return_exceptions=True)
        raise
    return trainer_group


async def _send_weight(engine: AsyncLLM, trainer_group: Any, name: str, value: torch.Tensor) -> None:
    update_info = {
        "names": [name],
        "dtype_names": [str(value.dtype).removeprefix("torch.")],
        "shapes": [list(value.shape)],
        "packed": False,
    }
    await engine.start_weight_update()
    receive = asyncio.create_task(
        engine.update_weights(WeightTransferUpdateRequest(update_info=update_info))
    )
    await asyncio.sleep(0)
    try:
        await asyncio.to_thread(
            NCCLWeightTransferEngine.trainer_send_weights,
            iter(((name, value),)),
            NCCLTrainerSendWeightsArgs(group=trainer_group, packed=False),
        )
        await receive
    except BaseException:
        if not receive.done():
            receive.cancel()
        await asyncio.gather(receive, return_exceptions=True)
        raise
    await engine.finish_weight_update()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_mean_logp_delta <= 0 or args.min_observed_change <= 0:
        raise ValueError("parity and observed-change bounds must be positive")
    if args.trainer_device < 0 or args.inference_device < 0:
        raise ValueError("CUDA device indices cannot be negative")
    if args.trainer_device == args.inference_device:
        raise ValueError("NCCL changed-weight qualification requires distinct trainer and inference GPUs")
    visible_devices = torch.cuda.device_count()
    if visible_devices <= max(args.trainer_device, args.inference_device):
        raise RuntimeError(
            "NCCL changed-weight qualification requires two visible GPUs; "
            f"found {visible_devices}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    actor = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=torch.float32,
        device_map="cpu",
    )
    actor.eval()
    token_id, actor_before = _actor_next_token(actor, prompt_ids)

    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            enable_sleep_mode=True,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            device_ids=[args.inference_device],
            gpu_memory_utilization=args.gpu_memory_utilization,
            logprobs_mode="raw_logprobs",
            disable_log_stats=True,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )
    )
    torch.cuda.set_device(args.trainer_device)
    trainer_group: Any | None = None
    update_weight: tuple[str, torch.Tensor] | None = None

    async def synchronize(version: str) -> None:
        if version == "base-0":
            return
        if version != "changed-1" or update_weight is None or trainer_group is None:
            raise RuntimeError(f"unexpected policy synchronization request {version!r}")
        name, value = update_weight
        await _send_weight(engine, trainer_group, name, value)
        if not await engine.reset_prefix_cache():
            raise RuntimeError("vLLM did not acknowledge prefix-cache reset after changed weights")

    session = AsyncVllmSession(engine, synchronize, sleep_level=1)
    try:
        trainer_group = await _init_weight_transfer(engine)
        await session.synchronize_policy("base-0")
        await session.open_policy("base-0")
        vllm_before = await _vllm_token_logprob(engine, prompt_ids, token_id, request_id="parity-base")
        await session.suspend_for_update()

        name, parameter = _final_norm(actor)
        with torch.no_grad():
            parameter.zero_()
        update_weight = (
            name,
            parameter.detach().to(device=f"cuda:{args.trainer_device}", dtype=torch.float16),
        )
        actor_after = _actor_token_logprob(actor, prompt_ids, token_id)

        await session.synchronize_policy("changed-1")
        await session.open_policy("changed-1")
        vllm_after = await _vllm_token_logprob(engine, prompt_ids, token_id, request_id="parity-changed")
        await session.suspend_for_update()

        parity_delta = abs(actor_after - vllm_after)
        observed_change = abs(vllm_after - vllm_before)
        if not all(math.isfinite(value) for value in (actor_before, vllm_before, actor_after, vllm_after)):
            raise RuntimeError("changed-weight parity produced non-finite log probabilities")
        if parity_delta > args.max_mean_logp_delta:
            raise RuntimeError(
                f"changed actor/sampler log-probability delta {parity_delta:.6f} exceeds "
                f"{args.max_mean_logp_delta:.6f}"
            )
        if observed_change < args.min_observed_change:
            raise RuntimeError(
                f"sampler log probability changed by only {observed_change:.6f}; "
                f"expected at least {args.min_observed_change:.6f}"
            )
        return {
            "model": args.model,
            "updated_weight": name,
            "token_id": token_id,
            "actor_logprob_before": actor_before,
            "sampler_logprob_before": vllm_before,
            "actor_logprob_after": actor_after,
            "sampler_logprob_after": vllm_after,
            "changed_weight_parity_delta": parity_delta,
            "observed_sampler_change": observed_change,
            "max_mean_logp_delta": args.max_mean_logp_delta,
            "changed_weight_parity_tested": True,
        }
    finally:
        try:
            if trainer_group is not None:
                trainer_group.destroy()
        finally:
            await session.aclose()


def main() -> None:
    print(json.dumps(asyncio.run(_run(_parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
