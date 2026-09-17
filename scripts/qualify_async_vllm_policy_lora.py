#!/usr/bin/env python3
"""Qualify one real native policy-LoRA optimizer step with async vLLM Uno."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from trl.generation.async_vllm_session import AsyncGenerationRequest
from trl.generation.vllm_generation import VLLMGeneration


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--uno-adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.42)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def _sampled_logprobs(output: object) -> list[float]:
    completion = output.outputs[0]
    values: list[float] = []
    for token_id, candidates in zip(completion.token_ids, completion.logprobs, strict=True):
        candidate = candidates[token_id]
        value = float(candidate.logprob)
        if not math.isfinite(value):
            raise RuntimeError("rollout returned a non-finite target-policy logprob")
        values.append(value)
    if not values:
        raise RuntimeError("rollout returned no sampled-token logprobs")
    return values


async def _generate(session: object, version: str, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
    await session.open_policy(version)
    output = await session.generate(
        AsyncGenerationRequest(
            request_id=f"probe-{version}",
            prompt_token_ids=tuple(prompt_ids),
            sampling_params=SamplingParams(
                max_tokens=16,
                temperature=0.0,
                logprobs=0,
            ),
        )
    )
    await session.stop_admission()
    await session.suspend_for_update()
    return list(output.outputs[0].token_ids), _sampled_logprobs(output)


def _optimizer_step(model: object, tokenizer: object, learning_rate: float) -> tuple[float, float]:
    prompt = "The capital of France is"
    target = " Berlin. Berlin is the capital of France."
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    batch = tokenizer(prompt + target, return_tensors="pt", add_special_tokens=False)
    input_ids = batch.input_ids.to(model.device)
    attention_mask = batch.attention_mask.to(model.device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = torch.cat([parameter.detach().float().flatten().cpu() for parameter in trainable])
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    if not torch.isfinite(loss):
        raise RuntimeError("LoRA optimizer step produced a non-finite loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = torch.cat([parameter.detach().float().flatten().cpu() for parameter in trainable])
    changed_norm = float(torch.linalg.vector_norm(after - before))
    if not math.isfinite(changed_norm) or changed_norm <= 0:
        raise RuntimeError("LoRA optimizer step did not change any trainable parameter")
    return float(loss.detach()), changed_norm


async def _run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="eager",
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.name_or_path = args.model
    model.print_trainable_parameters()

    generation = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=tokenizer,
        mode="colocate",
        request_mode="async",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_length=args.max_model_len,
        max_num_seqs=1,
        enable_sleep_mode=True,
        speculative_config={
            "method": "uno",
            "num_speculative_tokens": 7,
            "uno_adapter": args.uno_adapter,
            "uno_mask_token_id": 250624,
            "uno_noise_mode": "random_uniform",
            "draft_sample_method": "probabilistic",
        },
        engine_kwargs={"disable_log_stats": False},
        weight_sync_mode="lora",
        trust_remote_code=True,
    )
    session = await generation.create_async_session()
    prompt_ids = tokenizer("The capital of France is", add_special_tokens=False).input_ids
    try:
        await session.synchronize_policy("0")
        before_tokens, before_logprobs = await _generate(session, "0", prompt_ids)
        loss, changed_norm = _optimizer_step(model, tokenizer, args.learning_rate)
        await session.synchronize_policy("1")
        after_tokens, after_logprobs = await _generate(session, "1", prompt_ids)
        max_logprob_delta = max(
            (abs(left - right) for left, right in zip(before_logprobs, after_logprobs, strict=False)),
            default=0.0,
        )
        changed_probe = before_tokens != after_tokens or max_logprob_delta > 1e-5
        if not changed_probe:
            raise RuntimeError("policy version advanced but the controlled rollout probe did not change")
        if len(after_tokens) >= 4 and len(set(after_tokens)) == 1:
            raise RuntimeError("post-update rollout collapsed to one repeated token")
        return {
            "passed": True,
            "model": args.model,
            "uno_adapter": args.uno_adapter,
            "update_kind": "lora",
            "rollout_materialization": "native_policy_lora_plus_composite_uno",
            "policy_versions": ["0", "1"],
            "optimizer_loss": loss,
            "learning_rate": args.learning_rate,
            "trainable_parameter_delta_norm": changed_norm,
            "before_token_ids": before_tokens,
            "after_token_ids": after_tokens,
            "before_sampled_logprobs": before_logprobs,
            "after_sampled_logprobs": after_logprobs,
            "max_common_token_logprob_delta": max_logprob_delta,
            "all_sampled_logprobs_finite": True,
            "elapsed_seconds": time.time() - started,
        }
    finally:
        await session.aclose()


def main() -> None:
    args = _arguments()
    result = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
