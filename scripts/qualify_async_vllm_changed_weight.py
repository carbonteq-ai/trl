#!/usr/bin/env python
"""Qualify actor/server parity after a real asynchronous weight update.

Run this bounded probe on a trainer GPU while a separate ``vllm serve`` process
hosts the same checkpoint on another GPU. It deliberately uses async GRPO's
HTTP/NCCL clients instead of creating a second in-process engine, so the probe
exercises the production topology and works across hosts with routable network
interfaces.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.async_grpo.vllm_client import VLLMClient as AsyncVLLMClient
from trl.experimental.async_grpo.weight_transfer import WeightTransferClient
from trl.generation.vllm_client import VLLMClient as ScoringVLLMClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--prompt", default="Reliable software should")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--server-timeout", type=float, default=300.0)
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


def _actor_next_token(actor: torch.nn.Module, prompt_ids: list[int], device: torch.device) -> tuple[int, float]:
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = actor(input_ids=input_ids).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_id = int(torch.argmax(logprobs).item())
        return token_id, float(logprobs[token_id].item())


def _actor_token_logprob(
    actor: torch.nn.Module,
    prompt_ids: list[int],
    token_id: int,
    device: torch.device,
) -> float:
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = actor(input_ids=input_ids).logits[0, -1].float()
        return float(torch.log_softmax(logits, dim=-1)[token_id].item())


def _server_token_logprob(
    client: ScoringVLLMClient,
    prompt_ids: list[int],
    token_id: int,
) -> float:
    result = client.get_sequence_logprobs(
        [prompt_ids + [token_id]],
        [len(prompt_ids)],
        top_logprobs=1,
        temperature=1.0,
    )
    [[[value]]] = result["actual_logprobs"]
    return float(value)


def _weight_update_info(name: str, parameter: torch.Tensor) -> dict[str, list[Any]]:
    return {
        "names": [name],
        "dtype_names": [str(parameter.dtype).removeprefix("torch.")],
        "shapes": [list(parameter.shape)],
        "packed": True,
    }


def _one_weight(name: str, parameter: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    yield name, parameter


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_mean_logp_delta <= 0 or args.min_observed_change <= 0:
        raise ValueError("parity and observed-change bounds must be positive")
    device = torch.device(args.trainer_device)
    if device.type != "cuda":
        raise ValueError("the trainer side of NCCL weight transfer requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("the trainer side of NCCL weight transfer requires an available CUDA GPU")
    torch.cuda.set_device(device)

    async_client = AsyncVLLMClient(args.server_url, args.server_timeout)
    async_client.wait_for_server_ready()
    server_dtype_name = async_client.get_dtype().removeprefix("torch.")
    try:
        actor_dtype = getattr(torch, server_dtype_name)
    except AttributeError as exc:
        raise RuntimeError(f"unsupported server dtype {server_dtype_name!r}") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    actor = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=actor_dtype,
    ).to(device)
    actor.eval()
    token_id, actor_before = _actor_next_token(actor, prompt_ids, device)

    scoring_client = ScoringVLLMClient(
        base_url=args.server_url,
        connection_timeout=args.server_timeout,
    )
    server_before = _server_token_logprob(scoring_client, prompt_ids, token_id)
    base_parity_delta = abs(actor_before - server_before)
    if base_parity_delta > args.max_mean_logp_delta:
        raise RuntimeError(
            f"base actor/server log-probability delta {base_parity_delta:.6f} exceeds "
            f"{args.max_mean_logp_delta:.6f}; refusing to attribute later differences to weight transfer"
        )

    name, parameter = _final_norm(actor)
    transfer = WeightTransferClient(
        async_client,
        _weight_update_info(name, parameter),
        init_weight_transfer_timeout=int(args.server_timeout),
    )
    paused = False
    try:
        transfer.init_weight_transfer()
        transfer.pause()
        paused = True
        with torch.no_grad():
            parameter.zero_()
        actor_after = _actor_token_logprob(actor, prompt_ids, token_id, device)
        transfer.send_weights(_one_weight(name, parameter.detach()))
        scoring_client.reset_prefix_cache()
        transfer.resume()
        paused = False
        server_after = _server_token_logprob(scoring_client, prompt_ids, token_id)
    finally:
        if paused:
            transfer.resume()
        transfer.destroy()

    changed_parity_delta = abs(actor_after - server_after)
    observed_change = abs(server_after - server_before)
    values = (actor_before, server_before, actor_after, server_after)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("changed-weight parity produced non-finite log probabilities")
    if changed_parity_delta > args.max_mean_logp_delta:
        raise RuntimeError(
            f"changed actor/server log-probability delta {changed_parity_delta:.6f} exceeds "
            f"{args.max_mean_logp_delta:.6f}"
        )
    if observed_change < args.min_observed_change:
        raise RuntimeError(
            f"server log probability changed by only {observed_change:.6f}; "
            f"expected at least {args.min_observed_change:.6f}"
        )
    return {
        "model": args.model,
        "server_url": args.server_url,
        "updated_weight": name,
        "token_id": token_id,
        "actor_logprob_before": actor_before,
        "server_logprob_before": server_before,
        "base_parity_delta": base_parity_delta,
        "actor_logprob_after": actor_after,
        "server_logprob_after": server_after,
        "changed_weight_parity_delta": changed_parity_delta,
        "observed_server_change": observed_change,
        "max_mean_logp_delta": args.max_mean_logp_delta,
        "changed_weight_parity_tested": True,
    }


def main() -> None:
    print(json.dumps(_run(_parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
