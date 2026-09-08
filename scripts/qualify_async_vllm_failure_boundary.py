#!/usr/bin/env python
"""Qualify fail-closed async weight publication against a real vLLM server.

Run this bounded probe on a trainer GPU while a separate ``vllm serve`` process
hosts the same checkpoint on another GPU. The probe initializes the production
NCCL transfer group, then deliberately asks the server to finish a weight update
that was never started. The real server must reject that control request. TRL's
trainer lifecycle must preserve the old policy version, leave rollout admission
closed, and surface the failure. Explicit cleanup then resumes the unchanged
server and proves it still returns the original selected-token log probability.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import requests
import torch
from accelerate import PartialState
from qualify_async_vllm_changed_weight import (
    _actor_next_token,
    _final_norm,
    _server_token_logprob,
    _weight_update_info,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer
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
    parser.add_argument("--max-unchanged-logp-delta", type=float, default=1e-6)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


class _RejectingStartClient(AsyncVLLMClient):
    """Use a real, safely invalid server transition for the start phase."""

    def start_weight_update(self, timeout: int = 1800) -> None:
        self.finish_weight_update(timeout=timeout)


class _RolloutAdmissionRecorder:
    def __init__(self) -> None:
        self.prepared_versions: list[int] = []
        self.published_versions: list[int] = []

    def prepare_model_update(self, model_version: int) -> None:
        self.prepared_versions.append(model_version)

    def update_model_version(self, model_version: int) -> None:
        self.published_versions.append(model_version)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_unchanged_logp_delta < 0:
        raise ValueError("the unchanged log-probability bound must be non-negative")
    device = torch.device(args.trainer_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the trainer side of NCCL weight transfer requires an available CUDA GPU")
    torch.cuda.set_device(device)
    PartialState()

    client = _RejectingStartClient(args.server_url, args.server_timeout)
    client.wait_for_server_ready()
    server_dtype_name = client.get_dtype().removeprefix("torch.")
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
    token_id, _ = _actor_next_token(actor, prompt_ids, device)
    scoring_client = ScoringVLLMClient(base_url=args.server_url, connection_timeout=args.server_timeout)
    server_before = _server_token_logprob(scoring_client, prompt_ids, token_id)

    name, parameter = _final_norm(actor)
    transfer = WeightTransferClient(
        client,
        _weight_update_info(name, parameter),
        init_weight_transfer_timeout=int(args.server_timeout),
    )
    rollout_worker = _RolloutAdmissionRecorder()
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    trainer.model_version = 0
    trainer.model = actor
    trainer.rollout_worker = rollout_worker
    trainer.weight_transfer = transfer
    trainer._metrics = {"train": defaultdict(list)}
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        device=device,
        wait_for_everyone=lambda: None,
    )

    failure: requests.HTTPError | None = None
    transfer.init_weight_transfer()
    try:
        try:
            trainer._sync_weight()
        except requests.HTTPError as exc:
            failure = exc
        if failure is None:
            raise RuntimeError("the server accepted an invalid finish-without-start transition")
        if trainer.model_version != 0:
            raise RuntimeError(f"failed publication advanced model version to {trainer.model_version}")
        if rollout_worker.prepared_versions != [1]:
            raise RuntimeError(f"unexpected prepared versions: {rollout_worker.prepared_versions}")
        if rollout_worker.published_versions:
            raise RuntimeError(f"failed publication exposed versions: {rollout_worker.published_versions}")
    finally:
        # A run would terminate here. The bounded probe explicitly resumes the
        # unchanged server so it can establish post-failure usability and then
        # release the retained qualification worker.
        AsyncVLLMClient.resume(client)
        transfer.destroy()

    server_after = _server_token_logprob(scoring_client, prompt_ids, token_id)
    unchanged_delta = abs(server_after - server_before)
    if unchanged_delta > args.max_unchanged_logp_delta:
        raise RuntimeError(
            f"server log probability changed by {unchanged_delta:.9f} after rejected publication; "
            f"expected at most {args.max_unchanged_logp_delta:.9f}"
        )
    assert failure is not None
    return {
        "model": args.model,
        "server_url": args.server_url,
        "token_id": token_id,
        "rejected_status": failure.response.status_code if failure.response is not None else None,
        "model_version_before": 0,
        "model_version_after": trainer.model_version,
        "prepared_versions": rollout_worker.prepared_versions,
        "published_versions": rollout_worker.published_versions,
        "server_logprob_before": server_before,
        "server_logprob_after": server_after,
        "unchanged_logprob_delta": unchanged_delta,
        "failure_boundary_tested": True,
    }


def main() -> None:
    print(json.dumps(_run(_parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
