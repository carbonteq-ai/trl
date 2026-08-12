"""Exact-token replay under an XGrammar-constrained probability space.

This module is imported inside vLLM worker processes.  It deliberately keeps
only per-token scalar evidence; completion-by-vocabulary tensors and grammar
bitmasks are never retained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


_RESULTS: dict[str, _ReplayState] = {}
_RESULTS_LOCK = Lock()


def _decode_allowed_tokens(bitmask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Return a CPU boolean vector for bool or XGrammar compressed masks."""
    row = bitmask[0].cpu()
    if row.dtype == torch.bool:
        return row[:vocab_size].clone()
    bit_weights = 1 << torch.arange(32, dtype=torch.int64)
    return (
        (row.to(torch.int64).unsqueeze(1) & bit_weights.unsqueeze(0)) != 0
    ).reshape(-1)[:vocab_size]


def _position_digest(schema_digest: str, completion_ids: list[int], position: int) -> str:
    payload = {
        "completion_prefix": completion_ids[:position],
        "position": position,
        "schema_digest": schema_digest,
        "xgrammar_contract": "0.2.3-json-schema-any-whitespace-false",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class _ReplayState:
    request_id: str
    completion_ids: list[int]
    schema_digest: str
    allowed_token_ids: list[torch.Tensor | None]
    allowed_bitmasks: list[torch.Tensor | None]
    precomputed_allowed_counts: list[int]
    logprobs: list[torch.Tensor] = field(default_factory=list)
    allowed_counts: list[torch.Tensor] = field(default_factory=list)
    allowed_set_digests: list[str] = field(default_factory=list)


class ConstrainedReplayLogitsProcessor(AdapterLogitsProcessor):
    """vLLM adapter processor that scores and then forces an exact completion."""

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        args = sampling_params.extra_args or {}
        if "constrained_replay" not in args:
            return
        spec = args["constrained_replay"]
        required = {"request_id", "completion_ids", "json_schema", "schema_digest"}
        missing = required.difference(spec)
        if missing:
            raise ValueError(f"constrained replay is missing fields: {sorted(missing)}")

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        import xgrammar as xgr
        from transformers import AutoTokenizer

        # AdapterLogitsProcessor has useful state plumbing but cooperative
        # inheritance is not available because this class is loaded by name.
        super().__init__(vllm_config, device, is_pin_memory)
        self._device = device
        model_config = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
        vocab_size = model_config.get_vocab_size()
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self._compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=True)
        self._vocab_size = vocab_size
        self._xgr = xgr

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: Any):
        spec = (params.extra_args or {}).get("constrained_replay")
        if spec is None:
            return None
        request_id = str(spec["request_id"])
        completion_ids = [int(token_id) for token_id in spec["completion_ids"]]
        schema_text = json.dumps(spec["json_schema"], sort_keys=True, separators=(",", ":"))
        observed_digest = hashlib.sha256(schema_text.encode()).hexdigest()
        if observed_digest != spec["schema_digest"]:
            raise ValueError("constrained replay schema digest mismatch")
        context = self._compiler.compile_json_schema(schema_text, any_whitespace=False)
        matcher = self._xgr.GrammarMatcher(context)
        bitmask = self._xgr.allocate_token_bitmask(1, self._vocab_size)
        allowed_token_ids: list[torch.Tensor | None] = []
        allowed_bitmasks: list[torch.Tensor | None] = []
        allowed_counts: list[int] = []
        # Walk the known completion once on CPU. Doing grammar state transitions
        # inside vLLM's per-request GPU callback serializes Python/XGrammar work
        # across the resident wave and makes long exact-token replay unusably slow.
        for position, selected in enumerate(completion_ids):
            bitmask.fill_(0)
            matcher.fill_next_token_bitmask(bitmask, 0)
            allowed = _decode_allowed_tokens(bitmask, self._vocab_size)
            if not bool(allowed[selected]):
                raise ValueError(f"selected token {selected} is disallowed at position {position}")
            count = int(allowed.sum().item())
            allowed_counts.append(count)
            if count <= 1024:
                allowed_token_ids.append(
                    torch.nonzero(allowed, as_tuple=False).flatten().to(self._device)
                )
                allowed_bitmasks.append(None)
            else:
                allowed_token_ids.append(None)
                allowed_bitmasks.append(bitmask.clone().to(self._device))
            if not matcher.accept_token(selected):
                raise ValueError(f"selected token {selected} was rejected at position {position}")
        state = _ReplayState(
            request_id,
            completion_ids,
            observed_digest,
            allowed_token_ids,
            allowed_bitmasks,
            allowed_counts,
        )
        with _RESULTS_LOCK:
            if request_id in _RESULTS:
                raise ValueError(f"duplicate constrained replay request id: {request_id}")
            _RESULTS[request_id] = state

        def process(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            position = len(output_ids)
            if position >= len(completion_ids):
                raise ValueError(f"constrained replay exceeded completion for {request_id}")
            if position and int(output_ids[-1]) != completion_ids[position - 1]:
                raise ValueError(
                    f"constrained replay prefix diverged at {position - 1}: "
                    f"{output_ids[-1]} != {completion_ids[position - 1]}"
                )
            selected = completion_ids[position]
            float_logits = logits.float()
            allowed_ids = state.allowed_token_ids[position]
            if allowed_ids is not None:
                denominator = torch.logsumexp(float_logits.index_select(0, allowed_ids), dim=-1)
            else:
                constrained = float_logits.clone()
                bitmask_at_position = state.allowed_bitmasks[position]
                assert bitmask_at_position is not None
                self._xgr.apply_token_bitmask_inplace(
                    constrained.unsqueeze(0), bitmask_at_position
                )
                denominator = torch.logsumexp(constrained, dim=-1)
            state.logprobs.append(float_logits[selected] - denominator)
            state.allowed_counts.append(
                torch.tensor(state.precomputed_allowed_counts[position], device=self._device)
            )
            state.allowed_set_digests.append(_position_digest(state.schema_digest, completion_ids, position))
            logits.fill_(float("-inf"))
            logits[selected] = 0
            return logits

        return process


def collect_constrained_replay_results(request_ids: list[str]) -> list[dict[str, Any]]:
    """Collect completed replay evidence in request order and release worker state."""
    results: list[dict[str, Any]] = []
    with _RESULTS_LOCK:
        states = []
        for request_id in request_ids:
            state = _RESULTS.pop(request_id, None)
            if state is None:
                raise ValueError(f"missing constrained replay result: {request_id}")
            states.append(state)
    for state in states:
        if len(state.logprobs) != len(state.completion_ids):
            raise ValueError(
                f"incomplete constrained replay for {state.request_id}: "
                f"{len(state.logprobs)} != {len(state.completion_ids)}"
            )
        results.append(
            {
                "request_id": state.request_id,
                "completion_ids": state.completion_ids,
                "logprobs": torch.stack(state.logprobs).detach().cpu().tolist(),
                "allowed_counts": torch.stack(state.allowed_counts).detach().cpu().tolist(),
                "allowed_set_digests": state.allowed_set_digests,
                "schema_digest": state.schema_digest,
            }
        )
    return results
