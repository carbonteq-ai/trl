"""Opt-in native sampler receipts preserve processed scores and EOS alignment."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import trl.trainer.grpo_trainer as module
from trl.trainer.grpo_trainer import GRPOTrainer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_processed_generation_scores_align_after_eos(monkeypatch, dtype):
    # Row 0 ends before row 1. Its post-EOS padding must not acquire a score.
    sequences = torch.tensor([[4, 5, 1, 0, 0], [0, 6, 2, 1, 0]])
    scores = tuple(torch.tensor([[0., 2., -float("inf")], [1., 3., 2.]], dtype=dtype) for _ in range(3))
    model = SimpleNamespace(training=True, generate=Mock(return_value=SimpleNamespace(sequences=sequences, scores=scores)))
    monkeypatch.setattr(module, "profiling_context", lambda *args: nullcontext())
    monkeypatch.setattr(module, "unwrap_model_for_generation", lambda *args, **kwargs: nullcontext(model))
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.__dict__.update(accelerator=SimpleNamespace(device=torch.device("cpu")), model=model, model_wrapped=model,
        args=SimpleNamespace(ds3_gather_for_generation=False), use_vllm=False, use_transformers_continuous_batching=False,
        _tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0), generation_config=object(), generation_kwargs={},
        _is_vlm=False, _dist=SimpleNamespace(summon_full_params=lambda *args, **kwargs: nullcontext()),
        _metrics={"train": defaultdict(list)})
    monkeypatch.setattr(GRPOTrainer.__mro__[1], "_prepare_inputs", lambda self, inputs: inputs)
    tokens, logprobs = trainer._generate_single_turn([[4, 5], [6]], None, {}, return_generation_logprobs=True)
    assert tokens == [[1, 0], [2, 1, 0]]
    assert list(map(len, logprobs)) == [2, 3]
    for row, sampled in enumerate(tokens):
        for step, token in enumerate(sampled):
            # Independent normalization directly from processed values.
            values = scores[step][row].double()
            expected = values[token] - torch.logsumexp(values, dim=0)
            assert logprobs[row][step] == pytest.approx(expected.item(), abs=2e-7)
    assert model.generate.call_args.kwargs["output_scores"] is True
    assert model.generate.call_args.kwargs["return_dict_in_generate"] is True
    model.generate.return_value = sequences
    assert trainer._generate_single_turn([[4, 5], [6]], None, {}) == (tokens, None)
    assert "output_scores" not in model.generate.call_args.kwargs


def test_continuous_batching_rejects_unavailable_sampler_receipts():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.use_transformers_continuous_batching = True
    with pytest.raises(ValueError, match="continuous batching"):
        trainer._generate_single_turn([[1]], None, {}, return_generation_logprobs=True)
