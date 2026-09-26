"""On-policy importance sampling from the training forward (CarbonTeq)."""

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl.trainer.grpo_trainer import GRPOTrainer


def _trainer(mode):
    trainer = SimpleNamespace(
        vllm_importance_sampling_mode=mode,
        vllm_importance_sampling_clip_min=0.5,
        vllm_importance_sampling_clip_max=2.0,
        _deferred_is_stats=[],
        _metrics={"train": defaultdict(list)},
        model=SimpleNamespace(training=True),
        accelerator=SimpleNamespace(gather=lambda t: t),
    )
    for name in ("_vllm_importance_sampling_ratio", "_defer_importance_sampling", "_flush_deferred_importance_sampling"):
        setattr(trainer, name, getattr(GRPOTrainer, name).__get__(trainer))
    return trainer


def _rows():
    torch.manual_seed(0)
    actor = torch.randn(4, 6) * 0.3 - 1.0
    sampling = actor + torch.randn(4, 6) * 0.4
    sampling[1, 2] = float("nan")
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]]).float()
    return actor, sampling, mask


@pytest.mark.parametrize("mode", ["token_truncate", "sequence_truncate", "token_mask", "sequence_mask"])
def test_deferred_ratio_and_metrics_match_scoring_over_the_whole_batch(mode):
    actor, sampling, mask = _rows()
    whole = _trainer(mode)
    expected_ratio, expected_clamp, sequence_level = whole._vllm_importance_sampling_ratio(actor, sampling, mask)

    deferred = _trainer(mode)
    ratios = []
    for row in range(4):  # one micro-batch per row, as with per_device_train_batch_size=1
        inputs = {"sampling_per_token_logps": sampling[row : row + 1]}
        deferred._defer_importance_sampling(inputs, actor[row : row + 1].clone().requires_grad_(), mask[row : row + 1])
        ratios.append(inputs["importance_sampling_ratio"])
    torch.testing.assert_close(torch.cat(ratios), expected_ratio)

    deferred._flush_deferred_importance_sampling("train")
    metrics = deferred._metrics["train"]
    token_mask = mask.bool()
    delta = (actor - sampling).abs()[token_mask & ~torch.isnan(sampling)]
    flat = expected_ratio.flatten() if sequence_level else expected_ratio[token_mask]
    assert metrics["sampling/sampling_logp_difference/mean"][0] == pytest.approx(delta.mean().item())
    assert metrics["sampling/sampling_logp_difference/max"][0] == pytest.approx(delta.max().item())
    assert metrics["sampling/importance_sampling_ratio/mean"][0] == pytest.approx(flat.mean().item())
    assert metrics["sampling/importance_sampling_ratio/min"][0] == pytest.approx(flat.min().item())
    assert metrics["sampling/importance_sampling_ratio/max"][0] == pytest.approx(flat.max().item())
    clamped = expected_clamp.float().mean() if sequence_level else (expected_clamp & token_mask).float().sum() / mask.sum()
    assert metrics["sampling/importance_sampling_ratio/clamped_fraction"][0] == pytest.approx(clamped.item())
    assert deferred._deferred_is_stats == []


def test_evaluation_fills_the_ratio_without_recording_training_statistics():
    actor, sampling, mask = _rows()
    trainer = _trainer("token_truncate")
    trainer.model.training = False
    inputs = {"sampling_per_token_logps": sampling}
    trainer._defer_importance_sampling(inputs, actor, mask)
    assert "importance_sampling_ratio" in inputs
    assert trainer._deferred_is_stats == []
    assert math.isfinite(inputs["importance_sampling_ratio"].sum().item())
