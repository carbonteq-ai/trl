"""Sampled-token IW-OPD half logits retain FP32 probability arithmetic."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl.experimental.iw_opd.iw_opd_trainer import IWOPDTrainer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_logits_match_float32_loss_and_rounded_gradient(dtype):
    trainer = IWOPDTrainer.__new__(IWOPDTrainer)
    trainer.__dict__.update(
        temperature=0.7,
        iw_opd_gamma=0.5,
        iw_opd_epsilon=1e-8,
        model=SimpleNamespace(training=True),
        _metrics={"train": defaultdict(list)},
    )
    logits = torch.tensor([[[0.3, -1.2, 0.8], [-0.9, 0.4, 1.3], [1.7, 0.2, -0.3]]], dtype=dtype)
    tokens = torch.tensor([[2, 0, 1]])
    labels = torch.tensor([[2, -100, 1]])
    rollout = torch.tensor([[-1.1, float("nan"), -1.8]])
    teacher = torch.tensor([[-0.8, float("nan"), -2.1]])
    reference = logits.float().requires_grad_()
    expected = trainer._compute_iw_opd_loss(reference, tokens, labels, teacher, rollout_logprobs=rollout)
    expected_gradient = torch.autograd.grad(expected, reference)[0].to(dtype)
    leaf = logits.clone().requires_grad_()
    actual = trainer._compute_iw_opd_loss(leaf, tokens, labels, teacher, rollout_logprobs=rollout)
    gradient = torch.autograd.grad(actual, leaf)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
    assert torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1]))
