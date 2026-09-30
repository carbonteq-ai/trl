"""DPO policy and cached/online reference scores use FP32 probabilities."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl import DPOTrainer
from trl.trainer.dpo_trainer import _preference_log_softmax


def test_nearly_certain_token_retains_small_logit_gradient():
    logits = torch.tensor([[[20.0, 4.0, 0.0]]], requires_grad=True)
    actual = _preference_log_softmax(logits, torch.tensor([[0]])).sum()
    gradient = torch.autograd.grad(actual, logits)[0]
    probabilities = logits.detach().double().softmax(-1)
    expected = -probabilities
    expected[..., 0] += 1
    torch.testing.assert_close(gradient.double(), expected, atol=1e-8, rtol=0)
    assert gradient[..., 0] > 0


class LogitModel(torch.nn.Module):
    is_gradient_checkpointing = False

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


def trainer_for(model, kind="sigmoid", cached=True):
    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.__dict__.update(
        model=model,
        ref_model=model,
        args=SimpleNamespace(gradient_checkpointing_kwargs=None),
        accelerator=SimpleNamespace(device=torch.device("cpu"), gather=lambda x: x, gather_for_metrics=lambda x: x),
        aux_loss_enabled=False,
        ld_alpha=None,
        precompute_ref_logps=cached,
        f_divergence_type="reverse_kl",
        beta=0.1,
        loss_types=[kind],
        loss_weights=[1],
        use_weighting=False,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
    )
    return trainer


def inputs():
    return {
        "input_ids": torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "completion_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]]),
        "ref_chosen_logps": torch.tensor([-2.1]),
        "ref_rejected_logps": torch.tensor([-1.9]),
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["sigmoid", "sft"])
def test_half_preference_loss_matches_float32_and_rounded_gradient(dtype, kind):
    torch.manual_seed(42)
    half = torch.randn(2, 4, 3).to(dtype).requires_grad_()
    full = half.detach().float().requires_grad_()
    expected = trainer_for(LogitModel(full), kind)._compute_loss(LogitModel(full), inputs(), False)
    actual = trainer_for(LogitModel(half), kind)._compute_loss(LogitModel(half), inputs(), False)
    expected_gradient = torch.autograd.grad(expected, full)[0].to(dtype)
    gradient = torch.autograd.grad(actual, half)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
    assert torch.count_nonzero(gradient[:, 0]) == 0  # excluded prompt prediction
    assert torch.count_nonzero(gradient[:, -1]) == 0  # no next-token label


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_reference_scores_match_float32_for_cached_and_online_paths(dtype):
    torch.manual_seed(71)
    half = torch.randn(2, 4, 3).to(dtype)
    full = half.float()
    half_trainer, full_trainer = trainer_for(LogitModel(half)), trainer_for(LogitModel(full))
    for actual, expected in zip(
        half_trainer.compute_ref_log_probs(half_trainer.model, inputs()),
        full_trainer.compute_ref_log_probs(full_trainer.model, inputs()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    policy = LogitModel(torch.randn(2, 4, 3))
    half_trainer.precompute_ref_logps = full_trainer.precompute_ref_logps = False
    actual = half_trainer._compute_loss(policy, inputs(), False)
    expected = full_trainer._compute_loss(policy, inputs(), False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
