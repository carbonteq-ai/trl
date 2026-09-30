"""Independent scalar f-DPO values and logit derivatives, including alpha near one."""

import math

import pytest
import torch

from .test_dpo_half_precision import LogitModel, trainer_for


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [0.5, 0.999998, 1.000002, 1.5])
def test_alpha_divergence_matches_scalar_expm1_value_and_derivative(dtype, alpha):
    logits = torch.zeros(2, 2, 2, dtype=dtype, requires_grad=True)
    model = LogitModel(logits)
    trainer = trainer_for(model)
    trainer.beta = 1.0
    trainer.f_divergence_type = "alpha_divergence"
    trainer.f_alpha_divergence_coef = alpha
    score_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    policy = torch.tensor(-math.log(2), dtype=score_dtype)
    refs = torch.tensor([-math.log(2) - 0.7, -math.log(2) + 0.4], dtype=score_dtype)
    c, r = (policy - refs).double().tolist()
    scale = alpha - 1
    delta = (math.expm1(scale * c) - math.expm1(scale * r)) / scale
    expected_loss = math.log1p(math.exp(-delta))
    coefficient = 1 / (1 + math.exp(delta))
    expected = torch.zeros_like(logits)
    expected[0, 0, 0] = -0.5 * coefficient * math.exp(scale * c)
    expected[0, 0, 1] = -expected[0, 0, 0]
    expected[1, 0, 0] = 0.5 * coefficient * math.exp(scale * r)
    expected[1, 0, 1] = -expected[1, 0, 0]
    loss = trainer._compute_loss(
        model,
        {
            "input_ids": torch.zeros(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "completion_mask": torch.tensor([[0, 1], [0, 1]]),
            "ref_chosen_logps": refs[:1],
            "ref_rejected_logps": refs[1:],
        },
        False,
    )
    assert abs(loss.item() - expected_loss) < (1e-12 if dtype == torch.float64 else 2e-7)
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(gradient, expected, atol=1e-12 if dtype == torch.float64 else 2e-7, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_kl_small_score_retains_loss_value(dtype):
    logits = torch.zeros(2, 2, 2, dtype=dtype, requires_grad=True)
    model = LogitModel(logits)
    trainer = trainer_for(model)
    trainer.beta = 1e6
    trainer.f_divergence_type = "forward_kl"
    policy = torch.tensor(-math.log(2), dtype=dtype)
    refs = policy + torch.tensor([-1e-6, 0], dtype=dtype)
    c, r = (policy - refs).double().tolist()
    delta = -math.expm1(-c) + math.expm1(-r)
    expected = math.log1p(math.exp(-trainer.beta * delta))
    loss = trainer._compute_loss(
        model,
        {
            "input_ids": torch.zeros(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "completion_mask": torch.tensor([[0, 1], [0, 1]]),
            "ref_chosen_logps": refs[:1],
            "ref_rejected_logps": refs[1:],
        },
        False,
    )
    assert abs(loss.item() - expected) < (1e-12 if dtype == torch.float64 else 1e-7)
