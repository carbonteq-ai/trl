from collections import defaultdict
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch

from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer, _validate_precomputed_advantages


class _IdentityAccelerator:
    num_processes = 1

    def gather(self, value):
        return value

    def reduce(self, value, reduction):
        return value


class _Scores(GRPOTrainer):
    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        return self.scores, torch.ones_like(self.scores), None


def _loss(advantages, mask, log_ratio=0.0, accumulation=1, reference=None, old=None, level="sequence"):
    trainer = _Scores.__new__(_Scores)
    trainer.__dict__.update(
        accelerator=_IdentityAccelerator(),
        args=SimpleNamespace(use_bias_correction_kl=False, delta=None, steps_per_generation=1),
        top_entropy_quantile=1.0,
        aux_loss_enabled=False,
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        off_policy_mask_threshold=None,
        importance_sampling_level=level,
        beta=0.0 if reference is None else 1.0,
        loss_type="grpo",
        epsilon_low=0.003,
        epsilon_high=0.004,
        _entropy_bonus_enabled=False,
        _metrics={"train": defaultdict(list)},
        model=SimpleNamespace(training=True),
        current_gradient_accumulation_steps=accumulation,
        scores=torch.full((1, len(mask)), log_ratio, requires_grad=True),
    )
    ids = torch.zeros_like(trainer.scores, dtype=torch.long)
    inputs = {
        "prompt_ids": ids[:, :1],
        "prompt_mask": torch.ones_like(ids[:, :1]),
        "completion_ids": ids,
        "completion_mask": torch.ones_like(ids),
        "tool_mask": torch.tensor([mask]),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "old_per_token_logps": torch.zeros_like(trainer.scores) if old is None else torch.tensor([old]),
    }
    if reference is not None:
        inputs["ref_per_token_logps"] = torch.tensor([reference])
    loss = trainer._compute_loss(None, inputs)
    loss.backward()
    return loss.detach(), trainer.scores.grad


@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("excluded_old", [-112.0, float("nan")])
@pytest.mark.parametrize("advantages", [[0.7], [[1.0, 0.0, -1.0]]])
def test_excluded_behavior_scores_cannot_corrupt_loss_or_gradient(level, excluded_old, advantages):
    expected_loss, expected_grad = _loss(advantages, [1, 0, 1], level=level)
    loss, grad = _loss(advantages, [1, 0, 1], old=[0.0, excluded_old, 0.0], level=level)
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(grad, expected_grad)


def test_opposite_turn_credit_does_not_cancel_and_tool_tokens_have_no_gradient():
    loss, grad = _loss([[1.0, 99.0, -1.0]], [1, 0, 1])
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(grad, torch.tensor([[-0.5, 0.0, 0.5]]))


@pytest.mark.parametrize("advantages", [[0.7], [[0.7, 0.7, 0.7]]])
def test_constant_credit_matches_existing_sequence_gradient(advantages):
    loss, grad = _loss(advantages, [1, 0, 1], accumulation=2)
    torch.testing.assert_close(loss, torch.tensor(-0.35))
    torch.testing.assert_close(grad, torch.tensor([[-0.175, 0.0, -0.175]]))


@pytest.mark.parametrize(
    ("log_ratio", "expected"),
    [(0.01, [[0.0, 0.0, 0.505025]]), (-0.01, [[-0.495025, 0.0, 0.0]])],
)
def test_sequence_clipping_preserves_advantage_sign_and_local_credit(log_ratio, expected):
    _, grad = _loss([[1.0, 99.0, -1.0]], [1, 0, 1], log_ratio=log_ratio)
    torch.testing.assert_close(grad, torch.tensor(expected), atol=1e-6, rtol=1e-5)


def test_small_kl_is_positive_and_masked_overflow_has_no_gradient():
    loss, grad = _loss([[0.0, 0.0, 0.0]], [1, 0, 1], reference=[1e-4, 112.0, -1e-4])
    delta = torch.tensor([1e-4, -1e-4], dtype=torch.float64)
    expected = (torch.expm1(delta) - delta).mean().float()
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=5e-4)
    torch.testing.assert_close(grad, torch.tensor([[-0.00005, 0.0, 0.00005]]), atol=1e-8, rtol=1e-4)


@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("exponent", range(2, 9))
def test_small_kl_value_and_gradient_match_decimal(level, sign, exponent):
    delta = float(torch.tensor(sign * 10**-exponent, dtype=torch.float32))
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float(delta)
        expected_value = float(d.exp() - 1 - d)
        expected_gradient = float(-(d.exp() - 1) / 2)
    loss, grad = _loss([[0.0, 99.0, 0.0]], [1, 0, 1], reference=[delta, 112.0, delta], level=level)
    torch.testing.assert_close(loss, torch.tensor(expected_value), rtol=5e-6, atol=0)
    torch.testing.assert_close(grad, torch.tensor([[expected_gradient, 0.0, expected_gradient]]), rtol=5e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [-1e10, -50.0, -1.0, -0.010001, -0.01, -1e-8, 0.0, 1e-8, 0.01, 0.010001, 1.0, 50.0])
def test_stable_k3_branches_match_decimal(dtype, value):
    from trl.trainer.grpo_trainer import _stable_sampled_k3

    x = torch.tensor(value, dtype=dtype, requires_grad=True)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float(x.item())
        expected = float(d.exp() - 1 - d)
        expected_gradient = float(d.exp() - 1)
    result = _stable_sampled_k3(x)
    gradient = torch.autograd.grad(result, x)[0]
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype), atol=0, rtol=tolerance)
    torch.testing.assert_close(gradient, torch.tensor(expected_gradient, dtype=dtype), atol=0, rtol=tolerance)


def test_precomputed_advantages_require_rollout_path_not_liger(tmp_path):
    with pytest.raises(ValueError, match="Liger"):
        GRPOConfig(
            output_dir=str(tmp_path),
            use_precomputed_advantages=True,
            use_liger_kernel=True,
        )


def test_precomputed_advantages_are_token_aligned_and_finite():
    assert _validate_precomputed_advantages([[1, -0.5], [0.25]], [[10, 11], [12]]) == [
        [1.0, -0.5],
        [0.25],
    ]
    with pytest.raises(ValueError, match="align"):
        _validate_precomputed_advantages([[1.0]], [[10, 11]])
    with pytest.raises(ValueError, match="finite"):
        _validate_precomputed_advantages([[float("nan")]], [[10]])
