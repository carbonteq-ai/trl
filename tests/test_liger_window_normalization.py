# Copyright 2026 CarbonTeq
# Licensed under the Apache License, Version 2.0.
"""Offline checks of the native adapter's Liger 0.8 window rescaling."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl import GRPOTrainer


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo", "grpo"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("world_size", [1, 2])
def test_liger_window_normalization_preserves_loss_and_gradient(loss_type, training, world_size):
    # Two unequal microbatches are one generation window. For two ranks the
    # other rank contributes more tokens, so local and global means differ.
    global_window_tokens = 12 if world_size == 2 else 5
    parameter = torch.tensor(1.0, requires_grad=True)
    losses = []
    for active_tokens, global_micro_tokens in [(2, 6), (3, 6)]:
        mask = torch.tensor([[1, 1, int(active_tokens == 3)]])
        mean_micro_tokens = global_micro_tokens / world_size if world_size == 2 else active_tokens

        def fused_loss(mask=mask, active_tokens=active_tokens, **kwargs):
            # Liger >=0.8.2 receives the global generation-window count.
            # The native kernel tests below independently check its gradients.
            assert torch.equal(kwargs["attention_mask"], mask)
            assert kwargs["num_items_in_batch"] == global_window_tokens
            denominator = global_window_tokens / world_size if loss_type != "grpo" else active_tokens
            return parameter * active_tokens / denominator, (torch.tensor(0.0),)

        model = SimpleNamespace(training=training, lm_head=SimpleNamespace(weight=torch.ones(2, 2), bias=None))
        trainer = SimpleNamespace(
            model=model,
            loss_type=loss_type,
            beta=0.0,
            args=SimpleNamespace(steps_per_generation=4),
            current_gradient_accumulation_steps=2,
            accelerator=SimpleNamespace(
                num_processes=world_size,
                gather=lambda value: value,
                reduce=lambda value, reduction, count=mean_micro_tokens: torch.tensor(count),
            ),
            _get_last_hidden_state=lambda *args: torch.ones(1, 3, 2),
            _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
            liger_loss=fused_loss,
        )
        inputs = {
            "prompt_ids": torch.ones(1, 1, dtype=torch.long),
            "prompt_mask": torch.ones(1, 1),
            "completion_ids": torch.ones(1, 3, dtype=torch.long),
            "completion_mask": torch.ones(1, 3),
            "tool_mask": mask,
            "advantages": torch.ones(1),
            "num_items_in_batch": global_window_tokens,
        }
        losses.append(GRPOTrainer.compute_liger_loss(trainer, model, inputs))
    loss = sum(losses)
    loss.backward()
    if loss_type == "grpo":
        expected = 1.0 if training else 2.0
    else:
        expected = 5 / (global_window_tokens / world_size) * (2 if training else 1)
    assert loss.item() == pytest.approx(expected)
    assert parameter.grad.item() == pytest.approx(expected)


@pytest.mark.parametrize("beta", [0.0, 0.15])
def test_native_liger_dapo_matches_window_objective_gradient(beta):
    liger = pytest.importorskip("liger_kernel.chunked_loss.grpo_loss")
    torch.manual_seed(7)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    advantages = torch.tensor([1.0, -0.3])
    logps = (hidden @ weight.T).log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    old = logps.detach() + torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]])
    ref = old + 0.2
    ratio = (logps - old).exp()
    policy = -torch.minimum(ratio * advantages[:, None], ratio.clamp(0.8, 1.2) * advantages[:, None])
    kl = (ref - logps).exp() - (ref - logps) - 1
    expected = ((policy + beta * kl) * mask).sum() / mask.sum() * 2
    expected_grads = torch.autograd.grad(expected, (hidden, weight), retain_graph=True)
    fused_loss = liger.LigerFusedLinearGRPOLoss(beta=beta, compiled=False, loss_type="dapo")
    actual = torch.tensor(0.0)
    for row in range(2):
        model = SimpleNamespace(training=True, lm_head=SimpleNamespace(weight=weight, bias=None))
        trainer = SimpleNamespace(
            model=model,
            loss_type="dapo",
            beta=beta,
            args=SimpleNamespace(steps_per_generation=4),
            current_gradient_accumulation_steps=2,
            accelerator=SimpleNamespace(num_processes=1, gather=lambda x: x, reduce=lambda x, reduction: x),
            _get_last_hidden_state=lambda *args, row=row: hidden[row : row + 1],
            _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
            liger_loss=fused_loss,
        )
        inputs = {
            "prompt_ids": torch.ones(1, 1, dtype=torch.long),
            "prompt_mask": torch.ones(1, 1),
            "completion_ids": tokens[row : row + 1],
            "completion_mask": torch.ones(1, 3),
            "tool_mask": mask[row : row + 1],
            "advantages": advantages[row : row + 1],
            "num_items_in_batch": mask.sum(),
            "old_per_token_logps": old[row : row + 1],
            "ref_per_token_logps": ref[row : row + 1],
        }
        actual = actual + GRPOTrainer.compute_liger_loss(trainer, model, inputs)
    actual_grads = torch.autograd.grad(actual, (hidden, weight))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def _distributed_liger_worker(rank, rendezvous):
    import torch.distributed as dist
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    try:
        torch.manual_seed(17)
        hidden = torch.randn(4, 3, 4)
        weight = torch.randn(7, 4, requires_grad=True)
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]])
        advantages = torch.tensor([1.0, -0.3, 0.8, -0.5])
        logps = (hidden @ weight.T).log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        old, ref = logps.detach(), logps.detach() + 0.2
        ratio = (logps - old).exp()
        policy = -torch.minimum(ratio * advantages[:, None], ratio.clamp(0.8, 1.2) * advantages[:, None])
        kl = (ref - logps).exp() - (ref - logps) - 1
        expected = ((policy + 0.15 * kl) * mask).sum() / mask.sum() * 2
        expected_grad = torch.autograd.grad(expected, weight)[0]

        def reduce(value, reduction):
            assert reduction == "mean"
            value = value.clone()
            dist.all_reduce(value)
            return value / 2

        actual = torch.tensor(0.0)
        for row in [rank, rank + 2]:
            model = SimpleNamespace(training=True, lm_head=SimpleNamespace(weight=weight, bias=None))
            trainer = SimpleNamespace(
                model=model,
                loss_type="dapo",
                beta=0.15,
                args=SimpleNamespace(steps_per_generation=4),
                current_gradient_accumulation_steps=2,
                accelerator=SimpleNamespace(num_processes=2, gather=lambda x: x, reduce=reduce),
                _get_last_hidden_state=lambda *args, row=row: hidden[row : row + 1],
                _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
                liger_loss=LigerFusedLinearGRPOLoss(beta=0.15, compiled=False, loss_type="dapo"),
            )
            inputs = {
                "prompt_ids": torch.ones(1, 1, dtype=torch.long),
                "prompt_mask": torch.ones(1, 1),
                "completion_ids": tokens[row : row + 1],
                "completion_mask": torch.ones(1, 3),
                "tool_mask": mask[row : row + 1],
                "advantages": advantages[row : row + 1],
                "num_items_in_batch": mask.sum(),
                "old_per_token_logps": old[row : row + 1],
                "ref_per_token_logps": ref[row : row + 1],
            }
            actual = actual + GRPOTrainer.compute_liger_loss(trainer, model, inputs)
        actual_grad = torch.autograd.grad(actual, weight)[0]
        dist.all_reduce(actual_grad)
        actual_grad /= 2
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_liger_matches_complete_window_gradient(tmp_path):
    pytest.importorskip("liger_kernel.chunked_loss.grpo_loss")
    torch.multiprocessing.spawn(_distributed_liger_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2)
