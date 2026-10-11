"""The DPO ``sft`` loss term is the mean chosen-token NLL without copying full-vocabulary logits."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trl.trainer.dpo_trainer import DPOTrainer


class _FixedLogits(nn.Module):
    """A stand-in model whose output logits are a trainable leaf, so gradients can be compared directly."""

    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


def _trainer(model, loss_types=("sft",), loss_weights=None):
    trainer = object.__new__(DPOTrainer)
    trainer.model = model
    trainer.ref_model = None
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"), gather=lambda t: t, gather_for_metrics=lambda t: t
    )
    trainer.args = SimpleNamespace(gradient_checkpointing_kwargs=None, discopop_tau=0.05)
    trainer.aux_loss_enabled = False
    trainer.beta = 0.1
    trainer.f_alpha_divergence_coef = 1.0
    trainer.f_divergence_type = "reverse_kl"
    trainer.label_smoothing = 0.0
    trainer.ld_alpha = None
    trainer.loss_types = list(loss_types)
    trainer.loss_weights = list(loss_weights or [1.0] * len(loss_types))
    trainer.precompute_ref_logps = True
    trainer.use_weighting = False
    trainer._metrics = {"train": _Metrics(), "eval": _Metrics()}
    trainer._total_train_tokens = 0
    return trainer


class _Metrics(dict):
    def __missing__(self, key):
        self[key] = []
        return self[key]


def _batch(pairs=3, length=11, vocab=37, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = (torch.randn(2 * pairs, length, vocab, generator=generator) * 3).to(dtype)
    input_ids = torch.randint(vocab, (2 * pairs, length), generator=generator)
    completion_mask = torch.zeros(2 * pairs, length, dtype=torch.long)
    # Prompt prefix, variable completion length, then right padding; chosen and rejected differ.
    for row in range(2 * pairs):
        start = 2 + row % 3
        end = length - (row % 4)
        completion_mask[row, start:end] = 1
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "completion_mask": completion_mask,
        "ref_chosen_logps": torch.zeros(pairs),
        "ref_rejected_logps": torch.zeros(pairs),
    }
    return logits, inputs


def _old_sft_loss(logits, inputs):
    """The previous implementation: cross-entropy over a boolean-indexed copy of the chosen logits."""
    chosen_logits, _ = logits[..., :-1, :].chunk(2, dim=0)
    chosen_labels, _ = inputs["input_ids"][..., 1:].chunk(2, dim=0)
    chosen_mask, _ = inputs["completion_mask"][..., 1:].chunk(2, dim=0)
    return F.cross_entropy(chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])


@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_sft_loss_and_gradient_match_cross_entropy(dtype, tolerance):
    logits, inputs = _batch(dtype=dtype)
    model = _FixedLogits(logits.clone())
    loss = _trainer(model)._compute_loss(model, inputs, return_outputs=False)
    loss.backward()

    reference_logits = logits.clone().requires_grad_(True)
    reference = _old_sft_loss(reference_logits, inputs)
    reference.backward()

    torch.testing.assert_close(loss, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(model.logits.grad, reference_logits.grad, rtol=tolerance, atol=tolerance)


def test_sft_loss_matches_cross_entropy_in_bfloat16():
    logits, inputs = _batch(dtype=torch.bfloat16, seed=3)
    model = _FixedLogits(logits.clone())
    loss = _trainer(model)._compute_loss(model, inputs, return_outputs=False)
    loss.backward()

    reference_logits = logits.clone().requires_grad_(True)
    reference = _old_sft_loss(reference_logits, inputs)
    reference.backward()

    torch.testing.assert_close(loss.float(), reference.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(model.logits.grad.float(), reference_logits.grad.float(), rtol=2e-2, atol=2e-3)


def test_sft_loss_excludes_masked_tokens_and_rejected_completions():
    logits, inputs = _batch(seed=5)
    model = _FixedLogits(logits.clone())
    _trainer(model)._compute_loss(model, inputs, return_outputs=False).backward()

    grad = model.logits.grad
    pairs = inputs["ref_chosen_logps"].numel()
    # Rejected completions never contribute to the SFT term.
    assert torch.count_nonzero(grad[pairs:]) == 0
    # The logits at position t predict token t + 1; only those predicting chosen completion tokens get gradient.
    predicts_completion = torch.zeros_like(inputs["completion_mask"][:pairs], dtype=torch.bool)
    predicts_completion[:, :-1] = inputs["completion_mask"][:pairs, 1:].bool()
    assert torch.count_nonzero(grad[:pairs][~predicts_completion]) == 0
    assert torch.all(grad[:pairs][predicts_completion].abs().sum(-1) > 0)

    # Changing the rejected logits or the masked chosen logits leaves the loss unchanged.
    perturbed = logits.clone()
    perturbed[pairs:] += 100.0
    perturbed[:pairs][~predicts_completion] = -perturbed[:pairs][~predicts_completion]
    perturbed_model = _FixedLogits(perturbed)
    baseline = _trainer(model)._compute_loss(model, inputs, return_outputs=False)
    changed = _trainer(perturbed_model)._compute_loss(perturbed_model, inputs, return_outputs=False)
    torch.testing.assert_close(changed, baseline)


def test_sft_loss_combines_with_sigmoid_and_weight():
    logits, inputs = _batch(seed=7)
    model = _FixedLogits(logits.clone())
    sigmoid = _trainer(model, loss_types=("sigmoid",))._compute_loss(model, inputs, return_outputs=False)
    combined = _trainer(model, loss_types=("sigmoid", "sft"), loss_weights=[1.0, 0.25])._compute_loss(
        model, inputs, return_outputs=False
    )
    torch.testing.assert_close(combined, sigmoid + 0.25 * _old_sft_loss(logits, inputs))


class _VocabularyAllocations(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts elements of every operator output whose last dimension is the vocabulary."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(tensor, torch.Tensor) and tensor.ndim and tensor.shape[-1] == self.vocab:
                self.elements += tensor.numel()
        return out


def test_sft_term_adds_no_vocabulary_sized_tensors():
    vocab = 37
    logits, inputs = _batch(vocab=vocab, seed=11)

    def vocabulary_elements(loss_types):
        model = _FixedLogits(logits.clone())
        with _VocabularyAllocations(vocab) as counter:
            _trainer(model, loss_types=loss_types)._compute_loss(model, inputs, return_outputs=False).backward()
        return counter.elements

    assert vocabulary_elements(("sigmoid", "sft")) == vocabulary_elements(("sigmoid",))
