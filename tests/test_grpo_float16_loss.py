"""GRPO and RLOO score float16 models and compute their losses in float32.

Under float16 training the model's logits are float16. TRL used to compute the
per-token log-probabilities and entropies in that dtype, and every loss term
derived from them (the k3 KL `exp(ref - logp) - (ref - logp) - 1`, token and
sequence importance ratios, masked sums) with them. `exp` of a log-ratio above
ln(65504) ~= 11.09 is infinite in float16, and an infinite term on a masked
token (a tool-output token of a multi-turn completion, which the policy never
sampled) times the zero mask is NaN, so the loss and every gradient were NaN.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from trl import GRPOTrainer, RLOOTrainer


def _tiny_model(dtype):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, dtype=dtype).eval()


class _Accelerator:
    """The single-process slice of Accelerate the scoring and loss read."""

    num_processes = 1
    sync_gradients = True
    is_main_process = True

    def unwrap_model(self, model):
        return model

    def gather(self, tensor):
        return tensor

    def gather_for_metrics(self, tensor):
        return tensor

    def reduce(self, tensor, reduction="sum"):
        return tensor


def _bare(trainer_type, **attributes):
    trainer = trainer_type.__new__(trainer_type)
    trainer.__dict__.update(
        accelerator=_Accelerator(),
        args=SimpleNamespace(gradient_checkpointing=False, report_to=[]),
        model_kwarg_keys={"input_ids", "attention_mask", "logits_to_keep"},
        _is_vlm=False,
        _entropy_bonus_enabled=False,
        temperature=0.7,
        **attributes,
    )
    return trainer


def _score(trainer, model, ids):
    with torch.no_grad():
        logps, entropies, _ = trainer._get_per_token_logps_and_entropies(
            model, ids, torch.ones_like(ids), 4, compute_entropy=True
        )
    return logps, entropies


def _expected(model, ids, temperature=0.7):
    """Float32 log-probs and entropies of the model's own logits."""
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, -5:-1, :].float() / temperature
    logps = logits.log_softmax(-1)
    return logps.gather(-1, ids[:, -4:].unsqueeze(-1)).squeeze(-1), -(logps.exp() * logps).sum(-1)


@pytest.mark.parametrize("logits_chunk_size", [None, 2], ids=["full", "chunked"])
def test_grpo_scores_a_float16_model_in_float32(logits_chunk_size):
    model = _tiny_model(torch.float16)
    ids = torch.randint(0, 64, (2, 9), generator=torch.Generator().manual_seed(1))
    logps, entropies = _score(_bare(GRPOTrainer, logits_chunk_size=logits_chunk_size), model, ids)
    assert (logps.dtype, entropies.dtype) == (torch.float32, torch.float32)
    expected_logps, expected_entropies = _expected(model, ids)
    torch.testing.assert_close(logps, expected_logps, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropies, expected_entropies, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("logits_chunk_size", [None, 2], ids=["full", "chunked"])
def test_grpo_scoring_keeps_bfloat16(logits_chunk_size):
    model = _tiny_model(torch.bfloat16)
    ids = torch.randint(0, 64, (2, 9), generator=torch.Generator().manual_seed(1))
    logps, entropies = _score(_bare(GRPOTrainer, logits_chunk_size=logits_chunk_size), model, ids)
    assert (logps.dtype, entropies.dtype) == (torch.bfloat16, torch.bfloat16)


def test_rloo_scores_a_float16_model_in_float32():
    model = _tiny_model(torch.float16)
    ids = torch.randint(0, 64, (2, 9), generator=torch.Generator().manual_seed(1))
    logps, entropies = _score(_bare(RLOOTrainer), model, ids)
    assert (logps.dtype, entropies.dtype) == (torch.float32, torch.float32)
    torch.testing.assert_close(logps, _expected(model, ids)[0], atol=1e-5, rtol=1e-5)


class _Float16Scores(GRPOTrainer):
    """A scorer that still returns float16 (a subclass or an external scorer)."""

    policy = torch.tensor([[-0.5, -1.0, -14.0]], dtype=torch.float16)

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        return self.policy.clone().requires_grad_(), torch.full_like(self.policy, 1.5), None


@pytest.mark.parametrize(
    ("importance_sampling_level", "loss_type"),
    [("sequence", "grpo"), ("token", "dapo"), ("token", "cispo")],
)
@pytest.mark.parametrize("masked_policy_logp", [-14.0, -114.0])
def test_grpo_loss_is_float32_for_float16_scores(importance_sampling_level, loss_type, masked_policy_logp):
    """The real GRPO loss over one multi-turn completion: two policy tokens and a
    masked tool-output token whose reference log-prob is 12 or 112 nats above
    the policy's, overflowing float16 or float32 unless excluded before exponentiation."""

    trainer = _bare(
        _Float16Scores,
        top_entropy_quantile=1.0,
        aux_loss_enabled=False,
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        off_policy_mask_threshold=None,
        importance_sampling_level=importance_sampling_level,
        beta=0.04,
        loss_type=loss_type,
        epsilon_low=0.2,
        epsilon_high=0.28 if loss_type != "cispo" else 5.0,
        _metrics={"train": defaultdict(list)},
        model=SimpleNamespace(training=True),
        current_gradient_accumulation_steps=1,
        max_completion_length=3,
    )
    trainer.args = SimpleNamespace(use_bias_correction_kl=False, delta=None, steps_per_generation=1)
    trainer.policy = torch.tensor([[-0.5, -1.0, masked_policy_logp]], dtype=torch.float16)
    inputs = {
        "prompt_ids": torch.zeros((1, 2), dtype=torch.long),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.zeros((1, 3), dtype=torch.long),
        "completion_mask": torch.ones((1, 3), dtype=torch.long),
        "tool_mask": torch.tensor([[1, 1, 0]]),
        "advantages": torch.tensor([0.7]),
        "old_per_token_logps": trainer.policy.clone(),
        "ref_per_token_logps": torch.tensor([[-0.6, -0.9, -2.0]], dtype=torch.float16),
        "num_items_in_batch": torch.tensor(2.0),
    }
    loss = trainer._compute_loss(None, inputs)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(values)).all() for values in trainer._metrics["train"].values())
