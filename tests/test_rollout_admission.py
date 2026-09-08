"""Admission preserves source identity and the retained-sample gradient."""

from types import SimpleNamespace

import pytest
import torch

from trl import GRPOTrainer
from trl.trainer.rollout_admission import NoAdmittedRollouts, pad_admitted_rollout_batch, retained_rollout_indices
from trl.trainer.utils import split_tensor_dict


@pytest.mark.parametrize("indices", [[0, 1, 4, 5], [2, 3]])
def test_retains_complete_nonprefix_groups(indices):
    assert retained_rollout_indices(indices, 6, len(indices), 2) == indices


@pytest.mark.parametrize("indices", [[0], [0, 2], [1, 0], [0, 0], [-2, -1], [6, 7], [False, 1], []])
def test_rejects_invalid_group_identity(indices):
    with pytest.raises((ValueError, NoAdmittedRollouts)):
        retained_rollout_indices(indices, 6, len(indices), 2)


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_masked_accumulation_matches_retained_sequence_gradient(microbatch):
    parameter = torch.tensor(0.1, requires_grad=True)
    coefficients = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    batch = {
        "completion_ids": coefficients,
        "completion_mask": mask,
        "advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
        "num_items_in_batch": mask.sum(),
    }
    padded = pad_admitted_rollout_batch(batch, 8)
    assert padded["completion_mask"][4:].sum() == 0
    assert padded["num_items_in_batch"] == mask.sum()
    assert batch["completion_mask"].sum() == mask.sum()
    steps = 8 // microbatch

    def loss_fn(model, inputs):
        # Nonlinear per-token objective catches missing scale, mask and row alignment.
        values = (parameter * inputs["completion_ids"]).exp() * inputs["advantages"][:, None]
        active = inputs["completion_mask"]
        return ((values * active).sum(-1) / active.sum(-1).clamp(min=1)).mean() / steps

    trainer = SimpleNamespace(use_liger_kernel=False, _compute_loss=loss_fn)
    actual = sum(GRPOTrainer.compute_loss(trainer, None, part) for part in split_tensor_dict(padded, steps))
    expected = (((parameter * coefficients).exp() * batch["advantages"][:, None] * mask).sum(-1) / mask.sum(-1)).mean()
    assert actual.item() == pytest.approx(expected.item())
    assert torch.autograd.grad(actual, parameter, retain_graph=True)[0].item() == pytest.approx(
        torch.autograd.grad(expected, parameter)[0].item()
    )


@pytest.mark.parametrize("active_sampling,empty_round", [(False, False), (True, False), (True, True)])
def test_real_trainer_updates_after_dropping_nonprefix_groups(tmp_path, active_sampling, empty_round):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from trl import GRPOConfig

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<unk>": 1, "prompt": 2, "answer": 3, "<eos>": 4}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )
    model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_positions=16, n_embd=8, n_layer=1, n_head=1))
    before = model.transformer.wte.weight.detach().clone()
    calls = []

    def rollout(prompts, trainer, inputs):
        retained = [0, 1, 4, 5] if len(inputs) == 8 else list(range(len(inputs)))
        if empty_round:
            retained = [] if len(calls) % 2 == 0 else list(range(len(inputs)))
        calls.append([inputs[index]["example_id"] for index in retained])
        return {
            "prompt_ids": [[2]] * len(retained),
            "completion_ids": [[3, 4], [4]] * (len(retained) // 2),
            "logprobs": None,
            "retained_input_indices": retained,
            "expected_identity": calls[-1],
        }

    def reward(completions, example_id, expected_identity, **kwargs):
        assert example_id == expected_identity == calls[-1]
        assert len(completions) == len(expected_identity)
        return [1.0, 0.0] * (len(completions) // 2)

    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=2,
            max_steps=2,
            max_completion_length=2,
            beta=0.0,
            loss_type="dapo" if active_sampling else "grpo",
            active_sampling=active_sampling,
            active_sampling_max_batches=2,
            scale_rewards="none" if active_sampling else "group",
            report_to="none",
            save_strategy="no",
            gradient_checkpointing=False,
        ),
        train_dataset=Dataset.from_dict({"prompt": ["prompt"] * 8, "example_id": list(range(8))}),
        processing_class=tokenizer,
        reward_funcs=reward,
        rollout_func=rollout,
    )
    trainer.train()
    assert trainer.state.global_step == 2
    assert len(calls) == (4 if active_sampling else 2)
    assert not torch.equal(before, model.transformer.wte.weight)
