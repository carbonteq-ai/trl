from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl import GRPOConfig, GRPOTrainer


class _SingleProcessAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def gather(value):
        return value


def _scored_batch(group_stds):
    size = len(group_stds)
    return {
        "prompt_ids": torch.arange(size).unsqueeze(1),
        "prompt_mask": torch.ones((size, 1), dtype=torch.long),
        "completion_ids": torch.arange(size * 2).reshape(size, 2),
        "completion_mask": torch.ones((size, 2), dtype=torch.long),
        "advantages": torch.arange(size, dtype=torch.float),
        "num_items_in_batch": torch.tensor(size * 2),
        "group_reward_std": torch.tensor(group_stds, dtype=torch.float),
    }


def test_dynamic_sampling_requires_dapo():
    with pytest.raises(ValueError, match="requires loss_type='dapo'"):
        GRPOConfig(output_dir="/tmp/trl-test", loss_type="grpo", dynamic_sampling=True, report_to="none")


def test_dynamic_sampling_requires_a_bounded_positive_candidate_count():
    with pytest.raises(ValueError, match="positive integer"):
        GRPOConfig(
            output_dir="/tmp/trl-test",
            loss_type="dapo",
            dynamic_sampling=True,
            dynamic_sampling_max_batches=0,
            report_to="none",
        )


def test_active_sampling_and_dynamic_sampling_are_mutually_exclusive():
    with pytest.raises(ValueError, match="separate refill strategies"):
        GRPOConfig(
            output_dir="/tmp/trl-test",
            loss_type="dapo",
            dynamic_sampling=True,
            active_sampling=True,
            report_to="none",
        )


def test_active_sampling_refills_only_missing_rows():
    trainer = object.__new__(GRPOTrainer)
    trainer.active_sampling_max_batches = 2
    trainer.active_sampling_reward_std_epsilon = 0.0
    trainer.num_generations = 2
    trainer.accelerator = _SingleProcessAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._metrics = {"train": defaultdict(list)}
    scored_batches = iter(
        [
            _scored_batch([1.0, 1.0, 0.0, 0.0]),
            _scored_batch([2.0, 2.0]),
        ]
    )
    generated_sizes = []

    def generate(candidate_batch):
        generated_sizes.append(len(candidate_batch))
        return next(scored_batches)

    trainer._generate_and_score_completions = generate
    candidate_inputs = [{"prompt": str(index)} for index in range(8)]

    batch = trainer._prepare_active_sampling_inputs(candidate_inputs)

    assert generated_sizes == [4, 2]
    assert batch["completion_ids"].shape == (4, 2)
    assert trainer._metrics["train"]["active_sampling/generation_rounds"] == [2]
    assert trainer._metrics["train"]["active_sampling/generated_rows"] == [6]


def test_dynamic_sampling_retains_valid_groups_and_refills_only_missing_rows():
    trainer = object.__new__(GRPOTrainer)
    trainer.dynamic_sampling_max_batches = 2
    trainer.dynamic_sampling_reward_std_epsilon = 0.0
    trainer.accelerator = _SingleProcessAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._metrics = {"train": defaultdict(list)}
    scored_batches = iter(
        [
            _scored_batch([1.0, 1.0, 0.0, 0.0]),
            _scored_batch([2.0, 2.0, 0.0, 0.0]),
        ]
    )
    generated = []

    def generate(candidate_batch):
        generated.append(candidate_batch)
        return next(scored_batches)

    trainer._generate_and_score_completions = generate
    candidate_inputs = [{"prompt": str(index)} for index in range(8)]

    batch = trainer._prepare_dynamic_sampling_inputs(candidate_inputs)

    assert len(generated) == 2
    assert batch["completion_ids"].shape == (4, 2)
    assert batch["completion_ids"][:, 0].tolist() == [0, 2, 0, 2]
    assert batch["num_items_in_batch"].item() == 8
    assert trainer._metrics["train"]["dynamic_sampling/candidate_batches"] == [2]
    assert trainer._metrics["train"]["dynamic_sampling/retained_fraction"] == [0.5]


def test_dynamic_sampling_fails_instead_of_training_a_partial_batch():
    trainer = object.__new__(GRPOTrainer)
    trainer.dynamic_sampling_max_batches = 2
    trainer.dynamic_sampling_reward_std_epsilon = 0.0
    trainer.accelerator = _SingleProcessAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._metrics = {"train": defaultdict(list)}
    trainer._generate_and_score_completions = lambda _: _scored_batch([0.0, 0.0])

    with pytest.raises(RuntimeError, match="before every process filled"):
        trainer._prepare_dynamic_sampling_inputs([{"prompt": str(index)} for index in range(4)])
