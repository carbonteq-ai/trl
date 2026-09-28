from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.rollout_admission import NoAdmittedRollouts


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


def _active_sampling_trainer(max_batches, num_generations=2, oversample=0, oversample_refill=0):
    trainer = object.__new__(GRPOTrainer)
    trainer.active_sampling_max_batches = max_batches
    trainer.active_sampling_reward_std_epsilon = 0.0
    trainer.active_sampling_oversample = oversample
    trainer.active_sampling_oversample_refill = oversample_refill
    trainer.num_generations = num_generations
    trainer.accelerator = _SingleProcessAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._metrics = {"train": defaultdict(list)}
    return trainer


def test_active_sampling_refills_only_missing_rows():
    trainer = _active_sampling_trainer(max_batches=2)
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
    assert trainer._metrics["train"]["active_sampling/candidate_groups_reserved"] == [8]
    assert trainer._metrics["train"]["active_sampling/candidate_groups_generated"] == [6]
    assert trainer._metrics["train"]["active_sampling/candidate_groups_retained"] == [4]
    assert trainer._metrics["train"]["active_sampling/candidate_groups_unused"] == [2]


def _group_candidates(num_groups, num_generations=2):
    return [{"prompt": str(group), "group": group} for group in range(num_groups) for _ in range(num_generations)]


def _fake_group_rollouts(trainer, zero_spread_groups, no_admission_rounds=()):
    """Score candidates without a model: a group keeps its reward spread unless listed in `zero_spread_groups`."""
    generated_groups = []

    def generate(candidate_batch):
        groups = [row["group"] for row in candidate_batch[:: trainer.num_generations]]
        generated_groups.append(groups)
        if len(generated_groups) in no_admission_rounds:
            raise NoAdmittedRollouts("rollout admission retained no complete groups")
        size = len(candidate_batch)
        return {
            "prompt_ids": torch.tensor([[row["group"]] for row in candidate_batch]),
            "prompt_mask": torch.ones((size, 1), dtype=torch.long),
            "completion_ids": torch.tensor([[row["group"], 0] for row in candidate_batch]),
            "completion_mask": torch.ones((size, 2), dtype=torch.long),
            "advantages": torch.zeros(size),
            "num_items_in_batch": torch.tensor(size * 2),
            "group_reward_std": torch.tensor(
                [0.0 if row["group"] in zero_spread_groups else 1.0 for row in candidate_batch]
            ),
        }

    trainer._generate_and_score_completions = generate
    return generated_groups


def _trained_groups(batch, num_generations=2):
    return batch["completion_ids"][::num_generations, 0].tolist()


@pytest.mark.parametrize("zero_spread_groups", [set(), {1}, {0, 1, 2, 3, 5}])
def test_active_sampling_without_oversampling_is_unchanged(zero_spread_groups):
    # Four target groups from a three-batch pool; the exact-refill request sequence is fixed by the retained groups.
    trainer = _active_sampling_trainer(max_batches=3)
    generated = _fake_group_rollouts(trainer, zero_spread_groups)

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    expected = {
        frozenset(): ([[0, 1, 2, 3]], [0, 1, 2, 3]),
        frozenset({1}): ([[0, 1, 2, 3], [4]], [0, 2, 3, 4]),
        frozenset({0, 1, 2, 3, 5}): ([[0, 1, 2, 3], [4, 5, 6, 7], [8]], [4, 6, 7, 8]),
    }[frozenset(zero_spread_groups)]
    assert generated == expected[0]
    assert _trained_groups(batch) == expected[1]
    assert sorted(trainer._metrics["train"]) == [
        "active_sampling/candidate_groups_generated",
        "active_sampling/candidate_groups_reserved",
        "active_sampling/candidate_groups_retained",
        "active_sampling/candidate_groups_unused",
        "active_sampling/generated_rows",
        "active_sampling/generation_rounds",
        "active_sampling/retained_fraction",
    ]


def test_active_sampling_oversample_fills_the_target_in_one_round():
    trainer = _active_sampling_trainer(max_batches=3, oversample=2)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={1})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    assert generated == [[0, 1, 2, 3, 4, 5]]
    # The first four retained groups in candidate order are trained; group 5 had spread and is discarded.
    assert _trained_groups(batch) == [0, 2, 3, 4]
    assert batch["num_items_in_batch"].item() == 16
    metrics = trainer._metrics["train"]
    assert metrics["active_sampling/generation_rounds"] == [1]
    assert metrics["active_sampling/generated_rows"] == [12]
    assert metrics["active_sampling/candidate_groups_retained"] == [10]
    assert metrics["active_sampling/candidate_groups_unused"] == [12]
    assert metrics["active_sampling/oversampled_groups"] == [2]
    assert metrics["active_sampling/discarded_groups"] == [1]
    assert metrics["active_sampling/round_1_requested_groups"] == [6]
    assert metrics["active_sampling/round_1_generated_groups"] == [6]
    assert metrics["active_sampling/round_1_retained_groups"] == [5]


def test_active_sampling_refill_rounds_request_missing_plus_refill_oversampling():
    trainer = _active_sampling_trainer(max_batches=3, oversample=1, oversample_refill=2)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={0, 1, 2, 5, 6})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    # Round 1: 4 target + 1 extra; groups 3 and 4 retained. Round 2: 2 missing + 2 extra; groups 7 and 8 fill it.
    assert generated == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    assert _trained_groups(batch) == [3, 4, 7, 8]
    metrics = trainer._metrics["train"]
    assert metrics["active_sampling/generation_rounds"] == [2]
    assert metrics["active_sampling/oversampled_groups"] == [3]
    assert metrics["active_sampling/discarded_groups"] == [0]
    assert metrics["active_sampling/round_2_requested_groups"] == [4]
    assert metrics["active_sampling/round_2_generated_groups"] == [4]
    assert metrics["active_sampling/round_2_retained_groups"] == [2]


def test_active_sampling_refill_oversampling_applies_without_first_round_oversampling():
    trainer = _active_sampling_trainer(max_batches=3, oversample_refill=1)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={1})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    assert generated == [[0, 1, 2, 3], [4, 5]]
    assert _trained_groups(batch) == [0, 2, 3, 4]
    assert trainer._metrics["train"]["active_sampling/oversampled_groups"] == [1]
    assert trainer._metrics["train"]["active_sampling/discarded_groups"] == [1]


def test_active_sampling_refill_never_exceeds_the_first_round():
    trainer = _active_sampling_trainer(max_batches=3, oversample=1, oversample_refill=5)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={0, 1, 2, 3, 4})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    # Round 1 retains nothing, so 4 groups are missing; the refill asks for 4 + 5 but is capped at 4 + 1.
    assert generated == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert _trained_groups(batch) == [5, 6, 7, 8]
    metrics = trainer._metrics["train"]
    assert metrics["active_sampling/round_2_requested_groups"] == [5]
    assert metrics["active_sampling/round_2_generated_groups"] == [5]
    assert metrics["active_sampling/oversampled_groups"] == [2]
    assert metrics["active_sampling/discarded_groups"] == [1]


def test_active_sampling_oversampling_is_cut_to_the_remaining_pool():
    # Two target groups in a four-group pool: the first round asks for 2 + 5 groups and receives the whole pool.
    trainer = _active_sampling_trainer(max_batches=2, oversample=5)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={0})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(4))

    assert generated == [[0, 1, 2, 3]]
    assert _trained_groups(batch) == [1, 2]
    metrics = trainer._metrics["train"]
    assert metrics["active_sampling/round_1_requested_groups"] == [7]
    assert metrics["active_sampling/round_1_generated_groups"] == [4]
    assert metrics["active_sampling/oversampled_groups"] == [2]
    assert metrics["active_sampling/discarded_groups"] == [1]
    assert metrics["active_sampling/candidate_groups_unused"] == [0]


def test_active_sampling_refill_takes_what_is_left_of_the_pool():
    trainer = _active_sampling_trainer(max_batches=3, oversample=3, oversample_refill=4)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={0, 1, 2, 3})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(6))

    # Round 1 takes 5 of 6 groups and retains only group 4; the refill asks for 1 + 4 groups but only group 5 is left.
    assert generated == [[0, 1, 2, 3, 4], [5]]
    assert _trained_groups(batch) == [4, 5]
    assert trainer._metrics["train"]["active_sampling/round_2_requested_groups"] == [5]
    assert trainer._metrics["train"]["active_sampling/round_2_generated_groups"] == [1]


def test_active_sampling_oversampling_fails_when_the_pool_cannot_cover_missing_groups():
    trainer = _active_sampling_trainer(max_batches=3, oversample=3)
    generated = _fake_group_rollouts(trainer, zero_spread_groups={0, 1, 2, 3, 5})

    with pytest.raises(
        RuntimeError, match="exhausted its bounded candidate pool: 2 rows are missing but only 0 of 12"
    ):
        trainer._prepare_active_sampling_inputs(_group_candidates(6))

    assert generated == [[0, 1, 2, 3, 4], [5]]


def test_active_sampling_oversampling_survives_a_round_without_admitted_rollouts():
    trainer = _active_sampling_trainer(max_batches=3, oversample=1, oversample_refill=1)
    generated = _fake_group_rollouts(trainer, zero_spread_groups=set(), no_admission_rounds={1})

    batch = trainer._prepare_active_sampling_inputs(_group_candidates(12))

    assert generated == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert _trained_groups(batch) == [5, 6, 7, 8]
    metrics = trainer._metrics["train"]
    assert metrics["active_sampling/round_1_retained_groups"] == [0]
    assert metrics["active_sampling/oversampled_groups"] == [2]
    assert metrics["active_sampling/discarded_groups"] == [1]


@pytest.mark.parametrize("argument", ["active_sampling_oversample", "active_sampling_oversample_refill"])
def test_active_sampling_oversampling_requires_active_sampling(argument):
    with pytest.raises(ValueError, match="require active_sampling=True"):
        GRPOConfig(output_dir="/tmp/trl-test", loss_type="dapo", report_to="none", **{argument: 1})


@pytest.mark.parametrize("argument", ["active_sampling_oversample", "active_sampling_oversample_refill"])
def test_active_sampling_oversampling_rejects_negative_counts(argument):
    with pytest.raises(ValueError, match="must be non-negative"):
        GRPOConfig(
            output_dir="/tmp/trl-test", loss_type="dapo", active_sampling=True, report_to="none", **{argument: -1}
        )


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
