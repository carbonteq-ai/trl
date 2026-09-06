import dataclasses

import pytest

from trl import Olmo3GRPOConfig


def test_olmo3_grpo_config_resolves_published_algorithm_recipe():
    config = Olmo3GRPOConfig(output_dir="/tmp/trl-test", report_to="none")

    assert config.use_vllm is True
    assert config.dynamic_sampling is False
    assert config.active_sampling is True
    assert config.active_sampling_reward_std_epsilon == 0.0
    assert config.loss_type == "dapo"
    assert config.beta == 0.0
    assert config.epsilon == 0.2
    assert config.epsilon_high == 0.272
    assert config.scale_rewards == "none"
    assert config.importance_sampling_level == "token"
    assert config.vllm_importance_sampling_correction is True
    assert config.vllm_importance_sampling_mode == "token_truncate"
    assert config.vllm_importance_sampling_clip_min is None
    assert config.vllm_importance_sampling_clip_max == 2.0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("active_sampling", False),
        ("loss_type", "grpo"),
        ("beta", 0.01),
        ("epsilon_high", 0.2),
        ("scale_rewards", "group"),
        ("vllm_importance_sampling_mode", "sequence_mask"),
        ("vllm_importance_sampling_clip_max", 3.0),
    ],
)
def test_olmo3_grpo_config_rejects_recipe_overrides(argument, value):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{argument}'"):
        Olmo3GRPOConfig(output_dir="/tmp/trl-test", report_to="none", **{argument: value})


def test_olmo3_grpo_config_keeps_workload_settings_configurable():
    config = Olmo3GRPOConfig(
        output_dir="/tmp/trl-test",
        report_to="none",
        learning_rate=1e-6,
        num_generations=8,
        active_sampling_max_batches=4,
        max_completion_length=4096,
    )

    assert config.learning_rate == 1e-6
    assert config.num_generations == 8
    assert config.active_sampling_max_batches == 4
    assert config.max_completion_length == 4096

    init_fields = {item.name for item in dataclasses.fields(config) if item.init}
    assert "loss_type" not in init_fields
    assert "learning_rate" in init_fields
