import pytest

from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import _validate_precomputed_advantages


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
