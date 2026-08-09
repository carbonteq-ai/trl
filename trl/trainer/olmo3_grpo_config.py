# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class Olmo3GRPOConfig(GRPOConfig):
    r"""Configuration for the model-agnostic OlmoRL objective published with OLMo 3.

    The trainer remains [`GRPOTrainer`]. This configuration fixes the seven coupled algorithm choices that define
    the OLMo 3 recipe so selecting it cannot silently fall back to ordinary GRPO. Model, batch, rollout-length,
    learning-rate, and runtime-capacity arguments remain configurable.

    The fixed recipe uses zero-gradient group filtering with active refill, global token-level loss normalization,
    no KL penalty, asymmetric clipping, token-level truncated importance sampling (TIS), and mean-only group
    advantages.
    """

    use_vllm: bool = field(
        default=True,
        init=False,
        metadata={"help": "OlmoRL uses vLLM rollout probabilities for truncated importance sampling."},
    )
    beta: float = field(
        default=0.0,
        init=False,
        metadata={"help": "OlmoRL omits the KL penalty and reference model."},
    )
    epsilon: float = field(
        default=0.2,
        init=False,
        metadata={"help": "OlmoRL lower PPO-style clipping bound."},
    )
    epsilon_high: float = field(
        default=0.272,
        init=False,
        metadata={"help": "OlmoRL clip-higher upper PPO-style clipping bound."},
    )
    importance_sampling_level: str = field(
        default="token",
        init=False,
        metadata={"help": "OlmoRL applies policy ratios at token granularity."},
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        init=False,
        metadata={"help": "OlmoRL forms one scalar verifier reward before group-relative centering."},
    )
    scale_rewards: str = field(
        default="none",
        init=False,
        metadata={"help": "OlmoRL centers rewards without dividing by within-group standard deviation."},
    )
    loss_type: str = field(
        default="dapo",
        init=False,
        metadata={"help": "OlmoRL normalizes policy loss by active tokens across the global batch."},
    )
    dynamic_sampling: bool = field(
        default=False,
        init=False,
        metadata={"help": "OlmoRL uses active sampling rather than DAPO's whole-batch dynamic oversampling."},
    )
    active_sampling: bool = field(
        default=True,
        init=False,
        metadata={"help": "OlmoRL filters zero-gradient prompt groups and actively refills the batch."},
    )
    active_sampling_reward_std_epsilon: float = field(
        default=0.0,
        init=False,
        metadata={"help": "OlmoRL filters groups with exactly zero reward spread."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        init=False,
        metadata={"help": "OlmoRL corrects the learner/sampler probability mismatch."},
    )
    vllm_importance_sampling_mode: str = field(
        default="token_truncate",
        init=False,
        metadata={"help": "OlmoRL truncates the learner/vLLM ratio independently for each completion token."},
    )
    vllm_importance_sampling_clip_max: float | None = field(
        default=2.0,
        init=False,
        metadata={"help": "Published OlmoRL upper cap for truncated importance sampling."},
    )
    vllm_importance_sampling_clip_min: float | None = field(
        default=None,
        init=False,
        metadata={"help": "OlmoRL applies only the published upper TIS cap."},
    )
