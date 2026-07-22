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

"""vLLM-based generation backend for TRL trainers."""

import logging
import math
import os
import tempfile
from collections.abc import MutableMapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, is_bitsandbytes_available
from transformers.utils import (
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..distributed import DistributedBackend
from ..extras.profiling import ProfilingContext
from ..import_utils import is_vllm_available
from ..trainer.utils import ensure_master_addr_port
from .vllm_client import VLLMClient


if is_vllm_available():
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams


logger = logging.getLogger(__name__)

_SPEC_DECODE_COUNTERS = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
}

_KV_CACHE_CAPACITY_METRIC = "rollout/kv_cache_capacity_tokens"
_KV_CACHE_PEAK_USAGE_METRIC = "rollout/kv_cache_peak_usage_ratio"


class _KvCachePeakTracker:
    """Retain the exact peak scheduler-reported KV-cache usage for one generation call."""

    def __init__(self) -> None:
        self.peak_usage_ratio = 0.0

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0) -> None:
        del iteration_stats, mm_cache_stats, engine_idx
        if scheduler_stats is not None:
            self.peak_usage_ratio = max(self.peak_usage_ratio, float(scheduler_stats.kv_cache_usage))

    def reset(self) -> None:
        self.peak_usage_ratio = 0.0

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass

    def record_sleep_state(self, sleep=0, level=0) -> None:
        del sleep, level


def _compute_spec_decode_counter_delta(
    current: dict[str, float], previous: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert vLLM process-lifetime counters into metrics for one generation call."""
    if not current:
        return {}, previous
    delta = {
        name: value - previous.get(name, 0.0) if value >= previous.get(name, 0.0) else value
        for name, value in current.items()
    }
    drafts = delta.get("drafts", 0.0)
    draft_tokens = delta.get("draft_tokens", 0.0)
    accepted_tokens = delta.get("accepted_tokens", 0.0)
    return (
        {
            "rollout/spec_num_drafts": drafts,
            "rollout/spec_num_draft_tokens": draft_tokens,
            "rollout/spec_num_accepted_tokens": accepted_tokens,
            "rollout/spec_accept_rate": accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0,
            "rollout/spec_accept_length": 1.0 + accepted_tokens / drafts if drafts > 0 else 0.0,
        },
        current,
    )


def _accumulate_spec_decode_metrics(
    buffer: MutableMapping[str, list[float]], metrics: dict[str, float]
) -> None:
    """Accumulate turn-local counters into one step total with weighted rates."""
    if not metrics:
        return
    counter_names = (
        "rollout/spec_num_drafts",
        "rollout/spec_num_draft_tokens",
        "rollout/spec_num_accepted_tokens",
    )
    if any(name in metrics for name in counter_names):
        totals = {
            name: (buffer.get(name, [0.0])[-1] if buffer.get(name) else 0.0) + metrics.get(name, 0.0)
            for name in counter_names
        }
        for name, value in totals.items():
            buffer[name] = [value]
        drafts = totals["rollout/spec_num_drafts"]
        draft_tokens = totals["rollout/spec_num_draft_tokens"]
        accepted_tokens = totals["rollout/spec_num_accepted_tokens"]
        buffer["rollout/spec_accept_rate"] = [accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0]
        buffer["rollout/spec_accept_length"] = [1.0 + accepted_tokens / drafts if drafts > 0 else 0.0]

    if _KV_CACHE_CAPACITY_METRIC in metrics:
        buffer[_KV_CACHE_CAPACITY_METRIC] = [metrics[_KV_CACHE_CAPACITY_METRIC]]
    if _KV_CACHE_PEAK_USAGE_METRIC in metrics:
        prior_peak = buffer.get(_KV_CACHE_PEAK_USAGE_METRIC, [0.0])[-1]
        buffer[_KV_CACHE_PEAK_USAGE_METRIC] = [max(prior_peak, metrics[_KV_CACHE_PEAK_USAGE_METRIC])]


def _apply_turboquant_compatibility_patch() -> tuple[str, ...]:
    """Preserve TurboQuant's quantized cache marker in affected vLLM builds."""
    from vllm.v1.kv_cache_interface import KVQuantMode, TQFullAttentionSpec, get_kv_quant_mode

    if get_kv_quant_mode("turboquant_k8v4") != KVQuantMode.NONE:
        return ()
    if getattr(TQFullAttentionSpec, "_trl_quant_marker_patch", False):
        return ("turboquant-quant-marker",)

    inherited_post_init: Any = TQFullAttentionSpec.__post_init__

    def tq_post_init(self: Any) -> None:
        inherited_post_init(self)
        if self.kv_quant_mode == KVQuantMode.NONE:
            object.__setattr__(self, "kv_quant_mode", KVQuantMode.FP8_PER_TENSOR)

    spec_class: Any = TQFullAttentionSpec
    spec_class.__post_init__ = tq_post_init
    spec_class._trl_quant_marker_patch = True
    return ("turboquant-quant-marker",)


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (CUDA, MLU, MPS, NPU, XPU) and empties the
    cache of the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def extract_logprobs(all_outputs: list["RequestOutput"]):
    """
    Extract logprobs and token IDs from vLLM generation outputs.

    Returns logprobs and token IDs sorted by rank (most probable first). Each returned list has shape (num_sequences,
    seq_len, num_logprobs), where num_logprobs is determined by the `logprobs` parameter passed to vLLM (1 when
    `logprobs=0`, up to N+1 when `logprobs=N`). NaN logprob values are replaced with `None`.

    Args:
        all_outputs (list of `RequestOutput`):
            List of vLLM `RequestOutput` objects from generation.

    Returns:
        Tuple of (logprobs, logprob_token_ids), each of shape (num_sequences, seq_len, num_logprobs).
    """
    all_logprobs = []
    all_token_ids = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            if output.logprobs is None:
                return None, None
            seq_logprobs = []
            seq_token_ids = []
            for lp in output.logprobs:
                sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                seq_token_ids.append([token_id for token_id, _ in sorted_items])
                seq_logprobs.append([None if math.isnan(item.logprob) else item.logprob for _, item in sorted_items])
            all_logprobs.append(seq_logprobs)
            all_token_ids.append(seq_token_ids)
    return all_logprobs, all_token_ids


if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel


if is_bitsandbytes_available():
    import bitsandbytes as bnb


class VLLMGeneration:
    """Handles vLLM-based generation for trainers.

    Extracts all vLLM-specific logic (initialization, generation, weight sync) from trainers into a separate, testable
    class.

    Args:
        model ([`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to use for generation.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator for distributed training.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
            Tokenizer or processor for the model.

        > Parameters for vLLM:

        mode (`str`, *optional*, defaults to `"colocate"`):
            vLLM mode. Must be one of `"colocate"` or `"server"`.

            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).

        structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.

        > Parameters for "server" vLLM mode:

        server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `server_host` and
            `server_port` are ignored.
        server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.

        > Parameters for "colocate" vLLM mode:

        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            The number of GPUs to use for distributed execution with tensor parallelism. This setting only applies when
            `mode` is set to `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately
            when launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's throughput. However, if the value is
            too high, it may cause out-of- memory (OOM) errors. This setting only applies when `mode` is set to
            `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately when launching the
            vLLM server via the `--vllm_gpu_memory_utilization` flag.
        max_model_length (`int`, *optional*):
            Model context length (prompt and completion). Set it to at least the maximum prompt length in the dataset
            plus `max_completion_length`; if omitted, it is inferred from the model config.
        max_num_seqs (`int`, *optional*):
            Maximum number of sequences to process in parallel, effectively capping the batch size.
        enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for the engine to offload weights/cache during the optimizer step. Keeps GPU
            memory usage low, but waking the engine adds host–device transfer latency.
        speculative_config (`dict`, *optional*):
            Engine-level speculative decoding configuration for colocated vLLM. This value is forwarded to `LLM` and
            is ignored in server mode, whose engine must be configured when the server is launched.
        engine_kwargs (`dict`, *optional*):
            Additional non-conflicting `LLM` engine arguments for colocated mode. Arguments controlled directly by
            this adapter cannot be overridden. This value is ignored in server mode.
        weight_name_prefix (`str`, *optional*):
            Prefix added to parameter names before weight synchronization. Use this when the vLLM model keeps a
            composite-model namespace around the text model while the training model exposes the text model directly.
        weight_sync_mode (`str`, *optional*, defaults to `"full"`):
            How colocated vLLM receives current policy weights. `"full"` merges PEFT adapters and synchronizes model
            parameters. `"lora"` keeps vLLM's base weights unchanged and reloads the active PEFT LoRA adapter through
            vLLM's native dynamic-LoRA path. The latter avoids copying packed 4-bit parameter storage into vLLM and
            is supported only for PEFT LoRA models in colocated mode.
        model_impl (`str`, *optional*, defaults to `"auto"`):
            Model implementation to use for vLLM.
            - "auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers
              implementation if no vLLM implementation is available.
            - "vllm" will use the vLLM model implementation.
            - "transformers" will use the Transformers model implementation.
            - "terratorch" will use the TerraTorch model implementation.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.

        > Parameters for generation:

        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Parameter for repetition penalty. It penalizes new tokens based on whether they appear in the prompt and
            the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the
            model to repeat tokens. Default `1.0` means no penalty.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature. It controls the randomness of the sampling. Lower values make the model more
            deterministic, while higher values make the model more random and increase diversity.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p sampling parameter. It controls the cumulative probability of the top tokens to consider. Defaults to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter. It controls the number of top tokens to consider. Defaults to `0` to consider all
            tokens.
        min_p (`float`, *optional*, defaults to `0.0`):
            Min-p sampling parameter. It represents the minimum probability for a token to be considered, relative to
            the probability of the most likely token. Default `0.0` means min-p is disabled.
        max_completion_length (`int`, *optional*, defaults to `16`):
            Maximum number of tokens to generate for each prompt.
        logprobs (`int` or `None`, *optional*, defaults to `0`):
            Number of top logprobs to return per token. When 0 (default), only the sampled token's logprob is returned
            (inner dimension = 1). When N>0, returns up to N+1 logprobs sorted by descending probability, because vLLM
            always includes the sampled token's logprob alongside the top-N (the sampled token may or may not already
            be in the top-N).
        generation_kwargs (`dict`, *optional*):
            Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
            `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they will
            override them.

        > Parameters for chat/tools:

        chat_template (`str`, *optional*):
            Template to use for structuring the chat. If not provided, the model's default chat template will be used.
        chat_template_kwargs (`dict`, *optional*):
            Additional keyword arguments to customize the chat template used by the model.
        tools (`list`, *optional*):
            Tools available for tool calling during chat generation.
    """

    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # vLLM configuration
        mode: str = "colocate",
        structured_outputs_regex: str | None = None,
        # Server mode configuration
        server_base_url: str | None = None,
        server_host: str = "0.0.0.0",
        server_port: int = 8000,
        server_timeout: float = 240.0,
        group_port: int = 51216,
        # Colocate mode configuration
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        max_num_seqs: int | None = None,
        enable_sleep_mode: bool = False,
        speculative_config: dict | None = None,
        engine_kwargs: dict | None = None,
        weight_name_prefix: str | None = None,
        weight_sync_mode: str = "full",
        model_impl: str = "auto",
        trust_remote_code: bool = False,
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
    ):
        self.model = model
        self.accelerator = accelerator
        self._dist = DistributedBackend(accelerator)
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.structured_outputs_regex = structured_outputs_regex

        # Server mode configuration
        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.group_port = group_port
        self.server_timeout = server_timeout

        # Colocate mode configuration
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.max_num_seqs = max_num_seqs
        self.enable_sleep_mode = enable_sleep_mode
        self.speculative_config = speculative_config
        self.engine_kwargs = engine_kwargs or {}
        if weight_name_prefix is not None and (not weight_name_prefix or not weight_name_prefix.endswith(".")):
            raise ValueError("weight_name_prefix must be a non-empty module prefix ending with `.`")
        self.weight_name_prefix = weight_name_prefix
        if weight_sync_mode not in {"full", "lora"}:
            raise ValueError("weight_sync_mode must be either `full` or `lora`")
        if weight_sync_mode == "lora":
            if mode != "colocate":
                raise ValueError("LoRA weight synchronization is supported only in colocated vLLM mode")
            if not is_peft_model(model):
                raise ValueError("LoRA weight synchronization requires a PEFT model")
            if weight_name_prefix is not None:
                raise ValueError("weight_name_prefix does not apply to LoRA weight synchronization")
        self.weight_sync_mode = weight_sync_mode
        self._lora_directory = None
        self._lora_request = None
        self.model_impl = model_impl
        self.trust_remote_code = trust_remote_code

        # Generation configuration
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.logprobs = logprobs
        self.generation_kwargs = generation_kwargs or {}
        self.last_generation_metrics: dict[str, float] = {}
        self._spec_decode_counter_snapshot: dict[str, float] = {}
        self._kv_cache_capacity_tokens: float | None = None
        self._kv_cache_peak_tracker: _KvCachePeakTracker | None = None

        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM in server or colocate mode."""
        model = self.model
        accelerator = self.accelerator

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install trl[vllm]` to use it."
            )

        if self.mode == "server":
            if accelerator.is_main_process:
                if self.server_base_url is not None:
                    base_url = self.server_base_url
                else:
                    base_url = f"http://{self.server_host}:{self.server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=self.group_port, connection_timeout=self.server_timeout
                )
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

        elif self.mode == "colocate":
            # Make sure tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % self.tensor_parallel_size == 0:
                raise ValueError(
                    f"tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if self.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `tensor_parallel_size` ranks.
                # For example, if world_size=8 and tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
                        for i in range(accelerator.num_processes // self.tensor_parallel_size)
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
            # Ensure distributed rendezvous variables are set without colliding across concurrent runs
            ensure_master_addr_port()

            quantization = None
            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        quantization = "bitsandbytes"
                        break
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

            # Build LLM initialization kwargs
            llm_kwargs = {
                "model": model.name_or_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_length,
                "max_num_seqs": self.max_num_seqs,
                "enable_sleep_mode": self.enable_sleep_mode,
                "speculative_config": self.speculative_config,
                "model_impl": self.model_impl,
                "distributed_executor_backend": "external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                "seed": accelerator.process_index // self.tensor_parallel_size,
                # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                "max_num_batched_tokens": 4096,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                "logprobs_mode": "processed_logprobs",
                "quantization": quantization,
                "trust_remote_code": self.trust_remote_code,
            }
            observe_runtime_metrics = self.speculative_config is not None or str(
                self.engine_kwargs.get("kv_cache_dtype", "")
            ).startswith("turboquant_")
            if observe_runtime_metrics:
                requested_log_stats = self.engine_kwargs.get("disable_log_stats")
                if requested_log_stats is True:
                    raise ValueError("runtime metric collection requires vLLM disable_log_stats=False")
                if requested_log_stats is None:
                    llm_kwargs["disable_log_stats"] = False
            if self.weight_sync_mode == "lora":
                max_rank = max(config.r for config in model.peft_config.values())
                supported_ranks = (1, 8, 16, 32, 64, 128, 256, 320, 512)
                try:
                    max_lora_rank = next(rank for rank in supported_ranks if rank >= max_rank)
                except StopIteration as error:
                    raise ValueError(f"vLLM does not support LoRA rank {max_rank}") from error
                llm_kwargs.update({"enable_lora": True, "max_lora_rank": max_lora_rank})
            conflicts = sorted(set(llm_kwargs).intersection(self.engine_kwargs))
            if conflicts:
                raise ValueError(f"vLLM engine kwargs cannot override TRL-controlled arguments: {', '.join(conflicts)}")
            if str(self.engine_kwargs.get("kv_cache_dtype", "")).startswith("turboquant_"):
                _apply_turboquant_compatibility_patch()
            llm_kwargs.update(self.engine_kwargs)
            self.llm = LLM(**llm_kwargs)
            if observe_runtime_metrics:
                cache_config = self.llm.llm_engine.vllm_config.cache_config
                capacity = getattr(cache_config, "kv_cache_size_tokens", None)
                if isinstance(capacity, int) and capacity > 0:
                    self._kv_cache_capacity_tokens = float(capacity)
                logger_manager = self.llm.llm_engine.logger_manager
                if logger_manager is not None:
                    self._kv_cache_peak_tracker = _KvCachePeakTracker()
                    logger_manager.stat_loggers.append(self._kv_cache_peak_tracker)
            if self.weight_sync_mode == "lora":
                self._lora_directory = tempfile.TemporaryDirectory(prefix="trl-vllm-lora-")
                self._lora_request = LoRARequest(
                    "trl-training-policy",
                    1,
                    self._lora_directory.name,
                    load_inplace=True,
                )
            self._sleep_colocated_engine()
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.mode}'.")

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        accelerator.wait_for_everyone()

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Fix parameter name for vLLM compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _push_param_to_vllm(self, name: str, param) -> None:
        """Push a single parameter tensor to the vLLM engine (server or colocate mode)."""
        if self.weight_name_prefix is not None and not name.startswith(self.weight_name_prefix):
            name = f"{self.weight_name_prefix}{name}"
        if self.mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.update_named_param(name, param)
        elif self.mode == "colocate":
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    self._push_param_to_vllm(full_name, param.data)

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        """FSDP2-specific parameter synchronization."""
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            self._push_param_to_vllm(name, param)

    def _sync_fsdp_params_to_vllm(self, model: nn.Module):
        """Dispatch FSDP weight sync to the version-appropriate method."""
        if self._dist.fsdp_version == 1:
            self._sync_fsdp1_params_to_vllm(model)
        elif self._dist.fsdp_version == 2:
            self._sync_fsdp2_params_to_vllm(model)

    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
        if self.weight_sync_mode == "lora":
            # The base weights are immutable in this mode. Persist only the active adapter in PEFT's native format;
            # LoRARequest(load_inplace=True) reloads the same adapter ID on the next generation call.
            self.model.save_pretrained(self._lora_directory.name, safe_serialization=True)
            return

        # Wake up vLLM weights before loading to ensure device memory is mapped. Without this, load_weights() writes to
        # freed/unmapped memory when sleep mode is active, which crashes on backends with strict physical memory
        # management (e.g., Ascend NPU). See https://github.com/huggingface/trl/issues/5142
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])

        model = self.model
        accelerator = self.accelerator

        if is_peft_model(model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with self._dist.gather_params(list(model.parameters())):
                model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self._dist.is_fsdp:  # note if using FSDP, gather_params is a no-op
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
                        if model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        self._push_param_to_vllm(name, param.data)
                # Unmerge adapters while parameters are still gathered
                model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self._dist.is_fsdp:
                self._sync_fsdp_params_to_vllm(model)
            else:
                for name, param in model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with self._dist.gather_params([param]):
                        self._push_param_to_vllm(name, param.data)

        # Reset cache on vLLM
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()

    def _wake_weights_for_generation(self) -> None:
        """Restore colocated weights without reloading an immutable LoRA base model."""
        if self.mode != "colocate" or not self.enable_sleep_mode:
            return

        empty_cache()  # required to avoid OOM in some cases
        self.llm.wake_up(tags=["weights"])
        if self.weight_sync_mode == "lora":
            # Native LoRA synchronization never mutates the vLLM base model. Level-1 sleep restores its CPU-backed
            # allocations on wake, and the adapter is refreshed separately by LoRARequest(load_inplace=True). Calling
            # reload_weights here is both unnecessary and unsupported by vLLM's bitsandbytes loader.
            return

        # Work around https://github.com/vllm-project/vllm/issues/29341 for full-parameter synchronization.
        try:
            self.llm.collective_rpc("reload_weights")
        except NotImplementedError:
            # Non-CUDA vLLM backends (e.g., vllm-ascend's NPUWorkerV1), don't implement reload_weights.
            pass

    def _sleep_colocated_engine(self) -> None:
        """Release colocated memory using the level compatible with the synchronization strategy."""
        if self.mode != "colocate" or not self.enable_sleep_mode:
            return
        # Full synchronization can reconstruct discarded weights from the trainer, so it retains vLLM's level-2
        # behavior. Native LoRA synchronization keeps the quantized base immutable; level 1 preserves a CPU backup
        # because vLLM cannot reload a bitsandbytes checkpoint after level 2 discards those allocations.
        level = 1 if self.weight_sync_mode == "lora" else 2
        self.llm.sleep(level=level)

    def _collect_generation_metrics(self) -> None:
        """Snapshot optional MTP and KV-cache runtime metrics before the engine sleeps."""
        self.last_generation_metrics = {}
        if self.mode != "colocate":
            return
        if self.speculative_config is not None and hasattr(self.llm, "get_metrics"):
            current: dict[str, float] = {}
            for metric in self.llm.get_metrics():
                normalized_name = _SPEC_DECODE_COUNTERS.get(getattr(metric, "name", ""))
                value = getattr(metric, "value", None)
                if normalized_name is not None and isinstance(value, (int, float)):
                    current[normalized_name] = current.get(normalized_name, 0.0) + float(value)
            spec_metrics, self._spec_decode_counter_snapshot = _compute_spec_decode_counter_delta(
                current, self._spec_decode_counter_snapshot
            )
            self.last_generation_metrics.update(spec_metrics)
        if self._kv_cache_capacity_tokens is not None:
            self.last_generation_metrics[_KV_CACHE_CAPACITY_METRIC] = self._kv_cache_capacity_tokens
        if self._kv_cache_peak_tracker is not None:
            self.last_generation_metrics[_KV_CACHE_PEAK_USAGE_METRIC] = self._kv_cache_peak_tracker.peak_usage_ratio

    def generate(
        self,
        prompts: list[list[int]],
        images: list[list | None] | None,
        num_generations: int,
        profiler: ProfilingContext | None = None,
    ) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of token ID lists, one per prompt (already tokenized).
            images: Optional list of image lists for VLM support. Each element is a list of PIL images for the
                corresponding prompt, or `None` if no images for that prompt. `None` if no images at all.
            num_generations: Number of generations per prompt.
            profiler: Optional profiler for performance tracking.

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, logprob_token_ids).

            - `prompt_ids`: `list[list[int]]` of shape `(batch_size, prompt_len)`.
            - `completion_ids`: `list[list[int]]` of shape `(batch_size, completion_len)`.
            - `logprobs`: `list[list[list[float | None]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `logprob_token_ids`: `list[list[list[int]]]` of shape `(batch_size, completion_len, num_logprobs)`.

            `num_logprobs` is 1 when `logprobs=0`, or up to N+1 when `logprobs=N` (the sampled token is always included
            and may fall outside the top-N).
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length

        # Wake up colocated vLLM weights if needed (idempotent if already awake from sync_weights).
        self._wake_weights_for_generation()

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            all_prompts = gather_object(prompts)
            # Always gather images (even when None) to avoid deadlock: images may be None on some ranks
            # and non-None on others in mixed datasets, and gather_object is a collective operation.
            all_images = gather_object(images if images is not None else [None] * len(prompts))
            if all(img is None for img in all_images):
                all_images = None

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and
                # generate num_generations outputs for each one. This is faster than generating outputs for each
                # duplicate prompt individually.
                ordered_set_of_prompt_ids = all_prompts[::num_generations]
                ordered_set_of_images = all_images[::num_generations] if all_images is not None else None

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                }
                with profiler:
                    output = self.vllm_client.generate(
                        prompts=ordered_set_of_prompt_ids,
                        images=ordered_set_of_images,
                        **sampling_params,
                    )
                    payload = (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output.get("logprob_token_ids"),
                    )
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_logprob_token_ids = obj_list[0]

            # vllm_client.generate(n=num_generations) returns num_generations completions per prompt.
            # Duplicate prompt_ids to align with per-completion entries.
            all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            process_slice = slice(
                accelerator.process_index * len(prompts),
                (accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
            logprob_token_ids = all_logprob_token_ids[process_slice] if all_logprob_token_ids is not None else None

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0 if min_p is None else min_p,
                "max_tokens": max_completion_length,
                "logprobs": self.logprobs,
            }
            generation_kwargs.update(self.generation_kwargs)

            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            if self.tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts)
                gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                all_prompts = [p for sublist in gathered_prompts for p in sublist]
                # Always gather images (even when None) to avoid deadlock: images may be None on some
                # ranks and non-None on others in mixed datasets, and all_gather_object is collective.
                local_images = images if images is not None else [None] * len(prompts)
                gathered_images = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_images, local_images, group=self.tp_group)
                all_images = [img for sublist in gathered_images for img in sublist]
                if all(img is None for img in all_images):
                    all_images = None
            else:
                all_prompts = prompts
                all_images = images

            if self.enable_sleep_mode:
                self.llm.wake_up(tags=["kv_cache"])

            # Build vLLM-compatible prompt inputs with token IDs and optional multi-modal data
            vllm_prompts = []
            if all_images is not None:
                for ids, img_list in zip(all_prompts, all_images, strict=True):
                    row = {"prompt_token_ids": ids}
                    if img_list is not None:
                        row["multi_modal_data"] = {"image": img_list if len(img_list) > 1 else img_list[0]}
                    vllm_prompts.append(row)
            else:
                vllm_prompts = [{"prompt_token_ids": ids} for ids in all_prompts]

            # When PEFT is used, DDP gradient all-reduce only covers the small LoRA parameters, so
            # NCCL operations complete very quickly. On non-NVLink hardware (e.g. A40/A100), vLLM's
            # TP NCCL collective can race with NCCL's internal P2P/SHM channel cleanup from that
            # all-reduce, causing llm.generate() to hang. A barrier on the default process group
            # forces NCCL to fully drain before vLLM's TP communication starts. We pass device_ids
            # so NCCL uses this rank's device rather than guessing, which itself risks a hang.
            # See https://github.com/huggingface/trl/issues/3671
            if is_peft_model(self.model) and self.tensor_parallel_size > 1:
                torch.distributed.barrier(device_ids=[accelerator.local_process_index])

            with profiler:
                if self._kv_cache_peak_tracker is not None:
                    self._kv_cache_peak_tracker.reset()
                all_outputs = self.llm.generate(
                    vllm_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=self._lora_request,
                )

            all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            all_logprobs, all_logprob_token_ids = extract_logprobs(all_outputs)

            if self.tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                prompt_ids = all_prompt_ids[tp_slice]
                completion_ids = all_completion_ids[tp_slice]
                logprobs = all_logprobs[tp_slice] if all_logprobs is not None else None
                logprob_token_ids = all_logprob_token_ids[tp_slice] if all_logprob_token_ids is not None else None
            else:
                prompt_ids = all_prompt_ids
                completion_ids = all_completion_ids
                logprobs = all_logprobs
                logprob_token_ids = all_logprob_token_ids

            self._collect_generation_metrics()
            self._sleep_colocated_engine()

        return prompt_ids, completion_ids, logprobs, logprob_token_ids
