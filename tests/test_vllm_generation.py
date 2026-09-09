import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from trl.generation import vllm_generation
from trl.generation.vllm_generation import (
    VLLMGeneration,
    _accumulate_spec_decode_metrics,
    _compute_spec_decode_counter_delta,
)


_DISTRIBUTED_ENVIRONMENT_VARIABLES = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@pytest.fixture(autouse=True)
def restore_distributed_environment():
    original_environment = {name: os.environ.get(name) for name in _DISTRIBUTED_ENVIRONMENT_VARIABLES}
    yield
    for name, value in original_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_speculative_counters_are_reported_as_per_generation_deltas():
    first, snapshot = _compute_spec_decode_counter_delta(
        {"drafts": 10.0, "draft_tokens": 20.0, "accepted_tokens": 15.0}, {}
    )
    second, snapshot = _compute_spec_decode_counter_delta(
        {"drafts": 14.0, "draft_tokens": 28.0, "accepted_tokens": 21.0}, snapshot
    )

    assert first == {
        "rollout/spec_num_drafts": 10.0,
        "rollout/spec_num_draft_tokens": 20.0,
        "rollout/spec_num_accepted_tokens": 15.0,
        "rollout/spec_accept_rate": 0.75,
        "rollout/spec_accept_length": 2.5,
    }
    assert second == {
        "rollout/spec_num_drafts": 4.0,
        "rollout/spec_num_draft_tokens": 8.0,
        "rollout/spec_num_accepted_tokens": 6.0,
        "rollout/spec_accept_rate": 0.75,
        "rollout/spec_accept_length": 2.5,
    }
    assert snapshot == {"drafts": 14.0, "draft_tokens": 28.0, "accepted_tokens": 21.0}


def test_colocated_generation_collects_vllm_runtime_metrics():
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.speculative_config = {"method": "mtp", "num_speculative_tokens": 1}
    generation.last_generation_metrics = {}
    generation._spec_decode_counter_snapshot = {}
    generation._kv_cache_capacity_tokens = 4096.0
    generation._kv_cache_peak_tracker = vllm_generation._KvCachePeakTracker()
    generation._kv_cache_peak_tracker.peak_usage_ratio = 0.625
    generation.llm = SimpleNamespace(
        get_metrics=lambda: [
            SimpleNamespace(name="vllm:spec_decode_num_drafts", value=4),
            SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=8),
            SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=6),
            SimpleNamespace(name="vllm:unrelated", value=100),
        ]
    )

    generation._collect_generation_metrics()

    assert generation.last_generation_metrics == {
        "rollout/spec_num_drafts": 4.0,
        "rollout/spec_num_draft_tokens": 8.0,
        "rollout/spec_num_accepted_tokens": 6.0,
        "rollout/spec_accept_rate": 0.75,
        "rollout/spec_accept_length": 2.5,
        "rollout/kv_cache_capacity_tokens": 4096.0,
        "rollout/kv_cache_peak_usage_ratio": 0.625,
    }


def test_kv_cache_peak_tracker_retains_maximum_scheduler_sample():
    tracker = vllm_generation._KvCachePeakTracker()

    tracker.record(SimpleNamespace(kv_cache_usage=0.25), None)
    tracker.record(SimpleNamespace(kv_cache_usage=0.75), None)
    tracker.record(SimpleNamespace(kv_cache_usage=0.5), None)

    assert tracker.peak_usage_ratio == 0.75
    tracker.reset()
    assert tracker.peak_usage_ratio == 0.0


def test_speculative_turn_metrics_accumulate_as_step_totals():
    buffer = {}
    _accumulate_spec_decode_metrics(
        buffer,
        {
            "rollout/spec_num_drafts": 4.0,
            "rollout/spec_num_draft_tokens": 8.0,
            "rollout/spec_num_accepted_tokens": 6.0,
            "rollout/spec_accept_rate": 0.75,
            "rollout/spec_accept_length": 2.5,
        },
    )
    _accumulate_spec_decode_metrics(
        buffer,
        {
            "rollout/spec_num_drafts": 2.0,
            "rollout/spec_num_draft_tokens": 4.0,
            "rollout/spec_num_accepted_tokens": 2.0,
            "rollout/spec_accept_rate": 0.5,
            "rollout/spec_accept_length": 2.0,
        },
    )

    assert buffer == {
        "rollout/spec_num_drafts": [6.0],
        "rollout/spec_num_draft_tokens": [12.0],
        "rollout/spec_num_accepted_tokens": [8.0],
        "rollout/spec_accept_rate": [2 / 3],
        "rollout/spec_accept_length": [1 + 8 / 6],
    }


def test_kv_cache_runtime_metrics_keep_capacity_and_step_peak():
    buffer = {}

    _accumulate_spec_decode_metrics(
        buffer,
        {"rollout/kv_cache_capacity_tokens": 4096.0, "rollout/kv_cache_peak_usage_ratio": 0.4},
    )
    _accumulate_spec_decode_metrics(
        buffer,
        {"rollout/kv_cache_capacity_tokens": 4096.0, "rollout/kv_cache_peak_usage_ratio": 0.7},
    )

    assert buffer == {
        "rollout/kv_cache_capacity_tokens": [4096.0],
        "rollout/kv_cache_peak_usage_ratio": [0.7],
    }


def test_colocated_engine_receives_speculative_config(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(cache_config=SimpleNamespace(kv_cache_size_tokens=2048)),
                logger_manager=SimpleNamespace(stat_loggers=[]),
            )

    class FakeModel:
        name_or_path = "model"

        def named_modules(self):
            return []

    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    speculative = {"method": "qwen3_next_mtp", "num_speculative_tokens": 2}
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)

    VLLMGeneration(
        model=FakeModel(),
        accelerator=accelerator,
        processing_class=object(),
        speculative_config=speculative,
        engine_kwargs={"skip_mm_profiling": True},
    )

    assert captured["speculative_config"] == speculative
    assert captured["skip_mm_profiling"] is True
    assert captured["disable_log_stats"] is False


def test_colocated_engine_allows_bounded_sequence_waves(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(cache_config=SimpleNamespace(kv_cache_size_tokens=2048)),
                logger_manager=SimpleNamespace(stat_loggers=[]),
            )

    class FakeModel:
        name_or_path = "model"

        def named_modules(self):
            return []

    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)

    VLLMGeneration(
        model=FakeModel(),
        accelerator=accelerator,
        processing_class=object(),
        max_num_seqs=256,
        engine_kwargs={"max_num_seqs": 32, "max_num_batched_tokens": 32768},
    )

    assert captured["max_num_seqs"] == 32
    assert captured["max_num_batched_tokens"] == 32768


def test_async_colocated_engine_is_lazy_and_refreshes_lora_on_its_serving_loop(monkeypatch):
    captured = {}

    class FakePeftModel:
        name_or_path = "model"
        peft_config = {"default": SimpleNamespace(r=8)}

        def named_modules(self):
            return []

        def save_pretrained(self, path, safe_serialization):
            captured["saved"] = (path, safe_serialization)

    class FakeEngine:
        def __init__(self):
            self.loaded = {1}

        async def list_loras(self):
            return set(self.loaded)

        async def remove_lora(self, identifier):
            self.loaded.remove(identifier)

        async def add_lora(self, request):
            self.loaded.add(request.lora_int_id)
            return True

        async def reset_prefix_cache(self):
            captured["reset"] = True

        async def sleep(self, level=1):
            captured["sleep"] = level

        def shutdown(self):
            captured["shutdown"] = True

    engine = FakeEngine()

    class FakeAsyncLLM:
        @classmethod
        def from_engine_args(cls, args):
            captured["engine_args"] = args
            return engine

    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    model = FakePeftModel()
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "is_peft_model", lambda value: value is model)
    monkeypatch.setattr(vllm_generation, "AsyncEngineArgs", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    monkeypatch.setattr(vllm_generation, "AsyncLLM", FakeAsyncLLM, raising=False)
    monkeypatch.setattr(
        vllm_generation,
        "LoRARequest",
        lambda name, identifier, path, *, load_inplace: SimpleNamespace(
            lora_name=name,
            lora_int_id=identifier,
            lora_path=path,
            load_inplace=load_inplace,
        ),
        raising=False,
    )

    generation = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=object(),
        request_mode="async",
        weight_sync_mode="lora",
        enable_sleep_mode=True,
    )
    assert "engine_args" not in captured
    generation.sync_weights()

    async def exercise():
        session = await generation.create_async_session()
        assert await generation.create_async_session() is session
        await session.synchronize_policy("optimizer-step-0")
        await session.open_policy("optimizer-step-0")
        await session.stop_admission()
        await session.suspend_for_update()

    asyncio.run(exercise())

    assert captured["saved"] == (generation._lora_directory.name, True)
    assert captured["reset"] is True
    assert captured["sleep"] == 1
    assert engine.loaded == {1}


def test_async_request_mode_rejects_synchronous_batch_generation():
    generation = object.__new__(VLLMGeneration)
    generation.request_mode = "async"

    with pytest.raises(RuntimeError, match="create_async_session"):
        generation.generate([[1]], None, 1)


def test_colocated_generation_sends_bounded_request_waves_in_order():
    generation = object.__new__(VLLMGeneration)
    generation.max_num_seqs = 2
    generation._lora_request = None
    calls = []

    class FakeLLM:
        def generate(self, prompts, *, sampling_params, use_tqdm, lora_request):
            calls.append((prompts, sampling_params, use_tqdm, lora_request))
            return [SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[]) for prompt in prompts]

    generation.llm = FakeLLM()
    prompts = [{"prompt_token_ids": [index]} for index in range(5)]
    sampling_params = object()

    outputs = generation._generate_colocated_waves(prompts, sampling_params)

    assert [[row["prompt_token_ids"] for row in call[0]] for call in calls] == [[[0], [1]], [[2], [3]], [[4]]]
    assert [output.prompt_token_ids for output in outputs] == [[0], [1], [2], [3], [4]]
    assert all(call[1:] == (sampling_params, False, None) for call in calls)


def test_explicit_runtime_stats_setting_is_accepted_for_speculative_metrics(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(cache_config=SimpleNamespace(kv_cache_size_tokens=2048)),
                logger_manager=SimpleNamespace(stat_loggers=[]),
            )

    class FakeModel:
        name_or_path = "model"

        def named_modules(self):
            return []

    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)

    VLLMGeneration(
        model=FakeModel(),
        accelerator=accelerator,
        processing_class=object(),
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        engine_kwargs={"disable_log_stats": False},
    )

    assert captured["disable_log_stats"] is False

    with pytest.raises(ValueError, match="cannot override TRL-controlled"):
        VLLMGeneration(
            model=FakeModel(),
            accelerator=accelerator,
            processing_class=object(),
            engine_kwargs={"model": "other"},
        )

    with pytest.raises(ValueError, match="ending with"):
        VLLMGeneration(
            model=FakeModel(),
            accelerator=accelerator,
            processing_class=object(),
            weight_name_prefix="language_model",
        )


def test_weight_name_prefix_is_applied_at_the_vllm_boundary():
    captured = []
    loader = SimpleNamespace(load_weights=lambda weights: captured.extend(weights))
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.weight_name_prefix = "language_model."
    generation.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=loader)))
        )
    )

    generation.weight_sync_mode = "full"
    generation.enable_sleep_mode = False
    generation.accelerator = SimpleNamespace(is_main_process=True)
    generation.llm.reset_prefix_cache = lambda: None
    generation._iter_named_params = lambda: iter([
        (generation._fix_param_name_to_vllm("model.layers.0.weight"), "tensor"),
        (generation._fix_param_name_to_vllm("language_model.model.norm.weight"), "tensor-2"),
    ])
    generation.sync_weights()

    assert captured == [
        ("language_model.model.layers.0.weight", "tensor"),
        ("language_model.model.norm.weight", "tensor-2"),
    ]


def test_colocated_lora_sync_exports_adapter_without_merging_base_weights(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakePeftModel:
        name_or_path = "model"

        def __init__(self):
            self.peft_config = {"default": SimpleNamespace(r=8)}
            self.saved = []

        def named_modules(self):
            return []

        def save_pretrained(self, path, safe_serialization):
            self.saved.append((path, safe_serialization))

        def merge_adapter(self):
            raise AssertionError("LoRA sync must not merge packed base weights")

    model = FakePeftModel()
    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "is_peft_model", lambda value: value is model)
    monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)
    monkeypatch.setattr(
        vllm_generation,
        "LoRARequest",
        lambda name, identifier, path, *, load_inplace: SimpleNamespace(
            lora_name=name,
            lora_int_id=identifier,
            lora_path=path,
            load_inplace=load_inplace,
        ),
        raising=False,
    )

    generation = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=object(),
        weight_sync_mode="lora",
    )
    generation.sync_weights()

    assert captured["enable_lora"] is True
    assert captured["max_lora_rank"] == 8
    assert model.saved == [(generation._lora_directory.name, True)]
    assert generation._lora_request.lora_path == generation._lora_directory.name
    assert generation._lora_request.load_inplace is True


def test_colocated_lora_sync_applies_composite_model_prefix_to_disposable_adapter(monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

    class FakePeftModel:
        name_or_path = "model"

        def __init__(self):
            self.peft_config = {"default": SimpleNamespace(r=8)}

        def named_modules(self):
            return []

        def save_pretrained(self, path, safe_serialization):
            assert safe_serialization is True
            save_file(
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
                    "base_model.model.language_model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.ones(1),
                },
                Path(path) / "adapter_model.safetensors",
                metadata={"format": "pt", "source": "actor"},
            )

    model = FakePeftModel()
    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "is_peft_model", lambda value: value is model)
    monkeypatch.setattr(vllm_generation, "LLM", FakeLLM, raising=False)
    monkeypatch.setattr(
        vllm_generation,
        "LoRARequest",
        lambda name, identifier, path, *, load_inplace: SimpleNamespace(lora_path=path),
        raising=False,
    )

    generation = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=object(),
        weight_sync_mode="lora",
        weight_name_prefix="language_model.",
    )
    generation.sync_weights()

    exported = Path(generation._lora_request.lora_path) / "adapter_model.safetensors"
    with safe_open(exported, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "base_model.model.language_model.model.layers.0.self_attn.k_proj.lora_A.weight",
            "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        }
        assert handle.metadata() == {"format": "pt", "source": "actor"}


def test_colocated_lora_wake_does_not_reload_immutable_quantized_base(monkeypatch):
    calls = []
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.enable_sleep_mode = True
    generation.weight_sync_mode = "lora"
    generation.llm = SimpleNamespace(
        wake_up=lambda *, tags: calls.append(("wake_up", tags)),
        collective_rpc=lambda method: calls.append(("collective_rpc", method)),
    )
    monkeypatch.setattr(vllm_generation, "empty_cache", lambda: calls.append(("empty_cache",)))

    generation._wake_weights_for_generation()

    assert calls == [("empty_cache",), ("wake_up", ["weights"])]


def test_colocated_full_weight_wake_restores_current_actor_not_checkpoint(monkeypatch):
    calls = []
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.enable_sleep_mode = True
    generation.weight_sync_mode = "full"
    generation._llm_weights_sleeping = True
    generation.sync_weights = lambda: calls.append(("sync_current_actor",))
    generation.llm = SimpleNamespace(
        wake_up=lambda *, tags: calls.append(("wake_up", tags)),
        collective_rpc=lambda method: calls.append(("collective_rpc", method)),
    )
    monkeypatch.setattr(vllm_generation, "empty_cache", lambda: calls.append(("empty_cache",)))

    generation._wake_weights_for_generation()

    assert calls == [
        ("empty_cache",),
        ("wake_up", ["weights"]),
        ("sync_current_actor",),
    ]


@pytest.mark.parametrize(("sync_mode", "expected_level"), [("lora", 1), ("full", 2)])
def test_colocated_sleep_level_preserves_native_lora_base(sync_mode, expected_level):
    calls = []
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.enable_sleep_mode = True
    generation.weight_sync_mode = sync_mode
    generation.llm = SimpleNamespace(sleep=lambda *, level: calls.append(level))

    generation._sleep_colocated_engine()

    assert calls == [expected_level]


def test_colocated_policy_parity_probe_teacher_forces_observed_tokens(monkeypatch):
    calls = []
    captured = {}
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.tensor_parallel_size = 1
    generation.enable_sleep_mode = False
    generation.accelerator = SimpleNamespace()
    generation._lora_request = None
    generation._wake_weights_for_generation = lambda: calls.append("wake")
    generation._sleep_colocated_engine = lambda: calls.append("sleep")

    def fake_sampling_params(**kwargs):
        captured["sampling"] = kwargs
        return kwargs

    def fake_generate(prompts, sampling_params):
        captured["prompts"] = prompts
        assert sampling_params == captured["sampling"]
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 10, 11],
                prompt_logprobs=[
                    None,
                    {2: SimpleNamespace(logprob=-0.1)},
                    {10: SimpleNamespace(logprob=-0.2)},
                    {11: SimpleNamespace(logprob=-0.3)},
                ],
            )
        ]

    monkeypatch.setattr(vllm_generation, "SamplingParams", fake_sampling_params, raising=False)
    generation._generate_colocated_waves = fake_generate

    result = generation.score_completion_logprobs([[1, 2]], [[10, 11]])

    assert result == [[-0.2, -0.3]]
    assert captured["prompts"] == [{"prompt_token_ids": [1, 2, 10, 11]}]
    assert captured["sampling"] == {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": 1,
        "detokenize": False,
    }
    assert calls == ["wake", "sleep"]


def test_lora_sync_rejects_unsupported_execution_shapes(monkeypatch):
    class FakeModel:
        name_or_path = "model"

        def named_modules(self):
            return []

    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None),
        num_processes=1,
        process_index=0,
        local_process_index=0,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(vllm_generation, "is_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_generation, "is_peft_model", lambda _: False)

    with pytest.raises(ValueError, match="requires a PEFT model"):
        VLLMGeneration(
            model=FakeModel(),
            accelerator=accelerator,
            processing_class=object(),
            weight_sync_mode="lora",
        )
