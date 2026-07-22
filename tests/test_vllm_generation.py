from types import SimpleNamespace

import pytest

from trl.generation import vllm_generation
from trl.generation.vllm_generation import (
    VLLMGeneration,
    _accumulate_spec_decode_metrics,
    _compute_spec_decode_counter_delta,
)


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
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=loader))
            )
        )
    )

    generation._push_param_to_vllm("model.layers.0.weight", "tensor")
    generation._push_param_to_vllm("language_model.model.norm.weight", "tensor-2")

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
        peft_config = {"default": SimpleNamespace(r=8)}

        def __init__(self):
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


def test_colocated_full_weight_wake_retains_reload_workaround(monkeypatch):
    calls = []
    generation = object.__new__(VLLMGeneration)
    generation.mode = "colocate"
    generation.enable_sleep_mode = True
    generation.weight_sync_mode = "full"
    generation.llm = SimpleNamespace(
        wake_up=lambda *, tags: calls.append(("wake_up", tags)),
        collective_rpc=lambda method: calls.append(("collective_rpc", method)),
    )
    monkeypatch.setattr(vllm_generation, "empty_cache", lambda: calls.append(("empty_cache",)))

    generation._wake_weights_for_generation()

    assert calls == [
        ("empty_cache",),
        ("wake_up", ["weights"]),
        ("collective_rpc", "reload_weights"),
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
