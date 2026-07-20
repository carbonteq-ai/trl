from types import SimpleNamespace

import pytest

from trl.generation import vllm_generation
from trl.generation.vllm_generation import VLLMGeneration


def test_colocated_engine_receives_speculative_config(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

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
