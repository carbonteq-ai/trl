from types import SimpleNamespace

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
    )

    assert captured["speculative_config"] == speculative
