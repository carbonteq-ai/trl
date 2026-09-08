import inspect

import pytest


AsyncLLM = pytest.importorskip("vllm.v1.engine.async_llm").AsyncLLM


def test_native_async_llm_implements_session_engine_contract():
    generate = inspect.signature(AsyncLLM.generate)
    assert tuple(generate.parameters)[:4] == ("self", "prompt", "sampling_params", "request_id")
    assert "lora_request" in generate.parameters
    assert inspect.isasyncgenfunction(AsyncLLM.generate)

    for method_name in ("abort", "sleep", "wake_up"):
        assert inspect.iscoroutinefunction(getattr(AsyncLLM, method_name))
    assert not inspect.iscoroutinefunction(AsyncLLM.shutdown)
