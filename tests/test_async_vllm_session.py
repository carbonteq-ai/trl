import asyncio
from types import SimpleNamespace

import pytest

from trl.generation.async_vllm_session import AsyncGenerationRequest, AsyncVllmSession, SessionPhase


class FakeAsyncEngine:
    def __init__(self):
        self.started = {}
        self.release = {}
        self.aborted = []
        self.wakes = []
        self.sleeps = []
        self.shutdowns = 0
        self.lora_requests = []

    async def generate(self, prompt, sampling_params, request_id, *, lora_request=None):
        del prompt, sampling_params
        self.lora_requests.append(lora_request)
        self.started[request_id].set()
        await self.release[request_id].wait()
        yield {"request_id": request_id}

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.release[request_id].set()

    async def wake_up(self, tags=None):
        self.wakes.append(tags)

    async def sleep(self, level=1):
        self.sleeps.append(level)

    def shutdown(self):
        self.shutdowns += 1


def request(request_id):
    return AsyncGenerationRequest(request_id=request_id, prompt_token_ids=(1,), sampling_params={})


@pytest.mark.asyncio
async def test_session_applies_default_lora_to_endpoint_shaped_requests():
    engine = FakeAsyncEngine()
    engine.started = {"endpoint": asyncio.Event()}
    engine.release = {"endpoint": asyncio.Event()}
    engine.release["endpoint"].set()
    adapter = object()
    session = AsyncVllmSession(engine, lambda _version: None, default_lora_request=adapter)
    await session.synchronize_policy("policy-1")
    await session.open_policy("policy-1")

    await session.generate(
        SimpleNamespace(request_id="endpoint", prompt_token_ids=(1,), sampling_params={})
    )

    assert engine.lora_requests == [adapter]


@pytest.mark.asyncio
async def test_requests_complete_independently_and_policy_cannot_change_while_collecting():
    engine = FakeAsyncEngine()
    engine.started = {request_id: asyncio.Event() for request_id in ("long", "short")}
    engine.release = {request_id: asyncio.Event() for request_id in ("long", "short")}
    synchronized = []

    async def synchronize(version):
        synchronized.append(version)

    session = AsyncVllmSession(engine, synchronize)
    await session.synchronize_policy("policy-1")
    await session.open_policy("policy-1")
    long = asyncio.create_task(session.generate(request("long")))
    short = asyncio.create_task(session.generate(request("short")))
    await asyncio.gather(engine.started["long"].wait(), engine.started["short"].wait())

    engine.release["short"].set()
    assert await short == {"request_id": "short"}
    assert not long.done()
    with pytest.raises(RuntimeError, match="invalid async vLLM session phase"):
        await session.synchronize_policy("policy-2")

    await session.stop_admission()
    engine.release["long"].set()
    await session.suspend_for_update()
    assert await long == {"request_id": "long"}
    assert session.phase is SessionPhase.SUSPENDED
    assert engine.sleeps == [1]

    await session.synchronize_policy("policy-2")
    await session.open_policy("policy-2")
    assert synchronized == ["policy-1", "policy-2"]
    assert engine.wakes == [["weights"], ["kv_cache"]]


@pytest.mark.asyncio
async def test_abort_and_close_fence_requests_and_shutdown_once():
    engine = FakeAsyncEngine()
    engine.started = {"one": asyncio.Event()}
    engine.release = {"one": asyncio.Event()}
    session = AsyncVllmSession(engine, lambda _version: None)
    await session.synchronize_policy("policy-1")
    await session.open_policy("policy-1")
    task = asyncio.create_task(session.generate(request("one")))
    await engine.started["one"].wait()

    assert await session.abort("one") is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await session.abort("one") is False
    await session.aclose()
    await session.aclose()
    assert engine.aborted == ["one"]
    assert engine.shutdowns == 1
    assert session.phase is SessionPhase.CLOSED


@pytest.mark.asyncio
async def test_close_wakes_a_suspended_engine_before_shutdown():
    engine = FakeAsyncEngine()
    session = AsyncVllmSession(engine, lambda _version: None)
    await session.synchronize_policy("policy-1")
    await session.open_policy("policy-1")
    await session.suspend_for_update()

    await session.aclose()

    assert engine.sleeps == [1]
    assert engine.wakes == [None]
    assert engine.shutdowns == 1


@pytest.mark.asyncio
async def test_suspended_batch_probe_resumes_and_resuspends_engine():
    engine = FakeAsyncEngine()
    engine.started = {request_id: asyncio.Event() for request_id in ("probe-1", "probe-2")}
    engine.release = {request_id: asyncio.Event() for request_id in ("probe-1", "probe-2")}
    for release in engine.release.values():
        release.set()
    session = AsyncVllmSession(engine, lambda _version: None)
    await session.synchronize_policy("policy-1")
    await session.open_policy("policy-1")
    await session.suspend_for_update()

    outputs = await session.generate_suspended_batch([request("probe-1"), request("probe-2")])

    assert outputs == [{"request_id": "probe-1"}, {"request_id": "probe-2"}]
    assert session.phase is SessionPhase.SUSPENDED
    assert engine.wakes == [["weights"], ["kv_cache"]]
    assert engine.sleeps == [1, 1]
