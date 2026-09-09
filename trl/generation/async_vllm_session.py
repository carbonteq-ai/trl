"""Per-request asynchronous vLLM lifecycle for online rollouts.

This module deliberately owns only inference lifecycle.  It does not know about
environments, rewards, trainers, or optimizer steps.  Callers establish a fixed
policy version with :meth:`synchronize_policy`, admit requests with
:meth:`open_policy`, then drain and suspend the engine before an optimizer can
mutate the actor.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class AsyncVllmEngine(Protocol):
    """The small, stable subset of the vLLM async engine used by a session."""

    def generate(
        self,
        prompt: Mapping[str, Any],
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any | None = None,
    ) -> AsyncIterator[Any]: ...

    async def abort(self, request_id: str) -> None: ...

    async def sleep(self, level: int = 1) -> None: ...

    async def wake_up(self, tags: list[str] | None = None) -> None: ...

    def shutdown(self) -> None: ...


PolicySynchronizer = Callable[[str], Awaitable[None] | None]


class SessionPhase(StrEnum):
    READY = "ready"
    COLLECTING = "collecting"
    DRAINING = "draining"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class AsyncGenerationRequest:
    """One tokenized request submitted to the engine's continuous scheduler."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: Any
    lora_request: Any | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("async generation request_id cannot be empty")
        if not self.prompt_token_ids:
            raise ValueError("async generation requires at least one prompt token")


class AsyncVllmSession:
    """Fence asynchronous request execution to one synchronized policy version.

    ``synchronize_policy`` is intentionally supplied by the owning trainer. It
    may reload a LoRA adapter or stream full weights, but it is never invoked
    while this session has admitted live generation requests.
    """

    def __init__(
        self,
        engine: AsyncVllmEngine,
        synchronize: PolicySynchronizer,
        *,
        sleep_level: int = 1,
        default_lora_request: Any | None = None,
    ) -> None:
        if sleep_level < 1:
            raise ValueError("async vLLM sleep level must be positive")
        self._engine = engine
        self._synchronize = synchronize
        self._sleep_level = sleep_level
        self._default_lora_request = default_lora_request
        self._phase = SessionPhase.READY
        self._policy_version: str | None = None
        # AsyncLLM starts resident. vLLM supports a staged wake: weights can be
        # restored for synchronization while KV cache and request scheduling
        # remain suspended, then generation capacity is restored on open.
        self._weights_resident = True
        self._generation_ready = True
        self._requests: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def policy_version(self) -> str | None:
        return self._policy_version

    @property
    def active_request_ids(self) -> frozenset[str]:
        return frozenset(self._requests)

    async def synchronize_policy(self, version: str) -> None:
        """Wake, synchronize weights, and make ``version`` ready for collection."""
        if not version.strip():
            raise ValueError("policy version cannot be empty")
        async with self._lock:
            self._require_phase(SessionPhase.READY, SessionPhase.SUSPENDED)
            if self._requests:
                raise RuntimeError("cannot synchronize a policy while generation requests are active")
            if not self._weights_resident:
                await self._engine.wake_up(tags=["weights"])
                self._weights_resident = True
            result = self._synchronize(version)
            if inspect.isawaitable(result):
                await result
            self._policy_version = version
            self._phase = SessionPhase.READY

    async def open_policy(self, version: str) -> None:
        """Open request admission for an already-synchronized policy version."""
        async with self._lock:
            self._require_phase(SessionPhase.READY)
            if self._policy_version != version:
                raise RuntimeError(
                    f"cannot open policy {version!r}; synchronized policy is {self._policy_version!r}"
                )
            if not self._generation_ready:
                # A weights-only wake intentionally leaves vLLM's scheduler
                # paused until the KV cache is resident again.
                await self._engine.wake_up(tags=["kv_cache"])
                self._generation_ready = True
            self._phase = SessionPhase.COLLECTING

    async def generate(self, request: AsyncGenerationRequest) -> Any:
        """Return the final output for one independently scheduled request."""
        async with self._lock:
            self._require_phase(SessionPhase.COLLECTING)
            if request.request_id in self._requests:
                raise ValueError(f"duplicate active vLLM request id {request.request_id!r}")
            task = asyncio.current_task()
            if task is None:  # pragma: no cover - asyncio always has one
                raise RuntimeError("async vLLM generation requires an asyncio task")
            self._requests[request.request_id] = task

        final_output: Any | None = None
        try:
            async for output in self._engine.generate(
                {"prompt_token_ids": list(request.prompt_token_ids)},
                request.sampling_params,
                request.request_id,
                lora_request=getattr(request, "lora_request", None) or self._default_lora_request,
            ):
                final_output = output
            if final_output is None:
                raise RuntimeError(f"vLLM request {request.request_id!r} finished without an output")
            return final_output
        except asyncio.CancelledError:
            await self._engine.abort(request.request_id)
            raise
        finally:
            async with self._lock:
                self._requests.pop(request.request_id, None)

    async def stop_admission(self) -> None:
        """Forbid new requests while allowing already-admitted requests to finish."""
        async with self._lock:
            self._require_phase(SessionPhase.COLLECTING)
            self._phase = SessionPhase.DRAINING

    async def abort(self, request_id: str) -> bool:
        """Cancel a live request and wait for its engine-abort acknowledgement.

        ``False`` means the request already reached a terminal state.  A
        successful return guarantees that the request task cannot later be
        mistaken for a completed rollout.
        """
        async with self._lock:
            task = self._requests.get(request_id)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def drain(self) -> None:
        """Wait until every admitted request is terminal without admitting new work."""
        async with self._lock:
            self._require_phase(SessionPhase.DRAINING)
            requests = tuple(self._requests.values())
        if requests:
            await asyncio.gather(*requests)

    async def suspend_for_update(self) -> None:
        """Drain requests and release inference residency before an actor update."""
        if self._phase is SessionPhase.COLLECTING:
            await self.stop_admission()
        await self.drain()
        async with self._lock:
            self._require_phase(SessionPhase.DRAINING)
            if self._requests:
                raise RuntimeError("cannot suspend vLLM while generation requests are active")
            await self._engine.sleep(level=self._sleep_level)
            self._weights_resident = False
            self._generation_ready = False
            self._phase = SessionPhase.SUSPENDED

    async def aclose(self) -> None:
        """Cancel any remaining request and shut down the engine exactly once."""
        async with self._lock:
            if self._phase is SessionPhase.CLOSED:
                return
            request_ids = tuple(self._requests)
            self._phase = SessionPhase.DRAINING
        for request_id in request_ids:
            await self.abort(request_id)
        async with self._lock:
            requests = tuple(self._requests.values())
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)
        try:
            # vLLM's CUDA allocator expects sleeping allocations to be mapped
            # before engine teardown. This is a close-only transition; request
            # admission remains fenced.
            if not self._generation_ready:
                await self._engine.wake_up()
                self._weights_resident = True
                self._generation_ready = True
        finally:
            self._engine.shutdown()
            async with self._lock:
                self._phase = SessionPhase.CLOSED

    def _require_phase(self, *allowed: SessionPhase) -> None:
        if self._phase not in allowed:
            values = ", ".join(value.value for value in allowed)
            raise RuntimeError(f"invalid async vLLM session phase {self._phase.value!r}; expected one of {values}")
