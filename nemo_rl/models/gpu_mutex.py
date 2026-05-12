"""Ray-based per-GPU mutex (debug #63 v65).

Used to serialize CUDA driver activity (cudaFree from PyTorch caching alloc /
cuMemMap from vLLM CuMemAllocator) across distinct OS processes that share a
physical GPU. fcntl-based serialization (v62-v64) failed because EngineCore
subprocesses do not participate in the file lock. A Ray detached actor with
explicit acquire/release works across all Ray actor processes regardless of
fork/spawn semantics.

Set ``RLIX_DEBUG=1`` to emit per-acquire/release telemetry.
"""

import os

import ray

_RLIX_DEBUG = bool(os.environ.get("RLIX_DEBUG"))


@ray.remote(num_cpus=0)  # pragma: no cover
class _GPUMutexActor:
    """Singleton-per-GPU mutex actor.

    Implementation: a queue of waiter tokens. ``acquire`` returns immediately
    iff the lock is free; otherwise the request blocks until an earlier
    ``release`` runs. Ray's default async actor concurrency lets multiple
    inflight ``acquire`` RPCs queue up on ``self._lock`` without blocking
    actor execution overall.
    """

    def __init__(self) -> None:
        import asyncio
        self._lock = asyncio.Lock()
        self._holder = None

    async def acquire(self, owner: str = "?") -> None:
        import time as _t
        _enter = _t.time()
        await self._lock.acquire()
        _granted = _t.time()
        prev = self._holder
        self._holder = owner
        if _RLIX_DEBUG:
            print(
                f"[RLIX_MUTEX_LOG] t={_granted:.6f} fn=acquire owner={owner} "
                f"prev_holder={prev} wait_ms={(_granted-_enter)*1000:.1f}",
                flush=True,
            )

    async def release(self) -> None:
        # Codex: do NOT guard with `if self._lock.locked()` — the asyncio.Lock
        # may already be re-acquired by a queued waiter at this exact tick,
        # producing a double-release that lets two callers hold the mutex.
        # Unconditional release raises RuntimeError on programmer error,
        # which is the desired fail-loud behavior.
        import time as _t
        prev = self._holder
        self._holder = None
        self._lock.release()
        if _RLIX_DEBUG:
            print(
                f"[RLIX_MUTEX_LOG] t={_t.time():.6f} fn=release prev_holder={prev}",
                flush=True,
            )


def get_gpu_mutex(gpu_index: int = 0):
    """Return (creating if needed) a detached Ray actor for the given GPU.

    Detached + named so all Ray actors in any namespace share the singleton.
    """
    name = f"RLixGPUMutex_gpu{gpu_index}"
    return _GPUMutexActor.options(
        name=name, lifetime="detached", get_if_exists=True, num_cpus=0
    ).remote()


def gpu_mutex_acquire_sync(gpu_index: int = 0, owner: str = "?") -> object:
    """Synchronously acquire the per-GPU Ray mutex.

    Returns a token (the actor handle) to pass to ``gpu_mutex_release_sync``.
    Suitable for non-async call sites (e.g., MegatronPolicyWorker.offload_after_refit).
    """
    handle = get_gpu_mutex(gpu_index)
    ray.get(handle.acquire.remote(owner))
    return handle


def gpu_mutex_release_sync(handle: object) -> None:
    ray.get(handle.release.remote())
