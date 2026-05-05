# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron NCCL process group destroy/reload helpers for partial overlapping.

This module provides two public helpers used by the RLix partial-overlap
training lifecycle:

- :func:`destroy_megatron_nccl_groups` — after a training step finishes and
  the model/optimizer state has been offloaded to CPU, the NCCL communicator
  buffers attached to every Megatron process group still occupy GPU VRAM.
  This helper destroys all NCCL groups tracked in
  ``megatron.core.parallel_state`` so the GPU can be reclaimed by inference
  workers (see :mod:`nemo_rl.models.generation.vllm.vllm_generation`).

- :func:`reload_megatron_nccl_groups` — before the next training step, this
  helper rebuilds the Megatron process groups using the parallel configuration
  captured at ``destroy`` time.

Design rationale (vs. ROLL's ``ReloadableProcessGroup`` monkey-patch):

- ROLL hooks into ``torch.distributed.new_group`` so every NCCL group created
  in the process is automatically tracked and reloadable. That approach is
  intrusive (it modifies ``torch.distributed`` module state) and requires the
  patch to be applied before any group is created.
- NeMo RL uses Megatron-Core, which already keeps a registry of every group
  it creates in ``megatron.core.parallel_state``. We walk that registry
  directly — no monkey-patching, no ordering constraints, and the logic is
  fully contained in this module.

See ``plans/nemorl-port-plan.md`` Feature 11 and ``task3-plan.md`` for the
broader context.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch


# Offload mode selects which teardown path ``destroy_megatron_nccl_groups``
# uses.
#
# - ``"manual"``: walk ``parallel_state`` globals, call
#   ``torch.distributed.destroy_process_group`` on every NCCL ProcessGroup,
#   null the group attributes ourselves, and then delegate the residual
#   non-group state reset (``_GLOBAL_MEMORY_BUFFER``, ``_MPU_*``, virtual
#   pipeline trackers, etc.) to ``parallel_state.destroy_model_parallel()``
#   so we stay forward-compatible with Megatron versions. The manual path's
#   unique contribution versus ``"official"`` is the per-group
#   ``destroy_process_group`` call, which is what actually releases the
#   NCCL communicator buffers synchronously — essential for partial-overlap.
#
# - ``"official"``: delegate to ``parallel_state.destroy_model_parallel()``
#   and rely on Megatron's own cleanup. This is the fallback required by
#   Feature 11 / Gate 2.5: if the manual path exposes problems (stale
#   handles, VRAM not returned, instability across cycles), operators can
#   switch to this mode to retry.
#
# The two modes are expected to be functionally equivalent for a healthy
# Megatron install; they only differ in who drives the teardown.
NcclOffloadMode = Literal["manual", "official"]

_DEFAULT_MODE: NcclOffloadMode = "manual"


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def destroy_megatron_nccl_groups(
    *,
    mode: NcclOffloadMode = _DEFAULT_MODE,
    verbose: bool = False,
) -> dict[str, Any]:
    """Destroy every NCCL process group tracked by ``megatron.core.parallel_state``.

    Safe to call when Megatron parallel state has not been initialized (returns
    an empty snapshot). Idempotent: a second call is a no-op.

    Args:
        mode: Which teardown strategy to use.

            - ``"manual"`` (default): collect every ``ProcessGroup`` held by
              ``parallel_state`` globals, call
              ``torch.distributed.destroy_process_group`` on each, then null
              the globals.
            - ``"official"``: delegate teardown to
              ``parallel_state.destroy_model_parallel()``. This is the
              Gate 2.5 fallback required by Feature 11: if the manual path
              fails (VRAM not released, stale handles, instability across
              destroy/reload cycles), switch to this mode and retry Gate 2.5.
              Reload is always performed by
              ``parallel_state.initialize_model_parallel`` regardless of
              ``mode``.
        verbose: If True, print per-group destroy/skip diagnostics.

    Returns:
        A dictionary with the following keys:

        - ``initialized_before`` (bool): Whether Megatron parallel state was
          initialized at call time.
        - ``num_groups_destroyed`` (int): Number of distinct process groups
          for which ``torch.distributed.destroy_process_group`` was called.
          For ``mode="official"`` this is reported as ``-1`` because the
          official API does not expose a count.
        - ``vram_freed_bytes`` (int): Difference in
          ``torch.cuda.memory_allocated()`` across this call. Note: freed
          memory may remain in the PyTorch caching allocator even after
          ``empty_cache()``; this value is a useful observability signal,
          not a strict assertion.
        - ``state_snapshot`` (dict): Parallel-config snapshot to be passed
          back to :func:`reload_megatron_nccl_groups`. Empty if Megatron
          parallel state was not initialized.
        - ``mode`` (str): The mode that was actually used.

    Raises:
        ValueError: If ``mode`` is not one of ``"manual"`` / ``"official"``.
    """
    if mode not in ("manual", "official"):
        raise ValueError(
            f"Invalid mode {mode!r}; must be 'manual' or 'official'."
        )

    # Lazy import so the module is importable without Megatron installed
    # (e.g. in pure-Python unit tests that mock parallel_state).
    from megatron.core import parallel_state

    stats: dict[str, Any] = {
        "initialized_before": False,
        "num_groups_destroyed": 0,
        "vram_freed_bytes": 0,
        "state_snapshot": {},
        "mode": mode,
    }

    if not parallel_state.model_parallel_is_initialized():
        if verbose:
            print("[nccl_offload] parallel_state not initialized; nothing to do.")
        return stats

    stats["initialized_before"] = True
    # Snapshot must be captured *before* destruction — accessors may start
    # raising once globals are nulled.
    stats["state_snapshot"] = _capture_parallel_state_snapshot()

    vram_before = _get_vram_allocated()

    if mode == "manual":
        destroyed = _destroy_manual(verbose=verbose)
        stats["num_groups_destroyed"] = destroyed
    else:
        _destroy_official(verbose=verbose)
        # The official path does not expose a destroy count; use -1 as a
        # sentinel so callers can distinguish "no-op" (0) from "official
        # path was used" (-1).
        stats["num_groups_destroyed"] = -1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    vram_after = _get_vram_allocated()
    stats["vram_freed_bytes"] = max(0, vram_before - vram_after)

    if verbose:
        destroyed_str = (
            str(stats["num_groups_destroyed"])
            if stats["num_groups_destroyed"] >= 0
            else "(official: unknown)"
        )
        print(
            f"[nccl_offload] mode={mode} destroyed={destroyed_str}, "
            f"freed ~{stats['vram_freed_bytes'] / (1024**2):.1f} MiB"
        )

    return stats


def _destroy_manual(*, verbose: bool) -> int:
    """Walk ``parallel_state`` and call ``destroy_process_group`` on each.

    Returns the number of groups successfully destroyed. Per-group failures
    are tolerated (they are logged in verbose mode and counted only as
    failures); reload will rebuild everything from scratch regardless.

    After the per-group destruction loop, we also invoke
    ``parallel_state.destroy_model_parallel()`` to reset the non-group
    module-level state that Megatron tracks (e.g. ``_GLOBAL_MEMORY_BUFFER``,
    ``_MPU_*_WORLD_SIZE``, ``_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK``). That
    residual state cannot be nulled by our group-attribute walker because
    those globals do not follow the ``_*_GROUP*`` naming convention, and
    leaving them set causes the next ``initialize_model_parallel`` to
    raise ``AssertionError: global memory buffer is already initialized``.
    Delegating to Megatron's own reset keeps us automatically in sync with
    whatever state Megatron adds across versions.
    """
    from megatron.core import parallel_state

    groups = _collect_all_nccl_groups()
    destroyed = 0
    for name, pg in groups:
        try:
            torch.distributed.destroy_process_group(pg)
            destroyed += 1
            if verbose:
                print(f"[nccl_offload] destroyed NCCL group from {name}")
        except Exception as exc:  # noqa: BLE001 - best-effort, keep going
            if verbose:
                print(
                    f"[nccl_offload] destroy failed for {name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    _reset_parallel_state_globals()

    # Hand residual (non-group) module-state cleanup to Megatron's own
    # reset. This is the authoritative list of globals to null and keeps
    # us forward-compatible with Megatron versions that add new ones.
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
            if verbose:
                print(
                    "[nccl_offload] manual: delegated residual state reset "
                    "to parallel_state.destroy_model_parallel()"
                )
        except Exception as exc:  # noqa: BLE001 - surface diagnostics, don't crash
            if verbose:
                print(
                    "[nccl_offload] manual: residual reset via "
                    "destroy_model_parallel() raised "
                    f"{type(exc).__name__}: {exc}"
                )
    return destroyed


def _destroy_official(*, verbose: bool) -> None:
    """Delegate teardown to Megatron's ``parallel_state.destroy_model_parallel``.

    This is the Gate 2.5 fallback. We do not touch ``parallel_state``
    globals ourselves here — the official API owns both the destroy and
    the reset.
    """
    from megatron.core import parallel_state

    if not hasattr(parallel_state, "destroy_model_parallel"):
        # Extremely old Megatron versions might not expose this symbol.
        # In that case there is nothing we can meaningfully do; fall back
        # to the manual path rather than silently succeeding.
        if verbose:
            print(
                "[nccl_offload] official mode requested but "
                "parallel_state.destroy_model_parallel is missing; "
                "falling back to manual destroy."
            )
        _destroy_manual(verbose=verbose)
        return

    try:
        parallel_state.destroy_model_parallel()
        if verbose:
            print("[nccl_offload] official destroy_model_parallel() returned")
    except Exception as exc:  # noqa: BLE001 - surface diagnostics, don't crash
        if verbose:
            print(
                "[nccl_offload] official destroy_model_parallel() raised "
                f"{type(exc).__name__}: {exc}"
            )
        # If the official API raised, try to clean up any state it left
        # behind so the subsequent reload has a clean slate.
        _reset_parallel_state_globals()


def reload_megatron_nccl_groups(
    state_snapshot: dict[str, Any],
    *,
    verbose: bool = False,
) -> None:
    """Re-initialize Megatron parallel state using a snapshot from destroy.

    Idempotent: a second call while parallel state is already initialized is
    a no-op.

    Args:
        state_snapshot: The ``state_snapshot`` returned from
            :func:`destroy_megatron_nccl_groups`.
        verbose: If True, print diagnostics.

    Raises:
        ValueError: If ``state_snapshot`` is empty (i.e. destroy was called
            while parallel state was not initialized). Callers must not
            reload in that case.
    """
    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        if verbose:
            print("[nccl_offload] parallel_state already initialized; nothing to do.")
        return

    if not state_snapshot:
        raise ValueError(
            "Cannot reload Megatron NCCL groups: state_snapshot is empty. "
            "This usually means destroy was called before initialize — there "
            "is nothing to reload."
        )

    init_kwargs = _build_initialize_kwargs_from_snapshot(state_snapshot)

    parallel_state.initialize_model_parallel(**init_kwargs)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if verbose:
        print(f"[nccl_offload] reloaded NCCL groups with config: {init_kwargs}")


# --------------------------------------------------------------------------- #
#  Internal helpers (exposed for unit tests)
# --------------------------------------------------------------------------- #


def _collect_all_nccl_groups() -> list[tuple[str, "torch.distributed.ProcessGroup"]]:
    """Walk ``parallel_state`` globals and return every NCCL ProcessGroup.

    The discovery is intentionally introspection-based rather than hard-coded
    so it adapts to Megatron version changes (new or renamed ``_*_GROUP``
    globals). Results are deduplicated by ``id(pg)`` because multiple
    parallel_state globals can point to the same underlying ProcessGroup
    (e.g. ``_DATA_PARALLEL_GROUP`` and ``_DATA_PARALLEL_GROUP_WITH_CP`` when
    context parallel is disabled).

    Gloo-backed groups are skipped because they hold CPU-side collective
    resources only, not GPU NCCL buffers — they are not the source of the
    VRAM we are trying to reclaim, and destroying them would require a
    separate reload path.
    """
    from megatron.core import parallel_state

    seen_ids: set[int] = set()
    results: list[tuple[str, torch.distributed.ProcessGroup]] = []

    for attr_name in dir(parallel_state):
        if not _looks_like_group_attr(attr_name):
            continue
        try:
            value = getattr(parallel_state, attr_name)
        except Exception:  # noqa: BLE001 - defensive: some descriptors may raise
            continue

        for pg in _iter_process_groups(value):
            if pg is None:
                continue
            if not isinstance(pg, torch.distributed.ProcessGroup):
                continue
            if id(pg) in seen_ids:
                continue
            if _is_gloo(pg):
                continue
            seen_ids.add(id(pg))
            results.append((attr_name, pg))

    return results


def _capture_parallel_state_snapshot() -> dict[str, Any]:
    """Capture just enough parallel-config to drive ``initialize_model_parallel``.

    Must be called while parallel_state is still initialized.
    """
    from megatron.core import parallel_state

    snapshot: dict[str, Any] = {
        "tensor_model_parallel_size": parallel_state.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": parallel_state.get_pipeline_model_parallel_world_size(),
        "context_parallel_size": parallel_state.get_context_parallel_world_size(),
    }

    # Expert parallel is optional — older Megatron versions or non-MoE models
    # may not have the accessor. Default to 1 when absent.
    ep_getter = getattr(
        parallel_state, "get_expert_model_parallel_world_size", None
    )
    if callable(ep_getter):
        try:
            snapshot["expert_model_parallel_size"] = ep_getter()
        except Exception:  # noqa: BLE001 - EP may be uninitialized on non-MoE
            snapshot["expert_model_parallel_size"] = 1
    else:
        snapshot["expert_model_parallel_size"] = 1

    # Virtual pipeline parallel — optional, default unset.
    vpp_getter = getattr(
        parallel_state,
        "get_virtual_pipeline_model_parallel_world_size",
        None,
    )
    if callable(vpp_getter):
        try:
            vpp = vpp_getter()
            if vpp and vpp > 1:
                snapshot["virtual_pipeline_model_parallel_size"] = vpp
        except Exception:  # noqa: BLE001
            pass

    return snapshot


def _reset_parallel_state_globals() -> None:
    """Null out every ProcessGroup held by ``parallel_state`` module globals.

    Without this step, ``parallel_state.model_parallel_is_initialized()``
    would keep returning True after ``destroy_process_group``, and the
    next ``initialize_model_parallel`` would see stale handles.

    We only touch attributes that look like group holders and contain
    ProcessGroup instances; other ``parallel_state`` globals (e.g. the
    global memory buffer) are left untouched.
    """
    from megatron.core import parallel_state

    for attr_name in list(dir(parallel_state)):
        if not _looks_like_group_attr(attr_name):
            continue
        try:
            value = getattr(parallel_state, attr_name)
        except Exception:  # noqa: BLE001
            continue

        if value is None:
            continue

        if isinstance(value, torch.distributed.ProcessGroup):
            try:
                setattr(parallel_state, attr_name, None)
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(value, list):
            if any(isinstance(v, torch.distributed.ProcessGroup) for v in value):
                try:
                    setattr(parallel_state, attr_name, [])
                except Exception:  # noqa: BLE001
                    pass
        elif isinstance(value, tuple):
            if any(isinstance(v, torch.distributed.ProcessGroup) for v in value):
                try:
                    setattr(parallel_state, attr_name, ())
                except Exception:  # noqa: BLE001
                    pass


def _build_initialize_kwargs_from_snapshot(
    state_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Translate a captured snapshot into ``initialize_model_parallel`` kwargs."""
    init_kwargs: dict[str, Any] = {
        "tensor_model_parallel_size": int(
            state_snapshot.get("tensor_model_parallel_size", 1)
        ),
        "pipeline_model_parallel_size": int(
            state_snapshot.get("pipeline_model_parallel_size", 1)
        ),
        "context_parallel_size": int(state_snapshot.get("context_parallel_size", 1)),
        "expert_model_parallel_size": int(
            state_snapshot.get("expert_model_parallel_size", 1)
        ),
    }
    vpp = state_snapshot.get("virtual_pipeline_model_parallel_size")
    if vpp is not None and int(vpp) > 1:
        init_kwargs["virtual_pipeline_model_parallel_size"] = int(vpp)
    return init_kwargs


def _looks_like_group_attr(attr_name: str) -> bool:
    """Heuristic: parallel_state group holders all look like ``_..._GROUP...``."""
    if not attr_name.startswith("_"):
        return False
    return "_GROUP" in attr_name or "_GROUPS" in attr_name


def _iter_process_groups(value: Any):
    """Yield ProcessGroup-like values from a parallel_state attribute.

    Some parallel_state globals are single ``ProcessGroup`` objects; others
    (e.g. hybrid context-parallel layouts) are lists or tuples.
    """
    if isinstance(value, (list, tuple)):
        yield from value
    else:
        yield value


def _is_gloo(pg: "torch.distributed.ProcessGroup") -> bool:
    try:
        backend = torch.distributed.get_backend(pg)
    except Exception:  # noqa: BLE001
        return False
    return str(backend).lower() == "gloo"


def _get_vram_allocated() -> int:
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.memory_allocated())
        except Exception:  # noqa: BLE001
            return 0
    return 0


__all__ = [
    "destroy_megatron_nccl_groups",
    "reload_megatron_nccl_groups",
]
