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

"""Structural unit tests for the rlix-task2 selective-sync review fixes.

Three independent fixes; each is a CORRECTNESS issue that surfaces only
under TP>1 / multi-receiver topology. Tests here are mock-only — no
Ray, no GPUs — so they run in CI alongside the existing fast suite.

Behavioral verification of the deadlock path (Bug 1) requires real
Ray-actor scheduling under NCCL and is left to upstream multi-GPU
selective-sync integration tests; the lexical guard at the bottom of
this file catches obvious source-level regressions in the meantime.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------
# Bug 3 — vllm_generation receiver pass-throughs MUST dispatch to every
# TP/PP rank (run_rank_0_only_axes=[]). Filtering to TP rank 0 only
# leaves ranks 1..N-1 outside the NCCL collective → deadlock.
# ---------------------------------------------------------------------

# The 6 receiver pass-through methods that participate in the
# selective-sync NCCL collective. Each MUST forward
# run_rank_0_only_axes=[] to run_all_workers_single_data.
_RECEIVER_PASS_THROUGHS = [
    ("setup_collective_group", {
        "model_update_name": "g0",
        "comm_plan": {"g0": {"group_name": "g0", "master_addr": "1.2.3.4",
                              "master_port": 1234, "tgt_devices": []}},
        "mode": "receiver",
        "timeout_s": 10.0,
    }),
    ("update_parameter_in_bucket", {
        "payload": {"param_names": [], "shapes": [], "dtypes": [],
                    "offsets": [], "used_bytes": 0, "cpu_uint8_bucket": b""},
        "ipc_local_ranks": [0],
        "model_update_transport": "cpu_serialize",
    }),
    ("broadcast_parameter", {
        "group_name": "g0",
        "names": [],
        "dtypes": [],
        "shapes": [],
        "broadcast_local_ranks": [0],
    }),
    ("destroy_collective_group", {"group_name": "g0"}),
    ("verify_model", {"expected_stats": {}}),
    ("finalize_weight_update", {}),
]


@pytest.mark.parametrize("method_name,kwargs", _RECEIVER_PASS_THROUGHS,
                         ids=[m for m, _ in _RECEIVER_PASS_THROUGHS])
def test_receiver_passthroughs_dispatch_to_all_ranks(method_name, kwargs):
    """Every receiver pass-through must forward run_rank_0_only_axes=[]."""
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    fake_worker_group = MagicMock()
    fake_worker_group.run_all_workers_single_data.return_value = []

    instance = MagicMock(spec=VllmGeneration)
    instance.worker_group = fake_worker_group
    # Bind the real method to the mock instance so we exercise the actual
    # implementation rather than the auto-mock.
    method = getattr(VllmGeneration, method_name)
    method(instance, **kwargs)

    fake_worker_group.run_all_workers_single_data.assert_called_once()
    _, call_kwargs = fake_worker_group.run_all_workers_single_data.call_args
    assert call_kwargs.get("run_rank_0_only_axes") == [], (
        f"{method_name}: must dispatch to all TP/PP ranks "
        f"(run_rank_0_only_axes=[]); got "
        f"{call_kwargs.get('run_rank_0_only_axes')!r}"
    )


# ---------------------------------------------------------------------
# Bug 2 — vllm_backend.broadcast_parameter must use self.rank (worker-
# local) when comparing against broadcast_local_ranks (also worker-local
# ranks). Falling back to torch.distributed.get_rank() (global rank)
# under TP>1 / multi-node never matches → silent receiver early-return →
# the broadcast collective is never entered on the receiver side.
# ---------------------------------------------------------------------


def _make_vllm_extension(rank=None):
    """Build a minimal stand-in for VllmInternalWorkerExtension that
    exposes the attributes broadcast_parameter touches up to (and
    including) the post-guard ``torch.zeros(..., device=self.device)``
    call. We assert on whether torch.zeros was invoked rather than on
    the deeper ``group.broadcast`` because the function path between the
    two requires more state that's expensive to fake.
    """
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )
    import torch

    inst = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    if rank is not None:
        inst.rank = rank
    # Required by post-guard ``torch.zeros(..., device=self.device)``.
    inst.device = torch.device("cpu")
    # ``broadcast_parameter`` checks ``_model_update_groups``; populate
    # with a stand-in so the rank-comparison branch is reached.
    fake_group = MagicMock()
    fake_group.broadcast = MagicMock()
    inst._model_update_groups = {"g0": fake_group}
    return inst


def _broadcast_args():
    """Minimal args that make ``broadcast_parameter`` reach the rank
    check. ``names``/``dtypes``/``shapes`` empty so the aligned-size
    arithmetic is trivially 0 and no real tensor work is required."""
    return dict(
        group_name="g0",
        names=[],
        dtypes=[],
        shapes=[],
    )


# Sentinel exception raised by ``torch.zeros`` mock — surfaces "the
# function reached the post-guard code path" without requiring us to
# fake the rest of the model state (which sits past torch.zeros and
# would fail with AttributeError otherwise).
class _PastGuardSentinel(RuntimeError):
    pass


def test_broadcast_parameter_uses_self_rank_when_set():
    """When ``self.rank`` is in ``broadcast_local_ranks``, the receiver
    must NOT early-return — proven by observing the sentinel raised
    from the first post-guard ``torch.zeros`` call."""
    from nemo_rl.models.generation.vllm import vllm_backend

    inst = _make_vllm_extension(rank=1)

    # Patch torch.distributed.get_rank to a different value so a buggy
    # implementation (using global rank) would early-return on
    # `42 not in [1]` without raising the sentinel.
    with patch.object(vllm_backend, "torch") as mock_torch:
        mock_torch.distributed.is_initialized.return_value = True
        mock_torch.distributed.get_rank.return_value = 42
        # Allow `torch.empty(0, dtype=...).element_size()` to succeed so
        # the aligned-size loop doesn't crash before torch.zeros.
        mock_torch.empty.return_value.element_size.return_value = 1
        # Raise sentinel from the first post-guard call. If the function
        # early-returned at the rank check, we never get here.
        mock_torch.zeros.side_effect = _PastGuardSentinel("past guard")

        with pytest.raises(_PastGuardSentinel):
            vllm_backend.VllmInternalWorkerExtension.broadcast_parameter(
                inst,
                broadcast_local_ranks=[1],
                **_broadcast_args(),
            )


def test_broadcast_parameter_skips_when_rank_not_in_local_ranks():
    """When ``self.rank`` is NOT in ``broadcast_local_ranks``, the
    receiver early-returns (sentinel never raised)."""
    from nemo_rl.models.generation.vllm import vllm_backend

    inst = _make_vllm_extension(rank=1)

    with patch.object(vllm_backend, "torch") as mock_torch:
        mock_torch.distributed.is_initialized.return_value = True
        mock_torch.distributed.get_rank.return_value = 42
        # If reached, would raise — but we expect early return.
        mock_torch.zeros.side_effect = _PastGuardSentinel("past guard")

        # Should return None cleanly (no exception) — the early return
        # at `if local_rank not in broadcast_local_ranks: return` fires.
        result = vllm_backend.VllmInternalWorkerExtension.broadcast_parameter(
            inst,
            broadcast_local_ranks=[0],  # rank 1 not in here
            **_broadcast_args(),
        )
        assert result is None


def test_broadcast_parameter_falls_back_to_global_rank_when_self_rank_absent():
    """Backward-compat: callers that don't set ``self.rank`` fall
    through to ``torch.distributed.get_rank()`` (the original behavior)."""
    from nemo_rl.models.generation.vllm import vllm_backend

    inst = _make_vllm_extension(rank=None)
    # Don't set inst.rank at all — getattr returns None.

    with patch.object(vllm_backend, "torch") as mock_torch:
        mock_torch.distributed.is_initialized.return_value = True
        # Global rank 0 is in [0] → not skipped → sentinel raised.
        mock_torch.distributed.get_rank.return_value = 0
        mock_torch.empty.return_value.element_size.return_value = 1
        mock_torch.zeros.side_effect = _PastGuardSentinel("past guard")

        with pytest.raises(_PastGuardSentinel):
            vllm_backend.VllmInternalWorkerExtension.broadcast_parameter(
                inst,
                broadcast_local_ranks=[0],
                **_broadcast_args(),
            )


# ---------------------------------------------------------------------
# Bug 1 — sender selective_sync_active_cache must dispatch all
# broadcast_parameter receivers BEFORE entering dist.broadcast(). The
# reverse ordering deadlocks: the sender's Python thread is pinned
# inside the collective and never submits the .remote() calls.
# ---------------------------------------------------------------------
#
# This bug only manifests under real Ray-actor scheduling — a unit test
# of the function in isolation can't reproduce the deadlock because
# `.remote()` and `dist.broadcast()` both return synchronously when
# their dependencies are mocked. The lexical guard below catches
# obvious regressions (someone moving `dist.broadcast()` back above the
# `.remote()` loop in a future refactor); behavioral verification under
# real Ray + NCCL is the responsibility of upstream selective-sync
# integration tests on multi-GPU hardware.


def test_selective_sync_dispatch_ordering_lexical():
    """Lexical guard: in selective_sync_active_cache, the
    broadcast_parameter.remote(...) dispatch loop appears BEFORE the
    sender-side dist.broadcast(...) call within the NCCL-broadcast
    branch.

    Reads the source file directly via Path (no Python import) so the
    test does not pull in megatron / megatron_bridge dependencies that
    may not be present on every CI image.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "nemo_rl" / "models" / "policy" / "workers" / "megatron_policy_worker.py"
    assert src_path.is_file(), f"could not locate {src_path}"
    text = src_path.read_text()

    # Find the NCCL-broadcast branch; assert dispatch loop comes before
    # the dist.broadcast call.
    branch_marker = "if group_name in self._rlix_model_update_groups:"
    branch_idx = text.find(branch_marker)
    assert branch_idx > 0, "could not locate NCCL-broadcast branch marker"

    # Search within the next ~3000 chars (covers the per-bucket loop body).
    region = text[branch_idx : branch_idx + 3000]

    dispatch_idx = region.find(".broadcast_parameter.remote(")
    sender_idx = region.find("dist.broadcast(staging_buf")
    assert dispatch_idx > 0, "dispatch loop not found in NCCL branch"
    assert sender_idx > 0, "sender dist.broadcast not found in NCCL branch"
    assert dispatch_idx < sender_idx, (
        "regression: sender dist.broadcast appears BEFORE receiver "
        ".remote() dispatch — this reintroduces the deadlock the fix "
        "was meant to prevent."
    )
