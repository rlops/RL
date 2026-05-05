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

"""Unit tests for ``nemo_rl.models.megatron.nccl_offload``.

These tests are fully mock-based: they stub ``megatron.core.parallel_state``
and ``torch.distributed`` so the module can be exercised without a GPU or
a real Megatron installation. They verify the control flow of destroy /
reload (collection, deduplication, backend filtering, idempotency,
snapshot round-trip) rather than real VRAM behavior. Real VRAM behavior is
covered by the Gate 2.5 integration tests on GPU.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
#  Fake parallel_state factory
# --------------------------------------------------------------------------- #


class _FakeProcessGroup:
    """Stand-in for ``torch.distributed.ProcessGroup`` in unit tests.

    We register this type as ``torch.distributed.ProcessGroup`` via
    ``isinstance`` patching in the fixture so the module under test accepts
    it as a real ProcessGroup.
    """

    def __init__(self, name: str, backend: str = "nccl"):
        self.name = name
        self.backend = backend

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_FakeProcessGroup({self.name!r}, backend={self.backend!r})"


def _make_fake_parallel_state(
    initialized: bool = True,
    tp: int = 2,
    pp: int = 1,
    cp: int = 1,
    ep: int = 1,
    extra_groups: dict[str, Any] | None = None,
) -> types.ModuleType:
    """Build a fake ``megatron.core.parallel_state`` module for one test."""
    mod = types.ModuleType("megatron.core.parallel_state")
    mod.model_parallel_is_initialized = MagicMock(return_value=initialized)
    mod.get_tensor_model_parallel_world_size = MagicMock(return_value=tp)
    mod.get_pipeline_model_parallel_world_size = MagicMock(return_value=pp)
    mod.get_context_parallel_world_size = MagicMock(return_value=cp)
    mod.get_expert_model_parallel_world_size = MagicMock(return_value=ep)
    mod.initialize_model_parallel = MagicMock()

    # Canonical group holders. Intentionally set up a few non-trivial cases:
    #   - A list-valued holder (hybrid CP layouts sometimes look like this)
    #   - A Gloo group that must be skipped
    #   - A shared ProcessGroup referenced from two holders (dedupe check)
    tp_group = _FakeProcessGroup("tp")
    pp_group = _FakeProcessGroup("pp")
    dp_group = _FakeProcessGroup("dp")
    dp_group_cp_alias = dp_group  # same PG shared across two globals
    cp_group = _FakeProcessGroup("cp")
    gloo_group = _FakeProcessGroup("dp_gloo", backend="gloo")
    ep_group = _FakeProcessGroup("ep")
    hybrid_groups = [_FakeProcessGroup("hybrid_a"), _FakeProcessGroup("hybrid_b")]

    mod._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    mod._PIPELINE_MODEL_PARALLEL_GROUP = pp_group
    mod._DATA_PARALLEL_GROUP = dp_group
    mod._DATA_PARALLEL_GROUP_WITH_CP = dp_group_cp_alias
    mod._DATA_PARALLEL_GROUP_GLOO = gloo_group
    mod._CONTEXT_PARALLEL_GROUP = cp_group
    mod._EXPERT_MODEL_PARALLEL_GROUP = ep_group
    mod._HYBRID_DP_CP_GROUPS = hybrid_groups

    # Non-group globals that must be left alone by the reset helper.
    mod._GLOBAL_MEMORY_BUFFER = object()  # deliberately opaque

    if extra_groups:
        for name, val in extra_groups.items():
            setattr(mod, name, val)

    return mod


# --------------------------------------------------------------------------- #
#  Common fixture: install fake parallel_state + friendly torch.distributed
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_mcore(monkeypatch):
    """Install a fake ``megatron.core.parallel_state`` and return helpers.

    The fixture also patches ``torch.distributed.ProcessGroup`` to be the
    local ``_FakeProcessGroup`` class so ``isinstance`` checks inside the
    module under test accept our fakes.
    """
    fake_state = _make_fake_parallel_state()

    # Build a parent megatron.core module if one is not already present.
    mcore_parent = types.ModuleType("megatron.core")
    mcore_parent.parallel_state = fake_state
    # Register under sys.modules so ``from megatron.core import parallel_state``
    # inside the module under test resolves to our fake.
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", mcore_parent)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", fake_state)

    import torch

    # Swap the ProcessGroup type used by the module under test. Use
    # monkeypatch so it is restored cleanly at test teardown.
    monkeypatch.setattr(torch.distributed, "ProcessGroup", _FakeProcessGroup)

    destroyed: list[_FakeProcessGroup] = []

    def fake_destroy(pg):
        destroyed.append(pg)

    def fake_get_backend(pg):
        return pg.backend

    monkeypatch.setattr(torch.distributed, "destroy_process_group", fake_destroy)
    monkeypatch.setattr(torch.distributed, "get_backend", fake_get_backend)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", MagicMock())

    # The fake ``destroy_model_parallel`` simulates what the real Megatron
    # API is expected to do: drop the NCCL globals so
    # ``model_parallel_is_initialized`` returns False and a subsequent
    # ``initialize_model_parallel`` can run. Individual group destruction
    # is not simulated because the official API does not expose that
    # granularity (we only observe its effect on ``destroyed`` when
    # ``mode="manual"``).
    def fake_destroy_model_parallel():
        fake_state._TENSOR_MODEL_PARALLEL_GROUP = None
        fake_state._PIPELINE_MODEL_PARALLEL_GROUP = None
        fake_state._DATA_PARALLEL_GROUP = None
        fake_state._DATA_PARALLEL_GROUP_WITH_CP = None
        fake_state._DATA_PARALLEL_GROUP_GLOO = None
        fake_state._CONTEXT_PARALLEL_GROUP = None
        fake_state._EXPERT_MODEL_PARALLEL_GROUP = None
        fake_state._HYBRID_DP_CP_GROUPS = []
        fake_state.model_parallel_is_initialized.return_value = False

    fake_state.destroy_model_parallel = MagicMock(side_effect=fake_destroy_model_parallel)

    return {
        "parallel_state": fake_state,
        "destroyed": destroyed,
    }


# --------------------------------------------------------------------------- #
#  _collect_all_nccl_groups
# --------------------------------------------------------------------------- #


class TestCollectAllNcclGroups:
    def test_collects_all_nccl_groups(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _collect_all_nccl_groups

        collected = _collect_all_nccl_groups()
        names = {pg.name for _attr, pg in collected}

        assert "tp" in names
        assert "pp" in names
        assert "dp" in names
        assert "cp" in names
        assert "ep" in names
        assert "hybrid_a" in names
        assert "hybrid_b" in names

    def test_skips_gloo_groups(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _collect_all_nccl_groups

        collected = _collect_all_nccl_groups()
        names = {pg.name for _attr, pg in collected}

        assert "dp_gloo" not in names

    def test_dedupes_shared_process_group(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _collect_all_nccl_groups

        collected = _collect_all_nccl_groups()
        dp_entries = [pg for _attr, pg in collected if pg.name == "dp"]

        assert len(dp_entries) == 1, (
            "ProcessGroup shared across two parallel_state globals should be "
            "collected only once"
        )

    def test_ignores_non_process_group_globals(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _collect_all_nccl_groups

        collected = _collect_all_nccl_groups()
        # _GLOBAL_MEMORY_BUFFER is opaque and must not appear.
        for _attr, pg in collected:
            assert isinstance(pg, _FakeProcessGroup)

    def test_empty_when_no_groups(self, fake_mcore, monkeypatch):
        from nemo_rl.models.megatron.nccl_offload import _collect_all_nccl_groups

        fake = fake_mcore["parallel_state"]
        for attr in list(dir(fake)):
            if attr.startswith("_") and ("_GROUP" in attr or "_GROUPS" in attr):
                setattr(fake, attr, None)

        collected = _collect_all_nccl_groups()
        assert collected == []


# --------------------------------------------------------------------------- #
#  _capture_parallel_state_snapshot
# --------------------------------------------------------------------------- #


class TestCaptureSnapshot:
    def test_snapshot_contains_expected_keys(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            _capture_parallel_state_snapshot,
        )

        snap = _capture_parallel_state_snapshot()

        assert snap["tensor_model_parallel_size"] == 2
        assert snap["pipeline_model_parallel_size"] == 1
        assert snap["context_parallel_size"] == 1
        assert snap["expert_model_parallel_size"] == 1

    def test_snapshot_defaults_ep_when_accessor_missing(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            _capture_parallel_state_snapshot,
        )

        del fake_mcore["parallel_state"].get_expert_model_parallel_world_size

        snap = _capture_parallel_state_snapshot()
        assert snap["expert_model_parallel_size"] == 1

    def test_snapshot_captures_vpp_when_greater_than_one(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            _capture_parallel_state_snapshot,
        )

        fake_mcore["parallel_state"].get_virtual_pipeline_model_parallel_world_size = (
            MagicMock(return_value=4)
        )

        snap = _capture_parallel_state_snapshot()
        assert snap["virtual_pipeline_model_parallel_size"] == 4

    def test_snapshot_omits_vpp_when_one(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            _capture_parallel_state_snapshot,
        )

        fake_mcore["parallel_state"].get_virtual_pipeline_model_parallel_world_size = (
            MagicMock(return_value=1)
        )

        snap = _capture_parallel_state_snapshot()
        assert "virtual_pipeline_model_parallel_size" not in snap


class TestBuildInitializeKwargsFromSnapshot:
    def test_basic_snapshot_to_kwargs(self):
        from nemo_rl.models.megatron.nccl_offload import (
            _build_initialize_kwargs_from_snapshot,
        )

        kwargs = _build_initialize_kwargs_from_snapshot(
            {
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 2,
                "context_parallel_size": 2,
                "expert_model_parallel_size": 8,
            }
        )

        assert kwargs == {
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 8,
        }

    def test_missing_keys_default_to_one(self):
        from nemo_rl.models.megatron.nccl_offload import (
            _build_initialize_kwargs_from_snapshot,
        )

        kwargs = _build_initialize_kwargs_from_snapshot({})

        assert kwargs["tensor_model_parallel_size"] == 1
        assert kwargs["pipeline_model_parallel_size"] == 1
        assert kwargs["context_parallel_size"] == 1
        assert kwargs["expert_model_parallel_size"] == 1
        assert "virtual_pipeline_model_parallel_size" not in kwargs

    def test_vpp_included_when_greater_than_one(self):
        from nemo_rl.models.megatron.nccl_offload import (
            _build_initialize_kwargs_from_snapshot,
        )

        kwargs = _build_initialize_kwargs_from_snapshot(
            {"virtual_pipeline_model_parallel_size": 4}
        )

        assert kwargs["virtual_pipeline_model_parallel_size"] == 4

    def test_vpp_omitted_when_one(self):
        from nemo_rl.models.megatron.nccl_offload import (
            _build_initialize_kwargs_from_snapshot,
        )

        kwargs = _build_initialize_kwargs_from_snapshot(
            {"virtual_pipeline_model_parallel_size": 1}
        )

        assert "virtual_pipeline_model_parallel_size" not in kwargs


# --------------------------------------------------------------------------- #
#  _reset_parallel_state_globals
# --------------------------------------------------------------------------- #


class TestResetGlobals:
    def test_sets_single_group_globals_to_none(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _reset_parallel_state_globals

        fake = fake_mcore["parallel_state"]
        _reset_parallel_state_globals()

        assert fake._TENSOR_MODEL_PARALLEL_GROUP is None
        assert fake._PIPELINE_MODEL_PARALLEL_GROUP is None
        assert fake._DATA_PARALLEL_GROUP is None
        assert fake._CONTEXT_PARALLEL_GROUP is None
        assert fake._EXPERT_MODEL_PARALLEL_GROUP is None

    def test_clears_list_valued_group_globals(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _reset_parallel_state_globals

        fake = fake_mcore["parallel_state"]
        _reset_parallel_state_globals()

        assert fake._HYBRID_DP_CP_GROUPS == []

    def test_leaves_non_group_globals_untouched(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import _reset_parallel_state_globals

        fake = fake_mcore["parallel_state"]
        sentinel = fake._GLOBAL_MEMORY_BUFFER

        _reset_parallel_state_globals()

        assert fake._GLOBAL_MEMORY_BUFFER is sentinel


# --------------------------------------------------------------------------- #
#  destroy_megatron_nccl_groups (public API)
# --------------------------------------------------------------------------- #


class TestDestroyMegatronNcclGroups:
    def test_noop_when_not_initialized(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        fake_mcore["parallel_state"].model_parallel_is_initialized.return_value = False

        stats = destroy_megatron_nccl_groups()

        assert stats["initialized_before"] is False
        assert stats["num_groups_destroyed"] == 0
        assert stats["state_snapshot"] == {}
        assert fake_mcore["destroyed"] == []

    def test_destroys_all_nccl_groups_and_skips_gloo(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups()

        destroyed_names = [pg.name for pg in fake_mcore["destroyed"]]
        assert "tp" in destroyed_names
        assert "pp" in destroyed_names
        assert "dp" in destroyed_names
        assert "cp" in destroyed_names
        assert "ep" in destroyed_names
        assert "dp_gloo" not in destroyed_names
        # hybrid groups are list-valued but still NCCL — must be destroyed
        assert "hybrid_a" in destroyed_names
        assert "hybrid_b" in destroyed_names

        assert stats["initialized_before"] is True
        assert stats["num_groups_destroyed"] == len(destroyed_names)

    def test_state_snapshot_is_populated(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups()
        snap = stats["state_snapshot"]

        assert snap["tensor_model_parallel_size"] == 2
        assert snap["pipeline_model_parallel_size"] == 1
        assert snap["context_parallel_size"] == 1
        assert snap["expert_model_parallel_size"] == 1

    def test_stats_shape_contract(self, fake_mcore):
        """Verify destroy() always returns the documented fields with correct types."""
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups()

        assert set(stats.keys()) >= {
            "initialized_before",
            "num_groups_destroyed",
            "vram_freed_bytes",
            "state_snapshot",
            "mode",
        }
        assert isinstance(stats["initialized_before"], bool)
        assert isinstance(stats["num_groups_destroyed"], int)
        assert isinstance(stats["vram_freed_bytes"], int)
        assert stats["vram_freed_bytes"] >= 0
        assert isinstance(stats["state_snapshot"], dict)
        assert stats["mode"] in ("manual", "official")

    def test_parallel_state_globals_cleared(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        destroy_megatron_nccl_groups()
        fake = fake_mcore["parallel_state"]

        assert fake._TENSOR_MODEL_PARALLEL_GROUP is None
        assert fake._HYBRID_DP_CP_GROUPS == []

    def test_is_idempotent(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        first = destroy_megatron_nccl_groups()
        # Second call: simulate that initialize_* accessor now reports
        # uninitialized (consistent with cleared globals).
        fake_mcore["parallel_state"].model_parallel_is_initialized.return_value = False
        destroyed_count_after_first = len(fake_mcore["destroyed"])
        second = destroy_megatron_nccl_groups()

        assert first["initialized_before"] is True
        assert second["initialized_before"] is False
        assert second["num_groups_destroyed"] == 0
        assert len(fake_mcore["destroyed"]) == destroyed_count_after_first

    def test_per_group_destroy_failure_does_not_abort(self, fake_mcore, monkeypatch):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        import torch

        destroyed: list[_FakeProcessGroup] = []

        def flaky_destroy(pg):
            if pg.name == "pp":
                raise RuntimeError("simulated destroy failure")
            destroyed.append(pg)

        monkeypatch.setattr(torch.distributed, "destroy_process_group", flaky_destroy)

        stats = destroy_megatron_nccl_groups()

        destroyed_names = [pg.name for pg in destroyed]
        # pp failed but other groups still destroyed
        assert "tp" in destroyed_names
        assert "dp" in destroyed_names
        assert "pp" not in destroyed_names
        # Counter only counts successes
        assert stats["num_groups_destroyed"] == len(destroyed_names)


# --------------------------------------------------------------------------- #
#  reload_megatron_nccl_groups (public API)
# --------------------------------------------------------------------------- #


class TestReloadMegatronNcclGroups:
    def test_calls_initialize_with_snapshot_values(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            destroy_megatron_nccl_groups,
            reload_megatron_nccl_groups,
        )

        stats = destroy_megatron_nccl_groups()
        snapshot = stats["state_snapshot"]

        # Simulate state being torn down.
        fake_mcore["parallel_state"].model_parallel_is_initialized.return_value = False

        reload_megatron_nccl_groups(snapshot)

        init = fake_mcore["parallel_state"].initialize_model_parallel
        init.assert_called_once()
        _args, kwargs = init.call_args
        assert kwargs["tensor_model_parallel_size"] == 2
        assert kwargs["pipeline_model_parallel_size"] == 1
        assert kwargs["context_parallel_size"] == 1
        assert kwargs["expert_model_parallel_size"] == 1

    def test_noop_when_already_initialized(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            reload_megatron_nccl_groups,
        )

        reload_megatron_nccl_groups({"tensor_model_parallel_size": 2})

        fake_mcore["parallel_state"].initialize_model_parallel.assert_not_called()

    def test_raises_when_snapshot_empty(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            reload_megatron_nccl_groups,
        )

        fake_mcore["parallel_state"].model_parallel_is_initialized.return_value = False

        with pytest.raises(ValueError, match="state_snapshot is empty"):
            reload_megatron_nccl_groups({})

    def test_destroy_reload_roundtrip_uses_captured_config(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import (
            destroy_megatron_nccl_groups,
            reload_megatron_nccl_groups,
        )

        fake = fake_mcore["parallel_state"]
        fake.get_tensor_model_parallel_world_size.return_value = 4
        fake.get_pipeline_model_parallel_world_size.return_value = 2
        fake.get_context_parallel_world_size.return_value = 2
        fake.get_expert_model_parallel_world_size.return_value = 2

        stats = destroy_megatron_nccl_groups()
        fake.model_parallel_is_initialized.return_value = False

        reload_megatron_nccl_groups(stats["state_snapshot"])

        _args, kwargs = fake.initialize_model_parallel.call_args
        assert kwargs["tensor_model_parallel_size"] == 4
        assert kwargs["pipeline_model_parallel_size"] == 2
        assert kwargs["context_parallel_size"] == 2
        assert kwargs["expert_model_parallel_size"] == 2


# --------------------------------------------------------------------------- #
#  destroy with mode="official" (Gate 2.5 fallback path)
# --------------------------------------------------------------------------- #


class TestDestroyOfficialMode:
    def test_official_mode_delegates_to_destroy_model_parallel(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups(mode="official")

        fake_mcore["parallel_state"].destroy_model_parallel.assert_called_once_with()
        # Official path does not use our manual destroy, so the fake
        # destroy_process_group counter stays empty.
        assert fake_mcore["destroyed"] == []
        # Sentinel: official path returns -1 for destroyed count.
        assert stats["num_groups_destroyed"] == -1
        assert stats["mode"] == "official"

    def test_official_mode_still_captures_snapshot(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups(mode="official")

        assert stats["state_snapshot"]["tensor_model_parallel_size"] == 2
        assert stats["state_snapshot"]["pipeline_model_parallel_size"] == 1

    def test_official_mode_noop_when_not_initialized(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        fake_mcore["parallel_state"].model_parallel_is_initialized.return_value = False

        stats = destroy_megatron_nccl_groups(mode="official")

        fake_mcore["parallel_state"].destroy_model_parallel.assert_not_called()
        assert stats["initialized_before"] is False
        assert stats["num_groups_destroyed"] == 0  # Not -1: we never reached destroy
        assert stats["mode"] == "official"

    def test_official_mode_falls_back_to_manual_when_api_missing(
        self, fake_mcore
    ):
        """If Megatron somehow doesn't expose destroy_model_parallel, we
        quietly fall through to the manual path rather than no-oping."""
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        del fake_mcore["parallel_state"].destroy_model_parallel

        stats = destroy_megatron_nccl_groups(mode="official")

        # Manual destroy executed instead — we see PGs destroyed
        assert len(fake_mcore["destroyed"]) > 0

    def test_official_mode_tolerates_api_raising(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        fake_mcore["parallel_state"].destroy_model_parallel.side_effect = RuntimeError(
            "simulated"
        )

        # Must not raise
        stats = destroy_megatron_nccl_groups(mode="official")

        assert stats["mode"] == "official"
        # We still defensively reset globals even when the API raised.
        assert fake_mcore["parallel_state"]._TENSOR_MODEL_PARALLEL_GROUP is None

    def test_reload_after_official_destroy_works(self, fake_mcore):
        """End-to-end contract: official destroy + reload is a valid cycle."""
        from nemo_rl.models.megatron.nccl_offload import (
            destroy_megatron_nccl_groups,
            reload_megatron_nccl_groups,
        )

        stats = destroy_megatron_nccl_groups(mode="official")
        reload_megatron_nccl_groups(stats["state_snapshot"])

        fake_mcore["parallel_state"].initialize_model_parallel.assert_called_once()

    def test_invalid_mode_raises(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        with pytest.raises(ValueError, match="Invalid mode"):
            destroy_megatron_nccl_groups(mode="hybrid")  # type: ignore[arg-type]

    def test_default_mode_is_manual(self, fake_mcore):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        stats = destroy_megatron_nccl_groups()

        assert stats["mode"] == "manual"
        # Manual path was used — PGs were destroyed individually.
        assert len(fake_mcore["destroyed"]) > 0


# --------------------------------------------------------------------------- #
#  Manual mode residual-state reset contract
# --------------------------------------------------------------------------- #


class TestManualModeResidualReset:
    """Manual mode must delegate non-group state reset to Megatron.

    The manual path explicitly destroys NCCL ProcessGroups (its unique
    value-add over official mode) and nulls the group-shaped parallel_state
    globals. But Megatron also tracks *non-group* module state
    (``_GLOBAL_MEMORY_BUFFER``, ``_MPU_*_WORLD_SIZE``,
    ``_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK``, etc.) that does not match
    the ``_*_GROUP*`` naming convention. Leaving those set causes the next
    ``initialize_model_parallel`` to raise. To stay forward-compatible
    with Megatron versions that add new non-group globals, the manual
    path delegates that residual state reset to
    ``parallel_state.destroy_model_parallel()``. These tests lock that
    contract so regressions are caught without needing a GPU.
    """

    def test_manual_mode_invokes_destroy_model_parallel_for_residual_reset(
        self, fake_mcore
    ):
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        destroy_megatron_nccl_groups(mode="manual")

        fake_mcore["parallel_state"].destroy_model_parallel.assert_called_once_with()

    def test_manual_mode_destroys_groups_before_residual_reset(self, fake_mcore):
        """Per-group destroy must run *before* the residual reset.

        If we call destroy_model_parallel first, it nulls the group refs
        and we would lose our chance to invoke destroy_process_group on
        the underlying NCCL communicators.
        """
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        events: list[str] = []

        import torch

        original_destroy_pg = torch.distributed.destroy_process_group

        def tracking_destroy(pg):
            events.append(f"destroy_process_group:{pg.name}")
            original_destroy_pg(pg)

        torch.distributed.destroy_process_group = tracking_destroy

        fake_mcore["parallel_state"].destroy_model_parallel.side_effect = (
            lambda: events.append("destroy_model_parallel")
        )

        try:
            destroy_megatron_nccl_groups(mode="manual")
        finally:
            torch.distributed.destroy_process_group = original_destroy_pg

        per_group_events = [e for e in events if e.startswith("destroy_process_group:")]
        assert per_group_events, "expected at least one destroy_process_group call"
        assert "destroy_model_parallel" in events
        last_per_group_idx = max(
            events.index(e) for e in per_group_events
        )
        dmp_idx = events.index("destroy_model_parallel")
        assert last_per_group_idx < dmp_idx, (
            "destroy_model_parallel must run AFTER the last per-group destroy; "
            f"events were: {events}"
        )

    def test_manual_mode_tolerates_destroy_model_parallel_raising(self, fake_mcore):
        """A raising residual-reset must not mask the successful per-group destroys.

        If Megatron changes destroy_model_parallel semantics in a future
        version, we want to keep working: the destroy call still returns
        a usable stats dict and the per-group destroys already succeeded.
        """
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        fake_mcore["parallel_state"].destroy_model_parallel.side_effect = RuntimeError(
            "simulated residual-reset failure"
        )

        # Must not raise.
        stats = destroy_megatron_nccl_groups(mode="manual")

        assert stats["mode"] == "manual"
        # Per-group destroys still happened — NCCL communicators are freed.
        assert len(fake_mcore["destroyed"]) > 0

    def test_manual_mode_skips_residual_reset_when_api_missing(self, fake_mcore):
        """If parallel_state has no destroy_model_parallel at all, the
        manual path must still complete (it just can't delegate)."""
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        del fake_mcore["parallel_state"].destroy_model_parallel

        stats = destroy_megatron_nccl_groups(mode="manual")

        assert stats["mode"] == "manual"
        assert len(fake_mcore["destroyed"]) > 0
