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

"""GPU integration tests for ``nemo_rl.models.megatron.nccl_offload``.

These tests spawn ``torchrun`` as a subprocess to drive a real multi-process
Megatron NCCL lifecycle. They cover Gate 2.5 of the NeMo RL port plan (see
``plans/nemorl-port-plan.md`` Feature 11) *for the NCCL piece only* — the
full Gate 2.5 also involves Feature 4 (CPU bucket cache) which is TASK 2.

Requirements:
  - 2+ CUDA GPUs visible to the process
  - ``torchrun`` on PATH
  - ``megatron.core`` installed in the current environment

These tests are deliberately skipped when any of the above are missing so
they are safe to include in MacBook-local pytest runs (where they simply
skip). The worker script that actually runs under torchrun is
``_nccl_offload_gpu_worker.py`` in this same directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

_WORKER_SCRIPT = Path(__file__).parent / "_nccl_offload_gpu_worker.py"

# VRAM tolerance across cycles: communicator buffers can vary by a small
# amount from cycle to cycle because allocators reuse pages differently;
# anything within ~200 MiB is considered "stable". This is tight enough
# to catch real leaks (each cycle would leak at least a communicator
# buffer, which is 100s of MiB to 1 GiB) but loose enough to avoid
# flakiness.
_VRAM_STABILITY_TOLERANCE_BYTES = 200 * 1024 * 1024


def _cuda_device_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def _megatron_available() -> bool:
    try:
        import megatron.core  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


_REQUIRES_GPU = pytest.mark.skipif(
    _cuda_device_count() < 2,
    reason="nccl_offload GPU integration tests require >= 2 CUDA GPUs",
)
_REQUIRES_TORCHRUN = pytest.mark.skipif(
    shutil.which("torchrun") is None,
    reason="nccl_offload GPU integration tests require torchrun on PATH",
)
_REQUIRES_MEGATRON = pytest.mark.skipif(
    not _megatron_available(),
    reason="nccl_offload GPU integration tests require megatron.core",
)


def _resolve_torchrun() -> str:
    """Find a ``torchrun`` whose Python actually has ``megatron.core``.

    NeMo-RL installs Megatron into a per-worker uv venv (e.g.
    ``/opt/ray_venvs/...MegatronPolicyWorker/``) separate from the default
    interpreter. A plain ``shutil.which("torchrun")`` will typically return
    the main-venv torchrun, whose Python does not have ``megatron.core``
    and which therefore fails the worker import immediately.

    Resolution order:
      1. ``NCCL_OFFLOAD_TORCHRUN`` env var (explicit override).
      2. ``NCCL_OFFLOAD_MCORE_VENV`` env var pointing at a venv root
         (append ``bin/torchrun``).
      3. Autodiscover any ``/opt/ray_venvs/*MegatronPolicyWorker*/bin/torchrun``.
      4. Fall back to whatever ``shutil.which("torchrun")`` returns.
    """
    override = os.environ.get("NCCL_OFFLOAD_TORCHRUN")
    if override and Path(override).exists():
        return override

    venv_root = os.environ.get("NCCL_OFFLOAD_MCORE_VENV")
    if venv_root:
        candidate = Path(venv_root) / "bin" / "torchrun"
        if candidate.exists():
            return str(candidate)

    ray_venvs = Path("/opt/ray_venvs")
    if ray_venvs.is_dir():
        for entry in sorted(ray_venvs.iterdir()):
            if "MegatronPolicyWorker" in entry.name:
                candidate = entry / "bin" / "torchrun"
                if candidate.exists():
                    return str(candidate)

    fallback = shutil.which("torchrun")
    if fallback is None:
        # _REQUIRES_TORCHRUN should have skipped us; defensive raise for
        # callers that invoke _resolve_torchrun directly.
        raise RuntimeError("torchrun not found on PATH")
    return fallback


def _format_worker_failure(report_path: Path) -> str:
    """Read the worker's rank-0 JSON report and format any failure nicely.

    Returns a human-readable string. Returns an empty string if no report
    was written (worker died before writing) or the report has no failure.
    """
    if not report_path.exists():
        return ""
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:  # noqa: BLE001 - malformed report is diagnostic too
        return f"(worker wrote report but it was not valid JSON: {exc})"

    failure = report.get("failure")
    if failure is None:
        return ""

    return (
        f"Worker failure: {failure.get('type', '<unknown>')}: "
        f"{failure.get('message', '')}\n"
        f"Worker traceback:\n{failure.get('traceback', '<none>')}"
    )


def _run_worker(
    *,
    tp: int,
    cycles: int,
    mode: str,
    timeout_s: int = 180,
) -> dict[str, Any]:
    """Launch the worker via torchrun and return its rank-0 JSON report.

    On non-zero exit, surfaces the worker's ``failure.traceback`` (written
    to the JSON report by ``_nccl_offload_gpu_worker.py``) into the
    assertion message so the cause is visible in pytest output instead of
    being hidden behind torchrun's generic ``ChildFailedError``.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as report_file:
        report_path = Path(report_file.name)

    try:
        cmd = [
            _resolve_torchrun(),
            f"--nproc_per_node={tp}",
            # Force a single-node rendezvous on loopback; --standalone is
            # the simplest way to avoid needing a master addr/port when
            # running on a single host.
            "--standalone",
            str(_WORKER_SCRIPT),
            "--tp",
            str(tp),
            "--cycles",
            str(cycles),
            "--mode",
            mode,
            "--report-path",
            str(report_path),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

        if proc.returncode != 0:
            worker_failure = _format_worker_failure(report_path)
            raise AssertionError(
                f"torchrun worker failed (mode={mode}, tp={tp}, cycles={cycles}).\n"
                f"Exit code: {proc.returncode}\n"
                f"{worker_failure}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        if not report_path.exists():
            raise AssertionError(
                "Worker exited cleanly but no report was written. "
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        return json.loads(report_path.read_text())

    finally:
        try:
            report_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _assert_cycle_invariants(
    report: dict[str, Any],
    *,
    expected_mode: str,
    expected_cycles: int,
) -> None:
    assert report["failure"] is None, f"Worker reported failure: {report['failure']}"
    assert report["mode"] == expected_mode
    assert len(report["cycle_stats"]) == expected_cycles

    for cycle_stat in report["cycle_stats"]:
        ds = cycle_stat["destroy_stats"]

        # parallel_state must have been initialized at destroy time
        assert ds["initialized_before"], (
            f"cycle {cycle_stat['cycle']}: parallel_state was not initialized"
        )

        # Destroy should have actually reduced VRAM at least a bit; a
        # strict positive delta is too flaky because of allocator caching,
        # but vram_after_destroy must not be above vram_before.
        assert ds["vram_after_destroy"] <= ds["vram_before_destroy"], (
            f"cycle {cycle_stat['cycle']}: VRAM grew across destroy "
            f"(before={ds['vram_before_destroy']}, after={ds['vram_after_destroy']})"
        )


# --------------------------------------------------------------------------- #
#  Gate 2.5: manual mode
# --------------------------------------------------------------------------- #


@pytest.mark.mcore
@_REQUIRES_GPU
@_REQUIRES_TORCHRUN
@_REQUIRES_MEGATRON
class TestNcclOffloadGpuManual:
    def test_single_cycle_tp2(self):
        """Smoke test: one destroy/reload cycle at tp=2, manual mode."""
        report = _run_worker(tp=2, cycles=1, mode="manual")
        _assert_cycle_invariants(report, expected_mode="manual", expected_cycles=1)

    def test_gate_2_5_three_cycles_tp2(self):
        """Gate 2.5 (NCCL portion only): 3 destroy/reload cycles stable at tp=2."""
        cycles = 3
        report = _run_worker(tp=2, cycles=cycles, mode="manual")
        _assert_cycle_invariants(
            report, expected_mode="manual", expected_cycles=cycles
        )

        # VRAM stability across cycles: the post-reload VRAM should
        # return to a similar baseline each cycle. We compare every
        # cycle's post-reload VRAM against the first cycle's.
        post_reload = [c["vram_after_reload"] for c in report["cycle_stats"]]
        baseline = post_reload[0]
        for i, v in enumerate(post_reload):
            drift = abs(v - baseline)
            assert drift <= _VRAM_STABILITY_TOLERANCE_BYTES, (
                f"VRAM drift beyond tolerance on cycle {i}: "
                f"baseline={baseline}, observed={v}, drift={drift}, "
                f"tolerance={_VRAM_STABILITY_TOLERANCE_BYTES}"
            )


# --------------------------------------------------------------------------- #
#  Gate 2.5: official mode (fallback path)
# --------------------------------------------------------------------------- #


@pytest.mark.mcore
@_REQUIRES_GPU
@_REQUIRES_TORCHRUN
@_REQUIRES_MEGATRON
class TestNcclOffloadGpuOfficial:
    def test_single_cycle_tp2(self):
        """Smoke test: one destroy/reload cycle at tp=2 via the Megatron
        official destroy_model_parallel() fallback."""
        report = _run_worker(tp=2, cycles=1, mode="official")
        _assert_cycle_invariants(report, expected_mode="official", expected_cycles=1)

    def test_three_cycles_tp2(self):
        """Fallback path stability: 3 destroy/reload cycles at tp=2.

        If the primary (manual) path ever fails Gate 2.5, this test
        guards the fallback path that operators will switch to."""
        cycles = 3
        report = _run_worker(tp=2, cycles=cycles, mode="official")
        _assert_cycle_invariants(
            report, expected_mode="official", expected_cycles=cycles
        )
        # Same stability check as the manual path.
        post_reload = [c["vram_after_reload"] for c in report["cycle_stats"]]
        baseline = post_reload[0]
        for i, v in enumerate(post_reload):
            drift = abs(v - baseline)
            assert drift <= _VRAM_STABILITY_TOLERANCE_BYTES, (
                f"VRAM drift beyond tolerance on cycle {i} (official mode): "
                f"baseline={baseline}, observed={v}, drift={drift}, "
                f"tolerance={_VRAM_STABILITY_TOLERANCE_BYTES}"
            )
