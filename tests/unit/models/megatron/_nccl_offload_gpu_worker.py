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

"""Multi-process worker that exercises ``nccl_offload`` on real GPUs.

This script is *not* a pytest file. It is invoked via ``torchrun`` by
:mod:`tests.unit.models.megatron.test_nccl_offload_gpu`. Each torchrun
process becomes one Megatron rank; the script runs a small AllReduce,
then drives the destroy/reload cycle the requested number of times and
writes a JSON report at ``--report-path`` (only rank 0 writes).

Gate 2.5 coverage (see ``plans/nemorl-port-plan.md`` Feature 11):
  - ``--mode manual`` exercises the default manual destroy path
  - ``--mode official`` exercises the Gate 2.5 fallback path
  - ``--cycles N`` runs N destroy/reload cycles; the report shows whether
    peak VRAM stays stable across cycles and whether AllReduce continues
    to work after each reload

Usage (run directly, not from pytest):
  torchrun --nproc_per_node=2 \\
      tests/unit/models/megatron/_nccl_offload_gpu_worker.py \\
      --tp 2 --cycles 3 --mode manual --report-path /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_repo_on_sys_path() -> None:
    """Make ``nemo_rl`` importable when the worker is launched as a script.

    ``torchrun`` sets ``sys.path[0]`` to the script's directory (the test
    dir), which does not contain ``nemo_rl``. Walk up until we find a
    directory that does and prepend it.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "nemo_rl").is_dir():
            sys.path.insert(0, str(parent))
            return


_ensure_repo_on_sys_path()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2, help="Tensor-parallel size")
    parser.add_argument("--cycles", type=int, default=3, help="Destroy/reload cycles")
    parser.add_argument(
        "--mode",
        choices=("manual", "official"),
        default="manual",
        help="nccl_offload destroy mode",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="JSON report written by rank 0",
    )
    args = parser.parse_args()

    # Imports deferred until after sys.path fix; importing torch at module
    # top works, but deferring keeps the argparse --help path snappy.
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if args.tp != world_size:
        raise SystemExit(
            f"This worker expects --tp ({args.tp}) == WORLD_SIZE ({world_size}); "
            "launch with matching --nproc_per_node."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    report: dict[str, Any] = {
        "mode": args.mode,
        "world_size": world_size,
        "tp": args.tp,
        "cycles": args.cycles,
        "cycle_stats": [],
        "failure": None,
    }

    try:
        dist.init_process_group(backend="nccl")

        # Only import Megatron/NeMo helpers after the default PG is up,
        # because Megatron-core's init_method looks at the environment.
        from megatron.core import parallel_state  # noqa: F401 (ensures import works)

        from nemo_rl.models.megatron.nccl_offload import (
            destroy_megatron_nccl_groups,
            reload_megatron_nccl_groups,
        )

        initial_kwargs = {
            "tensor_model_parallel_size": args.tp,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        }

        _initialize_model_parallel(initial_kwargs)
        _run_allreduce_sanity(device, world_size)

        baseline_vram = torch.cuda.memory_allocated(device)
        report["baseline_vram_bytes_rank0"] = baseline_vram if rank == 0 else None

        for cycle in range(args.cycles):
            cycle_report: dict[str, Any] = {"cycle": cycle}

            vram_before_destroy = torch.cuda.memory_allocated(device)
            stats = destroy_megatron_nccl_groups(mode=args.mode, verbose=False)
            vram_after_destroy = torch.cuda.memory_allocated(device)

            cycle_report["destroy_stats"] = {
                # numeric keys only; mode echoed to confirm path
                "initialized_before": stats["initialized_before"],
                "num_groups_destroyed": stats["num_groups_destroyed"],
                "vram_freed_bytes_reported": stats["vram_freed_bytes"],
                "vram_before_destroy": vram_before_destroy,
                "vram_after_destroy": vram_after_destroy,
                "mode": stats["mode"],
            }

            # Invariant: after destroy, parallel state must not report
            # itself as initialized. This is a hard contract; if it's
            # broken, the next initialize_model_parallel will likely
            # raise.
            if parallel_state.model_parallel_is_initialized():
                raise RuntimeError(
                    f"cycle {cycle}: parallel_state still reports initialized "
                    f"after destroy (mode={args.mode})"
                )

            reload_megatron_nccl_groups(stats["state_snapshot"], verbose=False)
            vram_after_reload = torch.cuda.memory_allocated(device)
            cycle_report["vram_after_reload"] = vram_after_reload

            # AllReduce sanity after reload: if the NCCL groups weren't
            # rebuilt correctly, this call will hang or raise.
            _run_allreduce_sanity(device, world_size)

            report["cycle_stats"].append(cycle_report)

        # Post-loop invariant: VRAM should return to (roughly) baseline
        # each cycle. We record the final delta; the pytest wrapper
        # applies the tolerance assertion so we keep the worker dumb.
        final_vram = torch.cuda.memory_allocated(device)
        report["final_vram_bytes_rank0"] = final_vram if rank == 0 else None

        # Final teardown so we exit cleanly.
        destroy_megatron_nccl_groups(mode=args.mode, verbose=False)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    except Exception as exc:  # noqa: BLE001 - we want a clean report either way
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    # Only rank 0 writes the report so the pytest wrapper reads a single
    # file. All ranks flush stdout/stderr for torchrun to surface.
    if rank == 0:
        Path(args.report_path).write_text(json.dumps(report, indent=2))

    # Non-zero exit if we failed so torchrun also signals failure.
    if report["failure"] is not None:
        raise SystemExit(1)


def _initialize_model_parallel(kwargs: dict[str, Any]) -> None:
    """Initial bootstrap + a barrier; cycles beyond the first go through
    ``reload_megatron_nccl_groups`` so this helper is only used once."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(**kwargs)
    if dist.is_initialized():
        dist.barrier()


def _run_allreduce_sanity(device, world_size: int) -> None:
    """AllReduce a constant tensor and assert the expected sum.

    Any NCCL breakage (missing group, stale handles) shows up either as
    an exception here or as a wrong sum.
    """
    import torch
    import torch.distributed as dist

    t = torch.ones(1, device=device)
    dist.all_reduce(t)
    expected = float(world_size)
    if abs(t.item() - expected) > 1e-3:
        raise RuntimeError(
            f"AllReduce sanity failed: got {t.item()}, expected {expected}"
        )


if __name__ == "__main__":
    _main()
