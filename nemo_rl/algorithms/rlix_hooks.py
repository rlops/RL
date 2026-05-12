"""RLix hook protocol and no-op default for NeMo RL's async_grpo_train.

This module is the seam between NeMo RL (caller) and RLix (implementor).
Import direction:
    nemo_rl/algorithms/grpo.py  →  rlix_hooks.py  (this file, NeMo RL repo)
    rlix/pipeline/nemo_rl_pipeline.py  →  provides NemoRLRLixHooks (real impl)

NeMo RL code only depends on RLixHooksProtocol + NoOpRLixHooks from this file.
It never imports from the rlix package directly, preventing circular dependencies.
"""
from __future__ import annotations

from typing import Any, runtime_checkable

from typing_extensions import Protocol


@runtime_checkable
class RLixHooksProtocol(Protocol):
    """Protocol that async_grpo_train expects from its rlix_hooks argument.

    NeMo RL standalone mode uses NoOpRLixHooks (all methods are no-ops).
    RLix mode passes NemoRLRLixHooks (rlix/pipeline/nemo_rl_pipeline.py),
    which makes blocking Ray RPC calls to the scheduler.
    """

    def before_training(self, step: int) -> None:
        """Called before logprob inference + training; may block on scheduler.

        F5: in RLix mode, blocks until the scheduler grants actor_train GPUs.
        The scheduler shrinks overlap inference workers before returning.
        In standalone mode, this is a no-op.
        """
        ...

    def before_weight_sync(self, step: int) -> None:
        """Called immediately after policy.train() and BEFORE policy.offload_after_refit().

        F4: RLix builds the CPU bucket cache here while the freshly-updated
        Megatron parameters are still resident on GPU. Doing this after
        offload_after_refit() would observe zero-storage tensors (debug #34)
        because that call swaps .data with empty storage to release VRAM.

        In standalone mode, this is a no-op.
        """
        ...

    def after_training(self, step: int) -> int | None:
        """Called after grpo.py's offload_after_refit + destroy_megatron_nccl_groups.

        F5: in RLix mode, drives the post-train half of weight sync — pushes the
        cached weights to active inference workers (coordinator.sync_base_weights_to_active)
        and publishes the new weight version to ATC. The scheduler asynchronously
        calls coordinator.resize_infer (add=overlap_ranks), which routes to
        pipeline._expand_workers() (F6).  In standalone mode, this is a no-op.

        Preconditions (must be satisfied before calling in RLix mode):
            - CPU bucket cache built  (handled by before_weight_sync)
            - Training GPU VRAM offloaded (grpo.py:policy.offload_after_refit)
            - Megatron NCCL groups destroyed (grpo.py:destroy_megatron_nccl_groups)
        """
        ...

    def on_trajectory_collector_created(self, collector: Any) -> None:
        """Register the trajectory collector Ray actor handle with the pipeline.

        F6 dependency: _expand_workers calls collector.set_weight_version after
        each selective sync.  The handle must be registered here before the
        first expand fires, otherwise _expand_workers logs a warning and skips
        the version update.

        Called once, immediately after AsyncTrajectoryCollector is created and
        set_weight_version has been called with the initial value.
        """
        ...

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        """F9: Activate the progress stream for one training step.

        Must be called once before any end_progress_batch calls for that step.
        Passes step_target_trajectories to the scheduler so it can estimate
        remaining demand.

        Args:
            step: Current training step (weight_version).
            count_intended: Total trajectories grpo_train will consume this
                step (num_prompts_per_step).  Must be > 0.
        """
        ...

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        """F9: Report trajectories collected for the current step.

        Called by AsyncTrajectoryCollector after each successful buffer push.
        Increments the local counter and fires a fire-and-forget ProgressReport
        to the coordinator/scheduler at 2% granularity.

        Args:
            step: Target weight version the trajectories were collected for.
            trajectories_collected: Number of trajectories in this push (>= 0).
        """
        ...


class NoOpRLixHooks:
    """Default no-op implementation used in NeMo RL standalone mode.

    Satisfies RLixHooksProtocol so grpo.py always calls hooks.* without
    guarding against None.  In standalone mode these are all no-ops;
    the real implementations live in rlix/pipeline/nemo_rl_pipeline.py.
    """

    def before_training(self, step: int) -> None:
        pass

    def before_weight_sync(self, step: int) -> None:
        pass

    def after_training(self, step: int) -> int | None:
        return None

    def on_trajectory_collector_created(self, collector: Any) -> None:
        pass

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        pass

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        pass
