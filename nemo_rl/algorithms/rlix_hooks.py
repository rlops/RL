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

    def after_training(self, step: int) -> int | None:
        """Called after policy.train() completes; notifies scheduler to expand.

        F5: in RLix mode, notifies the scheduler that actor_train GPUs are
        released.  The scheduler asynchronously calls coordinator.resize_infer
        (add=overlap_ranks), which routes to pipeline._expand_workers() (F6).
        Weight sync and version update happen inside _expand_workers.
        In standalone mode, this is a no-op.

        Preconditions (must be satisfied before calling in RLix mode):
            - CPU bucket cache built (TODO F4: policy.build_cpu_bucket_cache)
            - Training GPU VRAM offloaded (TODO F11: policy.offload_training_gpu)
            - Megatron NCCL groups destroyed (TODO F11: destroy_nccl_groups)
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


class NoOpRLixHooks:
    """Default no-op implementation used in NeMo RL standalone mode.

    Satisfies RLixHooksProtocol so grpo.py always calls hooks.* without
    guarding against None.  In standalone mode these are all no-ops;
    the real implementations live in rlix/pipeline/nemo_rl_pipeline.py.
    """

    def before_training(self, step: int) -> None:
        pass

    def after_training(self, step: int) -> int | None:
        return None

    def on_trajectory_collector_created(self, collector: Any) -> None:
        pass
