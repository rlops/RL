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
import gc
import os
import re
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge.training.checkpointing import (
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.config import MegatronGenerationConfig
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_obj_from_pp_rank,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    TopkLogitsPostProcessor,
    aggregate_training_statistics,
    megatron_forward_backward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# This is useful when using worker extension classes.
class MegatronPolicyWorkerImpl(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        **kwargs: Any,
    ):
        """Initialize the MegatronPolicyWorker."""
        # Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files
        apply_transformer_engine_patch()

        self.cfg = config

        # Set rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()

        # Step 1: Setup distributed
        setup_distributed()

        # Step 2: Validate and setup model paths
        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            config
        )
        # Handle model import if needed
        handle_model_import(
            config, hf_model_name, pretrained_path, pt_checkpoint_exists
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        runtime_config = validate_and_set_config(
            config,
            self.rank,
            hf_model_name,
            pretrained_path,
            weights_path,
            optimizer_path,
        )

        self.megatron_cfg = runtime_config.megatron_cfg
        self.dtype = runtime_config.dtype
        self.optimizer_cpu_offload = runtime_config.optimizer_cpu_offload
        self.offload_optimizer_for_logprob = (
            runtime_config.offload_optimizer_for_logprob
        )
        self.is_generation_colocated = runtime_config.is_generation_colocated
        self.sampling_params = runtime_config.sampling_params
        self.final_padded_vocab_size = runtime_config.final_padded_vocab_size

        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Store FP8 config for later use
        self.fp8_cfg = config["megatron_cfg"].get("fp8_cfg", None)

        # Validate configuration
        self.megatron_cfg.validate()

        # Step 4: Setup Megatron model and components
        model_and_optimizer_state = setup_model_and_optimizer(
            config, self.megatron_cfg, init_optimizer
        )

        self.mcore_state = model_and_optimizer_state.state
        self.model = model_and_optimizer_state.model
        self.optimizer = model_and_optimizer_state.optimizer
        self.scheduler = model_and_optimizer_state.scheduler
        self.checkpointing_context = model_and_optimizer_state.checkpointing_context
        param_sync_func = model_and_optimizer_state.param_sync_func
        self.draft_model = model_and_optimizer_state.draft_model

        # Set the param sync function for the model if needed
        if param_sync_func is not None:
            self.megatron_cfg.param_sync_func = param_sync_func

        # Step 5: Setup reference model if needed
        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            self.reference_state_dict = setup_reference_model_state(
                config, self.megatron_cfg, pretrained_path
            )
            self.model = self.move_model(self.model, "cuda")

        # Step 6: Finalize setup
        (
            self.megatron_tokenizer,
            self.megatron_bridge,
            self.should_disable_forward_pre_hook,
            self.dp_size,
        ) = finalize_megatron_setup(
            config,
            self.megatron_cfg,
            hf_model_name,
            worker_sharding_annotations,
            self.model,
            self.optimizer,
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = None
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Note: zero_grad_buffer is called at the start of each global batch iteration
        # in the loop below, so we don't need to call it here.
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                gb_result = process_global_batch(
                    data,
                    loss_fn=loss_fn,
                    dp_group=parallel_state.get_data_parallel_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size,
                    seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                )
                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(num_microbatches)

                loss_post_processor = LossPostProcessor(
                    loss_fn=loss_fn,
                    cfg=self.cfg,
                    num_microbatches=num_microbatches,
                    sampling_params=self.sampling_params,
                    draft_model=self.draft_model,
                )

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    draft_enabled = "draft" in self.cfg and self.cfg["draft"]["enabled"]
                    losses_reduced = megatron_forward_backward(
                        model=self.model,
                        data_iterator=data_iterator,
                        num_microbatches=num_microbatches,
                        seq_length=padded_seq_length,
                        mbs=micro_batch_size,
                        post_processing_fn=loss_post_processor,
                        forward_only=eval_mode,
                        defer_fp32_logits=self.defer_fp32_logits,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        sampling_params=self.sampling_params,
                        straggler_timer=self.mcore_state.straggler_timer,
                        draft_model=self.draft_model,
                        enable_hidden_capture=draft_enabled,
                        use_linear_ce_fusion_loss=self.cfg["megatron_cfg"].get(
                            "use_linear_ce_fusion_loss", False
                        ),
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                pg_collection = get_pg_collection(self.model)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful, mp_group=pg_collection.mp
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm, mp_group=pg_collection.mp
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad, mp_group=pg_collection.mp
                )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            if "_min" in k or "_max" in k:
                                loss_metrics[k] = x[k]
                            else:
                                loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                else:
                    gb_loss_metrics = None

                # Broadcast loss metrics from last stage to all stages
                gb_loss_metrics = broadcast_loss_metrics_from_last_stage(
                    gb_loss_metrics
                )
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=losses,
            data_parallel_group=parallel_state.get_data_parallel_group(),
        )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": mb_metrics,
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )

        use_linear_ce_fusion = self.cfg["megatron_cfg"].get(
            "use_linear_ce_fusion_loss", False
        )
        logprobs_post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            sampling_params=self.sampling_params,
            use_linear_ce_fusion=use_linear_ce_fusion,
        )

        list_of_logprobs = megatron_forward_backward(
            model=self.model,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=logprobs_post_processor,
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
            sampling_params=self.sampling_params,
            straggler_timer=self.mcore_state.straggler_timer,
            use_linear_ce_fusion_loss=use_linear_ce_fusion,
        )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = seq_length - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            tensors = {"logprobs": logprobs}
        else:
            tensors = {"logprobs": None}
        logprobs = broadcast_tensors_from_last_stage(tensors)["logprobs"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    def _apply_state_dict_to_model(
        self,
        source_state_dict: dict,
        *,
        raise_if_key_missing: bool = False,
    ) -> None:
        """Apply a state dict to self.model in-place.

        - Tensors with matching shape: in-place copy (parameters / buffers).
        - _extra_state keys (e.g. FP8 scale/amax) with shape mismatch or non-Tensor value:
          resolve the submodule and call set_extra_state(); supports DDP and Float16Module unwrap.

        Args:
            source_state_dict: State dict to apply (e.g. reference_state_dict or saved model_state_dict).
            raise_if_key_missing: If True, raise when a key in self.model.state_dict() is missing
                from source_state_dict; if False, skip such keys.
        """
        for state_dict_key, param_or_buf in self.model.state_dict().items():
            if (
                not isinstance(param_or_buf, torch.Tensor)
                or "draft_model." in state_dict_key
            ):
                continue
            if state_dict_key not in source_state_dict:
                if raise_if_key_missing:
                    raise ValueError(
                        f"Key '{state_dict_key}' not in source state_dict."
                    )
                continue
            source_value = source_state_dict[state_dict_key]

            # Case 1: Same shape → in-place copy (parameters / buffers)
            if (
                isinstance(source_value, torch.Tensor)
                and param_or_buf.shape == source_value.shape
            ):
                param_or_buf.copy_(source_value)
                continue

            # Case 2: _extra_state (shape mismatch or non-Tensor) → set_extra_state()
            assert "extra_state" in state_dict_key, (
                f"the {state_dict_key} is not an extra_state, but the param_or_buf is mismatched with the reference_state_dict {source_value.shape} != {param_or_buf.shape}."
            )

            submodule_path = state_dict_key.rsplit("._extra_state", 1)[0]
            base_module = getattr(self.model, "module", self.model)
            # Unwrap Float16Module/MoEFloat16Module: state_dict keys are relative to inner .module
            top_level_name = submodule_path.split(".", 1)[0]
            if not hasattr(base_module, top_level_name):
                base_module = getattr(base_module, "module", base_module)
            target_module = base_module.get_submodule(submodule_path)
            target_module.set_extra_state(source_value)

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references.
                  Also disables top-k/top-p filtering since the reference policy's distribution
                  is different from the current policy, making filtered logprobs incompatible.
        On exit: Restores original references and re-flips cuda/cpu, restores sampling_params.
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            # Save original references
            model_state_dict = {}
            for name, item in self.model.state_dict().items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().to(device="cpu", non_blocking=True, copy=True)
                model_state_dict[name] = item

            # Swap reference model state_dict into self.model (reference weights + optional FP8 extra_state)
            self._apply_state_dict_to_model(
                self.reference_state_dict,
                raise_if_key_missing=True,
            )

            if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                gc.collect()
                torch.cuda.empty_cache()

            # Temporarily disable top-k/top-p filtering for reference policy logprobs.
            # The reference policy has different weights, so its top-k/top-p set is
            # inherently different from the current policy. Using filtered logprobs
            # would cause -inf mismatches that cannot be resolved by masking.
            # Note: We keep temperature scaling since it was applied to prev_logprobs.
            saved_sampling_params = self.sampling_params
            if saved_sampling_params is not None:
                self.sampling_params = TrainingSamplingParams(
                    top_k=None,
                    top_p=1.0,
                    temperature=saved_sampling_params.temperature,
                )
            else:
                self.sampling_params = None

            # - self.model is the original reference_model, now on CUDA
            # - self.reference_model is the original model, now on CPU
            yield

            # Restore sampling_params
            self.sampling_params = saved_sampling_params

            # Restore original policy state (weights + FP8 extra_state) from saved model_state_dict
            self._apply_state_dict_to_model(
                model_state_dict,
                raise_if_key_missing=True,
            )

            if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                gc.collect()
                torch.cuda.empty_cache()

            ## re-enable overlap param gather after weight swap
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )

        list_of_outputs = megatron_forward_backward(
            model=self.model,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=TopkLogitsPostProcessor(cfg=self.cfg, k=k),
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
            sampling_params=self.sampling_params,
            straggler_timer=self.mcore_state.straggler_timer,
        )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = seq_length - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            tensors_to_broadcast = {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        else:
            tensors_to_broadcast = {
                "topk_logits": None,
                "topk_indices": None,
            }

        # Broadcast tensors from last stage to all stages
        broadcasted = broadcast_tensors_from_last_stage(tensors_to_broadcast)
        topk_logits = broadcasted["topk_logits"]
        topk_indices = broadcasted["topk_indices"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        mcore_generation_config = cast(
            MegatronGenerationConfig, self.cfg["generation"]["mcore_generation_config"]
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        local_rank = torch.cuda.current_device()
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        model_config.inference_sampling_seed = (node_idx * 1024) + local_rank

        inference_config = InferenceConfig(
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_size_gb=mcore_generation_config["buffer_size_gb"],
            num_cuda_graphs=mcore_generation_config["num_cuda_graphs"],
            block_size_tokens=mcore_generation_config["block_size_tokens"],
            use_cuda_graphs_for_non_decode_steps=mcore_generation_config[
                "use_cuda_graphs_for_non_decode_steps"
            ],
            enable_chunked_prefill=mcore_generation_config["enable_chunked_prefill"],
            unified_memory_level=mcore_generation_config["unified_memory_level"],
            max_tokens=mcore_generation_config["max_tokens"],
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=False,
        )

        dynamic_context = DynamicInferenceContext(model_config, inference_config)
        inference_wrapped_model = GPTInferenceWrapper(self.model, dynamic_context)

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern()
            result.extend(result_step["finished_request_records"])

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [
                x.requests[0].prompt_tokens.tolist() + x.requests[0].generated_tokens
                for x in result
            ],
            "logprobs": [
                x.requests[0].prompt_log_probs + x.requests[0].generated_log_probs
                for x in result
            ],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.requests[0].generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False
        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = [
            task
            for task in self.megatron_bridge.get_conversion_tasks([self.model])
            if task is not None
        ]
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_obj_from_pp_rank(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            get_vllm_qkv_scale_names,
        )

        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        if self.draft_model is not None:
            from nemo_rl.models.megatron.draft import export_eagle_weights_to_hf

            draft_weights = export_eagle_weights_to_hf(
                self.draft_model,
            )
            for name, tensor in draft_weights:
                yield f"draft.{name}", tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self,
        buffer_size_bytes: int = 0,
        kv_scales: Optional[dict[str, float]] = None,
        model_update_transport: str = "cuda_ipc",
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
            model_update_transport=model_update_transport,
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    # ------------------------------------------------------------------
    # rlix integration: CPU bucket cache support (Feature 4)
    # ------------------------------------------------------------------
    #
    # Module-level helpers defined immediately below the class are used
    # by the rlix integration methods.  They are module-level (not class
    # methods) so they can be tested without constructing a full worker.
    #
    # Two-pointer versioning mirrors ROLL megatron_strategy.py:1049–1065:
    #   build_latest_bucket_cache(v)  — called after train_step; all PP ranks
    #                                   participate in the collective gather.
    #   promote_active_checkpoint(v)  — called by BucketCacheLifecycle.promote()
    #                                   to atomically switch the active pointer.
    #
    # selective_sync_active_cache()   — called by ModelUpdateService on the
    #                                   owner rank to transport buckets to infer
    #                                   workers (IPC or NCCL per bucket).
    # ------------------------------------------------------------------

    def _rlix_is_cache_owner(self) -> bool:
        """Return True only for the single rank that builds/holds the cache."""
        return (
            parallel_state.is_pipeline_first_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
            and parallel_state.get_context_parallel_rank() == 0
        )

    def _rlix_get_versioned_cache(self):
        """Lazy-init and return the per-worker VersionedBucketCache."""
        import threading as _threading

        from rlix.pipeline.bucket_cache import VersionedBucketCache

        if not hasattr(self, "_rlix_versioned_cache"):
            self._rlix_versioned_cache = VersionedBucketCache()
            self._rlix_model_update_groups: dict = {}
            self._rlix_cache_init_lock = _threading.Lock()
        return self._rlix_versioned_cache

    @torch.no_grad()
    def build_latest_bucket_cache(self, checkpoint_version: int) -> None:
        """Gather all HF weights into CPU bucket cache as a new 'latest' version.

        ALL PP/TP/EP ranks must participate simultaneously to keep the Megatron
        PP collective alive.  Only the cache owner (pp_rank==0, dp_rank==0,
        tp_rank==0, cp_rank==0) stores the resulting List[BucketRecord].
        Non-owners drain the generator and return.

        Called by the pipeline after train_step returns.  Equivalent to
        ROLL worker.py:363 build_latest_bucket_cache.

        Args:
            checkpoint_version: Step number (or -1 for base model).
        """
        import logging

        from rlix.pipeline.bucket_cache import _bucket_named_tensors

        logger = logging.getLogger(__name__)
        checkpoint_version = int(checkpoint_version)
        is_owner = self._rlix_is_cache_owner()

        # Ensure refit_conversion_tasks is populated (needed by the iterator).
        self.prepare_refit_info()

        # Bucket packing: accumulate named tensors until we reach bucket_size_bytes,
        # then flush a BucketRecord.  Size is read from the worker config (key
        # "rlix_bucket_size_bytes") or the env var RLIX_BUCKET_SIZE_BYTES.
        # A hardcoded silent default is intentionally prohibited — callers must set
        # the config or env var so the value is always visible in logs.
        bucket_size_bytes: int = _rlix_get_bucket_size_bytes(self)
        if is_owner and checkpoint_version == -1:
            # Init-time VRAM check: verify the bucket fits in available GPU memory.
            _rlix_check_vram(bucket_size_bytes, logger)

        buckets = []
        current_batch: list = []
        current_bytes = 0

        for name, tensor in self._iter_params_with_optional_kv_scales():
            if not is_owner:
                # Non-owner: must still exhaust the generator to keep the PP
                # collective alive; do NOT store anything.
                continue

            cpu_t = tensor.detach().cpu().contiguous()
            nbytes = cpu_t.numel() * cpu_t.element_size()

            # Fail fast: a single tensor larger than bucket_size_bytes can never be
            # staged within the GPU VRAM budget (spec: nemorl-port-plan.md line 342-343).
            # Matches ROLL's send_recv_utils.py assertion pattern.
            if nbytes > bucket_size_bytes:
                raise RuntimeError(
                    f"[rlix] Parameter '{name}' ({nbytes >> 20} MB) exceeds "
                    f"bucket_size_bytes ({bucket_size_bytes >> 20} MB). "
                    "Increase RLIX_BUCKET_SIZE_BYTES or bucket_size_bytes config."
                )

            if current_batch and current_bytes + nbytes > bucket_size_bytes:
                # Flush current batch into a BucketRecord before appending.
                buckets.append(_bucket_named_tensors(current_batch))
                current_batch = []
                current_bytes = 0

            current_batch.append((name, cpu_t))
            current_bytes += nbytes

        if is_owner and current_batch:
            buckets.append(_bucket_named_tensors(current_batch))

        if is_owner:
            cache = self._rlix_get_versioned_cache()
            cache.build_latest(checkpoint_version, buckets)
            total_bytes = sum(b.cpu_uint8_bucket.numel() for b in buckets)
            logger.info(
                "[rlix] build_latest_bucket_cache version=%d "
                "buckets=%d total_bytes=%d",
                checkpoint_version, len(buckets), total_bytes,
            )
            # Host-RAM fail-fast: two-pointer versioning keeps ≤ 2 full model copies.
            # Check against actual packed model size, not per-bucket size.
            # Spec: nemorl-port-plan.md line 337 — "估算 total_cpu_cache_bytes … fail fast".
            if checkpoint_version == -1:
                try:
                    import psutil as _psutil
                    available_ram = _psutil.virtual_memory().available
                    ram_budget = int(available_ram * 0.8)
                    two_copy = 2 * total_bytes
                    if two_copy > ram_budget:
                        raise RuntimeError(
                            f"[rlix] Host RAM budget exceeded: "
                            f"2 × model ({two_copy >> 20} MB) > "
                            f"80% of available RAM ({ram_budget >> 20} MB). "
                            "Reduce model size or increase host RAM."
                        )
                    logger.info(
                        "[rlix] host_ram_check_ok two_copy=%d MB available_ram=%d MB",
                        two_copy >> 20, available_ram >> 20,
                    )
                except ImportError:
                    logger.warning("[rlix] psutil not installed — skipping host-RAM budget check")

    def promote_active_checkpoint(self, version: int) -> None:
        """Atomically switch the active cache pointer to *version*.

        Non-owner ranks return immediately (no-op).  Only the cache owner
        (pp_rank==0, dp_rank==0, tp_rank==0, cp_rank==0) has a live cache.

        Called by ``BucketCacheLifecycle.promote()`` after
        ``build_latest_bucket_cache(version)`` has completed on all workers.
        Equivalent to ROLL worker.py:387 promote_active_checkpoint.

        Args:
            version: Must match a version passed to ``build_latest_bucket_cache``.
        """
        import logging

        logger = logging.getLogger(__name__)
        version = int(version)

        if not self._rlix_is_cache_owner():
            return

        cache = self._rlix_get_versioned_cache()
        cache.promote(version)
        logger.info("[rlix] promote_active_checkpoint version=%d", version)

    @torch.no_grad()
    def selective_sync_active_cache(
        self,
        sync_id: str,
        comm_plan: Optional[dict],
        tgt_dp_ranks: list[int],
        tgt_workers: list,
        tgt_device_mapping: list[int],
        tgt_num_gpus_per_worker: int,
        adapters_to_sync: Optional[list[str]] = None,
        model_update_transport: str = "cpu_serialize",
    ) -> Optional[dict]:
        """Transport active cache buckets to inference workers (IPC or NCCL).

        Non-owner ranks return immediately.  Owner holds the cache lock for
        the entire transport loop to prevent a concurrent promote/build from
        racing the sender read.

        Per-bucket staging constraint: CPU→GPU one bucket at a time; delete
        immediately after the barrier.  Forbidden to load the full model to
        GPU at once.

        Args:
            sync_id: Unique sync identifier (used for group name lookup).
            comm_plan: Communication plan built by ModelUpdateService for the
                owner rank.  Non-owners receive None.
            tgt_dp_ranks: Inference DP ranks to update.
            tgt_workers: All inference worker Ray actor handles.
            tgt_device_mapping: GPU device indices per infer worker.
            tgt_num_gpus_per_worker: Number of GPUs per infer worker.
            adapters_to_sync: Unused; reserved for LoRA adapter sync.

        Returns:
            ``{"weight_stats": {...}}`` from the owner for post-sync
            verification, or ``None`` from non-owners.
        """
        import logging

        import torch.distributed as dist

        logger = logging.getLogger(__name__)

        if not self._rlix_is_cache_owner() or comm_plan is None:
            return None

        cache = self._rlix_get_versioned_cache()
        ipc_targets: list[dict] = comm_plan[next(iter(comm_plan))].get("ipc_targets", [])
        broadcast_local_ranks_by_dp_rank: dict[int, list[int]] = (
            comm_plan[next(iter(comm_plan))].get("broadcast_local_ranks_by_dp_rank", {})
        )
        group_name: str = comm_plan[next(iter(comm_plan))]["group_name"]
        dp_rank_to_worker = {
            int(dp_rank): tgt_workers[dp_rank]
            for dp_rank in tgt_dp_ranks
        }

        # Hold cache lock for the entire transport to prevent a concurrent
        # promote/build from modifying the active pointer during transport.
        with cache._cache_lock:
            buckets = cache.get_active_buckets()
            n_buckets = len(buckets)

            for bucket_idx, bucket in enumerate(buckets):
                # Stage single bucket CPU→GPU; release immediately after barrier.
                staging_buf: Optional[torch.Tensor] = None
                try:
                    staging_buf = bucket.cpu_uint8_bucket.pin_memory().cuda()
                    logger.info(
                        "[ModelUpdateService] bucket_send bucket_idx=%d/%d "
                        "bytes=%d group_name=%s sync_id=%s",
                        bucket_idx, n_buckets, bucket.used_bytes, group_name, sync_id,
                    )

                    recv_refs = []

                    # IPC path: colocated same-GPU workers.
                    # model_update_transport selects the payload format:
                    # - "cuda_ipc": CUDA IPC handle (zero-copy, same physical GPU).
                    #   Spec line 316: NCCL CANNOT form a group on the same GPU; IPC is required.
                    # - "cpu_serialize": CPU uint8 bucket DMA to receiver GPU.
                    #   Used when CUDA IPC is unavailable (e.g. containerized or cross-GPU).
                    for ipc_target in ipc_targets:
                        dp_rank = int(ipc_target["dp_rank"])
                        local_ranks = ipc_target["local_ranks"]

                        if model_update_transport == "cuda_ipc":
                            # Zero-copy IPC: share the GPU staging buffer with the colocated process.
                            from nemo_rl.models.policy.utils import get_handle_from_tensor
                            torch.cuda.current_stream().synchronize()
                            cuda_ipc_handle = get_handle_from_tensor(staging_buf)
                            payload = {
                                "param_names": bucket.param_names,
                                "shapes": bucket.shapes,
                                "dtypes": bucket.dtypes,
                                "offsets": bucket.offsets,
                                "used_bytes": bucket.used_bytes,
                                "cuda_ipc_handle": cuda_ipc_handle,
                            }
                        else:
                            # cpu_serialize: send the CPU uint8 bucket (DMA on receiver side).
                            payload = {
                                "param_names": bucket.param_names,
                                "shapes": bucket.shapes,
                                "dtypes": bucket.dtypes,
                                "offsets": bucket.offsets,
                                "used_bytes": bucket.used_bytes,
                                "cpu_uint8_bucket": bucket.cpu_uint8_bucket,
                            }

                        recv_refs.append(
                            dp_rank_to_worker[dp_rank].update_parameter_in_bucket.remote(
                                payload, local_ranks, model_update_transport
                            )
                        )

                    # NCCL broadcast path: cross-GPU workers.
                    if group_name in self._rlix_model_update_groups:
                        nccl_group = self._rlix_model_update_groups[group_name]
                        dist.broadcast(staging_buf, src=0, group=nccl_group)

                        for dp_rank, broadcast_local_ranks in broadcast_local_ranks_by_dp_rank.items():
                            recv_refs.append(
                                dp_rank_to_worker[int(dp_rank)].broadcast_parameter.remote(
                                    group_name,
                                    bucket.param_names,
                                    bucket.dtypes,
                                    bucket.shapes,
                                    broadcast_local_ranks,
                                )
                            )

                    import ray as _ray
                    _ray.get(recv_refs)

                    logger.info(
                        "[ModelUpdateService] bucket_ack bucket_idx=%d/%d sync_id=%s",
                        bucket_idx, n_buckets, sync_id,
                    )
                finally:
                    # Release GPU staging buffer immediately after barrier.
                    del staging_buf
                    staging_buf = None

            # Flush GPU streams before teardown: dist.broadcast is async; synchronize
            # ensures all NCCL kernels have completed before destroying the communicator.
            # _ray.get(recv_refs) above already confirmed receivers finished, so this
            # just ensures sender-side CUDA stream is clean.
            torch.cuda.synchronize()
            # Tear down the NCCL collective group while still holding _cache_lock.
            # Spec (nemorl-port-plan.md line 402): lock must span "cache lookup →
            # transport → NCCL teardown" — releasing before teardown completes
            # would allow a concurrent build_latest / promote to race the sender.
            self.destroy_collective_group(group_name)

        # Compute weight stats for optional post-sync verification.
        weight_stats: dict = {}
        try:
            sd = {n: t for n, t in self._iter_params_with_optional_kv_scales()}
            vals = [t.float() for t in sd.values() if t.numel() > 0]
            if vals:
                all_flat = torch.cat([v.flatten() for v in vals])
                weight_stats = {
                    "sum": float(all_flat.sum()),
                    "max": float(all_flat.max()),
                    "min": float(all_flat.min()),
                }
        except Exception:
            pass

        return {"weight_stats": weight_stats}

    def setup_collective_group(
        self,
        model_update_name: str,
        comm_plan: dict,
        mode: str,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Join a dynamic NCCL group for selective model weight broadcast.

        The sender (mode='sender') joins as rank 0; receivers join at
        their assigned rank from the comm_plan.

        Args:
            model_update_name: Unique sync identifier (used as group name).
            comm_plan: Communication plan dict with master_addr/port and
                world size info.
            mode: 'sender' (rank 0) or 'receiver'.
            timeout_s: Optional NCCL init timeout in seconds.
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        cache = self._rlix_get_versioned_cache()
        plan_entry = comm_plan[next(iter(comm_plan))]
        group_name: str = plan_entry["group_name"]
        master_addr: str = plan_entry["master_addr"]
        master_port: int = int(plan_entry["master_port"])

        if mode == "sender":
            tgt_devices = plan_entry.get("tgt_devices", [])
            world_size = 1 + len(tgt_devices)
            rank = 0
        else:
            # Receiver: find our rank from tgt_devices list.
            tgt_devices = plan_entry.get("tgt_devices", [])
            world_size = 1 + len(tgt_devices)
            local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            rank = 1  # default; real multi-receiver assignment handled by StatelessProcessGroup ordering
            for i, dev in enumerate(tgt_devices):
                if int(dev.get("rank", -1)) == local_rank:
                    rank = i + 1
                    break

        pg = StatelessProcessGroup(
            master_address=master_addr,
            port=master_port,
            rank=rank,
            world_size=world_size,
        )
        pg.init_nccl_communicator(device=self.device if hasattr(self, "device") else torch.device("cuda"))
        self._rlix_model_update_groups[group_name] = pg

    def destroy_collective_group(self, group_name: str) -> None:
        """Destroy a dynamic NCCL group created by setup_collective_group.

        No-op if the group does not exist (IPC-only ranks never join the
        NCCL group, so this guard is required).

        Args:
            group_name: Group name as used in setup_collective_group.
        """
        import logging

        import torch.distributed as dist

        logger = logging.getLogger(__name__)
        groups = getattr(self, "_rlix_model_update_groups", {})
        if group_name not in groups:
            return
        pg = groups.pop(group_name)
        try:
            dist.destroy_process_group(pg)
        except Exception as exc:
            logger.warning(
                "[rlix] destroy_collective_group failed group_name=%s: %s",
                group_name, exc,
            )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    def destroy_nccl_groups(self):
        """Release Megatron NCCL communicator buffers owned by this worker."""
        from nemo_rl.models.megatron.nccl_offload import destroy_megatron_nccl_groups

        return destroy_megatron_nccl_groups()

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                new_state_dict = {}
                for name, item in model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device=device, non_blocking=True, copy=True
                        )
                    new_state_dict[name] = item
                model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            convert_calibration_to_vllm_format,
        )

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


# ---------------------------------------------------------------------------
# rlix module-level helpers for bucket cache (Feature 4)
# ---------------------------------------------------------------------------
# These are module-level so they can be imported and tested without
# constructing a full MegatronPolicyWorkerImpl.
# ---------------------------------------------------------------------------

_RLIX_BUCKET_SIZE_ENV = "RLIX_BUCKET_SIZE_BYTES"
_RLIX_BUCKET_SIZE_DEFAULT = 256 * 1024 * 1024  # 256 MB documented default
# Transport scratch (NCCL send-side staging overhead estimate).
_RLIX_TRANSPORT_SCRATCH_MB = 64


def _rlix_get_bucket_size_bytes(worker) -> int:
    """Return the configured bucket size in bytes for rlix cache building.

    Priority order:
    1. ``worker.cfg["rlix"]["bucket_size_bytes"]`` (explicit config key).
    2. ``RLIX_BUCKET_SIZE_BYTES`` environment variable.
    3. Documented default ``_RLIX_BUCKET_SIZE_DEFAULT`` (256 MB), emitted
       as a WARNING so users know the default is active.

    This function is intentionally NOT a silent fallback — every code path
    logs the active value so callers are always aware.

    Args:
        worker: MegatronPolicyWorkerImpl instance (for cfg access).

    Returns:
        Bucket size in bytes (positive int).

    Raises:
        ValueError: If the resolved value is <= 0.
    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    # 1. Worker config
    cfg = getattr(worker, "cfg", {}) or {}
    rlix_cfg = cfg.get("rlix", {}) or {}
    if "bucket_size_bytes" in rlix_cfg:
        val = int(rlix_cfg["bucket_size_bytes"])
        logger.info("[rlix] bucket_size_bytes=%d (from worker.cfg['rlix'])", val)
        if val <= 0:
            raise ValueError(f"[rlix] bucket_size_bytes must be > 0, got {val}")
        return val

    # 2. Environment variable
    env_val = os.environ.get(_RLIX_BUCKET_SIZE_ENV)
    if env_val is not None:
        val = int(env_val)
        logger.info("[rlix] bucket_size_bytes=%d (from env %s)", val, _RLIX_BUCKET_SIZE_ENV)
        if val <= 0:
            raise ValueError(f"[rlix] {_RLIX_BUCKET_SIZE_ENV} must be > 0, got {val}")
        return val

    # Spec (nemorl-port-plan.md line 343): bucket_size_bytes must be an explicit
    # configuration value — no implicit default is allowed.  Fail fast so operators
    # are forced to make the staging-VRAM budget decision visible in config.
    raise RuntimeError(
        "[rlix] bucket_size_bytes is not configured. "
        f"Set worker.cfg['rlix']['bucket_size_bytes'] or env {_RLIX_BUCKET_SIZE_ENV}. "
        "No implicit default is permitted (spec: nemorl-port-plan.md line 343)."
    )


def _rlix_check_vram(bucket_size_bytes: int, logger) -> None:
    """Fail fast if bucket_size_bytes exceeds available GPU VRAM margin.

    Called once at init time (when ``checkpoint_version == -1``).
    Peak staging VRAM estimate: ``bucket_size_bytes + _RLIX_TRANSPORT_SCRATCH_MB * 1024^2``.

    Args:
        bucket_size_bytes: Configured bucket size in bytes.
        logger: Logger instance (already has worker context).

    Raises:
        RuntimeError: If estimated peak staging VRAM exceeds 90% of free GPU memory.
    """
    try:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        scratch_bytes = _RLIX_TRANSPORT_SCRATCH_MB * 1024 * 1024
        peak_bytes = bucket_size_bytes + scratch_bytes
        threshold = 0.9 * free_bytes
        logger.info(
            "[rlix] vram_check free_gb=%.2f peak_staging_gb=%.2f bucket_size_mb=%d",
            free_bytes / 1024 ** 3,
            peak_bytes / 1024 ** 3,
            bucket_size_bytes // (1024 * 1024),
        )
        if peak_bytes > threshold:
            raise RuntimeError(
                f"[rlix] bucket_size_bytes={bucket_size_bytes} exceeds VRAM margin: "
                f"peak_staging={peak_bytes / 1024**3:.2f} GB > 90% of free={free_bytes / 1024**3:.2f} GB. "
                f"Reduce RLIX_BUCKET_SIZE_BYTES or free GPU memory before training."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        # Non-CUDA environments (CPU-only, mock): skip the check.
        logger.debug("[rlix] vram_check skipped: %s", exc)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(MegatronPolicyWorkerImpl):
    pass
