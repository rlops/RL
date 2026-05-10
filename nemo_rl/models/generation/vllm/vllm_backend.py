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
import io
import pickle
import traceback
from typing import Any

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_model.load_weights(weights=draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None
        policy_weights = None
        draft_weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                frames = self.zmq_socket.recv_multipart(copy=False)

                if len(frames) == 1:
                    payload = pickle.loads(frames[0].buffer)
                    if payload == IPCProtocol.COMPLETE:
                        # means the update is done
                        from vllm.model_executor.model_loader.utils import (
                            process_weights_after_loading,
                        )

                        process_weights_after_loading(
                            self.model_runner.model, self.model_config, self.device
                        )
                        self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                        break

                    ipc_handle_or_bytes, list_keys, used_bytes = payload
                    buffer = rebuild_cuda_tensor_from_ipc(
                        ipc_handle_or_bytes, self.device.index
                    )
                elif len(frames) == 3 and bytes(frames[0].buffer) == b"cpu_serialize":
                    list_keys, used_bytes = pickle.loads(frames[1].buffer)
                    bucket_data = torch.load(
                        io.BytesIO(frames[2].buffer), weights_only=True
                    )
                    buffer = bucket_data["bucket"]
                    if not buffer.is_pinned():
                        buffer = buffer.pin_memory()
                    buffer = buffer.to(device=self.device, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                else:
                    raise RuntimeError(
                        f"Unexpected ZMQ frame format in update_weights_via_ipc_zmq: {len(frames)} frame(s)"
                    )

                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weights.append(
                        (
                            key,
                            buffer[offset : offset + size_in_bytes]
                            .view(dtype=dtype)
                            .view(shape),
                        )
                    )
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size
                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )
                # Load weights into the model
                from nemo_rl.models.generation.vllm.quantization import fp8

                policy_weights, draft_weights = self._split_policy_and_draft_weights(
                    weights
                )
                if fp8.is_fp8_model(self.model_runner.vllm_config):
                    # the fp8 load_weights additionally casts bf16 weights into fp8
                    fp8.load_weights(policy_weights, self.model_runner)
                else:
                    self.model_runner.model.load_weights(weights=policy_weights)

                self._load_draft_weights(draft_weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weights, policy_weights, draft_weights, buffer
                weights = None
                policy_weights = None
                draft_weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        def _load_model_weights(weights, model_runner):
            """Load model weights.

            Args:
                weights: List[(name, tensor)]
                model_runner: vLLM ModelRunner

            Returns:
                None
            """
            from nemo_rl.models.generation.vllm.quantization import fp8

            policy_weights, draft_weights = self._split_policy_and_draft_weights(
                weights
            )

            if fp8.is_fp8_model(model_runner.vllm_config):
                # the fp8 load_weights additionally casts bf16 weights into fp8
                fp8.load_weights(policy_weights, model_runner)
            else:
                model_runner.model.load_weights(weights=policy_weights)

            self._load_draft_weights(draft_weights)

        load_model_weight_func = lambda x: _load_model_weights(x, self.model_runner)

        try:
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # rlix integration: selective sync receiver methods (Feature 4)
    # ------------------------------------------------------------------

    def setup_collective_group(
        self,
        model_update_name: str,
        comm_plan: dict,
        mode: str,
        timeout_s: float | None = None,
    ) -> None:
        """Join a dynamic NCCL group for selective model weight broadcast.

        Stores the group in ``self._model_update_groups[group_name]``.

        Args:
            model_update_name: Unique sync identifier.
            comm_plan: Communication plan with master_addr/port/world_size.
            mode: 'receiver' (inference workers are always receivers).
            timeout_s: Optional NCCL init timeout in seconds (unused; StatelessProcessGroup uses its own timeout).
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        if not hasattr(self, "_model_update_groups"):
            self._model_update_groups: dict = {}  # pyrefly: ignore[implicitly-defined-attribute]

        plan_entry = comm_plan[next(iter(comm_plan))]
        group_name: str = plan_entry["group_name"]
        master_addr: str = plan_entry["master_addr"]
        master_port: int = int(plan_entry["master_port"])
        tgt_devices: list = plan_entry.get("tgt_devices", [])
        world_size = 1 + len(tgt_devices)

        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank = 1
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
        pg.init_nccl_communicator(device=self.device)
        self._model_update_groups[group_name] = pg

    def update_parameter_in_bucket(
        self,
        payload: dict,
        ipc_local_ranks: list[int],
        model_update_transport: str,
        is_lora: bool = False,
    ) -> None:
        """Receive a packed weight bucket and load it into the model.

        Two transport modes (spec: nemorl-port-plan.md lines 316-323, 344-345):

        ``"cpu_serialize"`` — payload contains ``cpu_uint8_bucket`` (CPU uint8 tensor).
            DMA copies the buffer to GPU, then unpacks per-param tensors.
            Used for cross-GPU or containerized deployments where CUDA IPC is unavailable.

        ``"cuda_ipc"`` — payload contains ``cuda_ipc_handle`` (CUDA IPC handle tuple).
            Rebuilds the GPU tensor directly via ``rebuild_cuda_tensor_from_ipc()``
            (zero-copy for same-physical-GPU colocated processes).
            Required when sender and receiver share a physical GPU; NCCL cannot
            form a group between two ranks on the same GPU (spec line 316).

        Args:
            payload: Transport dict. cpu_serialize: keys ``param_names``, ``shapes``,
                ``dtypes``, ``offsets``, ``used_bytes``, ``cpu_uint8_bucket``.
                cuda_ipc: same keys but ``cuda_ipc_handle`` instead of ``cpu_uint8_bucket``.
            ipc_local_ranks: Infer-local ranks that should process this bucket.
            model_update_transport: ``"cpu_serialize"`` or ``"cuda_ipc"``.
            is_lora: Reserved for LoRA adapter weights (not yet used).
        """
        # Use the vLLM worker's own rank, not the distributed process-group rank.
        # ipc_local_ranks contains LOCAL ranks within the worker (comm-plan contract,
        # spec nemorl-port-plan.md:406-412); torch.distributed.get_rank() would be
        # the wrong identity in multi-node setups. (Matches ROLL worker.py:757.)
        local_rank = getattr(self, "rank", None)
        if local_rank is None:
            # Fallback for workers that don't expose .rank — use distributed rank.
            local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if local_rank not in ipc_local_ranks:
            return

        from rlix.pipeline.bucket_cache import BucketRecord, unpack_bucket_record

        # --- Reconstruct named tensors from transport payload ---
        if model_update_transport == "cuda_ipc":
            # Zero-copy: sender and receiver share the same physical GPU.
            # Rebuild GPU tensor from IPC handle — no CPU roundtrip.
            # Spec line 316: NCCL cannot form a group on the same GPU; IPC is required.
            from torch.multiprocessing.reductions import rebuild_cuda_tensor
            device_id = self.device.index if hasattr(self.device, "index") else 0
            ipc_args = list(payload["cuda_ipc_handle"][0])
            ipc_args[6] = device_id  # patch device index
            buf_gpu = rebuild_cuda_tensor(*ipc_args)
            torch.cuda.current_stream().synchronize()

            # Reconstruct named tensors directly from GPU buffer (no CPU copy)
            weights = []
            for name, shape, dtype, offset in zip(
                payload["param_names"], payload["shapes"],
                payload["dtypes"], payload["offsets"]
            ):
                num_elements = 1
                for s in shape:
                    num_elements *= s
                nbytes = num_elements * torch.empty(0, dtype=dtype).element_size()
                flat = buf_gpu[offset : offset + nbytes].view(dtype)
                weights.append((name, flat.reshape(shape)))
        else:
            # cpu_serialize: DMA copy pinned CPU buffer → GPU, then unpack
            buf_gpu = payload["cpu_uint8_bucket"].pin_memory().to(self.device, non_blocking=True)
            torch.cuda.current_stream().synchronize()

            record = BucketRecord(
                param_names=payload["param_names"],
                shapes=payload["shapes"],
                dtypes=payload["dtypes"],
                offsets=payload["offsets"],
                used_bytes=payload["used_bytes"],
                cpu_uint8_bucket=buf_gpu.cpu(),
            )
            named_tensors = unpack_bucket_record(record)
            weights = [(name, t.to(self.device)) for name, t in named_tensors]

        from nemo_rl.models.generation.vllm.quantization import fp8

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)
        self._load_draft_weights(draft_weights)
        torch.cuda.current_stream().synchronize()
        del buf_gpu, weights, policy_weights, draft_weights

    def broadcast_parameter(
        self,
        group_name: str,
        names: list[str],
        dtypes: list,
        shapes: list,
        broadcast_local_ranks: list[int],
        is_lora: bool = False,
    ) -> None:
        """Receive a packed NCCL broadcast and load weights into the model.

        Reuses the ``packed_broadcast_consumer`` pattern from
        ``update_weights_from_collective`` (vllm_backend.py:294–303).

        Args:
            group_name: NCCL group name created by ``setup_collective_group``.
            names: HF param names in order (matches sender's bucket).
            dtypes: Per-param dtypes.
            shapes: Per-param shapes.
            broadcast_local_ranks: Infer-local ranks that participate.
                Ranks not in this list return immediately (guard).
            is_lora: If True, payload contains LoRA adapter weights (reserved,
                not yet used — base weights always use False).
        """
        if not hasattr(self, "_model_update_groups"):
            return
        if group_name not in self._model_update_groups:
            return

        # Use the vLLM worker's own local rank (same convention as
        # update_parameter_in_bucket above). torch.distributed.get_rank()
        # returns the global rank in WORLD; broadcast_local_ranks carries
        # LOCAL ranks within the worker, so under TP>1 / multi-node the
        # global rank never matches and every receiver silently
        # early-returns.
        local_rank = getattr(self, "rank", None)
        if local_rank is None:
            local_rank = (
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            )
        if local_rank not in broadcast_local_ranks:
            return

        group = self._model_update_groups[group_name]

        # Calculate total buffer size (aligned, same as sender packing).
        from nemo_rl.models.policy.utils import calculate_aligned_size

        total_bytes = 0
        for name, shape, dtype in zip(names, shapes, dtypes):
            num_elements = 1
            for s in shape:
                num_elements *= s
            nbytes = num_elements * torch.empty(0, dtype=dtype).element_size()
            total_bytes = calculate_aligned_size(total_bytes + nbytes)

        recv_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=self.device)
        group.broadcast(recv_buf, src=0)

        weights = []
        offset = 0
        for name, shape, dtype in zip(names, shapes, dtypes):
            num_elements = 1
            for s in shape:
                num_elements *= s
            nbytes = num_elements * torch.empty(0, dtype=dtype).element_size()
            flat = recv_buf[offset : offset + nbytes].view(dtype)
            weights.append((name, flat.reshape(shape)))
            offset = calculate_aligned_size(offset + nbytes)

        from nemo_rl.models.generation.vllm.quantization import fp8

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)
        self._load_draft_weights(draft_weights)
        torch.cuda.current_stream().synchronize()
        del recv_buf, weights, policy_weights, draft_weights

    def destroy_collective_group(self, group_name: str) -> None:
        """Destroy a dynamic NCCL group.

        No-op guard: IPC-only ranks never join the NCCL group, so
        ``group_name`` may not be present.

        Args:
            group_name: Group name as used in ``setup_collective_group``.
        """
        import torch.distributed as dist

        if not hasattr(self, "_model_update_groups"):
            return
        if group_name not in self._model_update_groups:
            return
        pg = self._model_update_groups.pop(group_name)
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    def verify_model(self, expected_stats: dict) -> None:
        """Verify model weights match expected statistics after sync.

        Args:
            expected_stats: Dict with keys ``sum``, ``max``, ``min`` computed
                by the sender over all weight tensors.

        Raises:
            RuntimeError: If any statistic deviates from expected by > 1e-3.
        """
        state_dict = self.model_runner.model.state_dict()
        vals = [t.float() for t in state_dict.values() if t.numel() > 0]
        if not vals:
            return
        all_flat = torch.cat([v.flatten() for v in vals])
        actual = {
            "sum": float(all_flat.sum()),
            "max": float(all_flat.max()),
            "min": float(all_flat.min()),
        }
        tol = 1e-3
        for key in ("sum", "max", "min"):
            if key not in expected_stats:
                continue
            if abs(actual[key] - expected_stats[key]) > tol * (abs(expected_stats[key]) + 1.0):
                raise RuntimeError(
                    f"verify_model: {key} mismatch: "
                    f"expected={expected_stats[key]:.6f} actual={actual[key]:.6f}"
                )

    def finalize_weight_update(self) -> None:
        """Run post-loading weight processing (FP8 KV cache, etc.).

        Must be called after all buckets have been loaded via
        ``update_parameter_in_bucket`` or ``broadcast_parameter``.
        """
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        process_weights_after_loading(
            self.model_runner.model, self.model_config, self.device
        )
        self._maybe_process_fp8_kv_cache()

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
