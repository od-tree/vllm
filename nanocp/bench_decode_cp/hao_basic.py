"""DLSlime ``KernelImpl::Basic`` for this bench.

This is ``intranode_alltoall_kernel`` from nano-infra/DLSlime
``4f1fce6`` (``feat/hao-basic-alltoall-offsets``), not the older
one-block ``all_to_all_intra_ll_kernel``. One launch per rank:

- grid is ``min(16, batch) * world``, block is 16 warps
- each block writes one destination's token slice with ``int4``
  non-cached loads and streaming stores
- the last block for that destination publishes the flag and waits
  until the destination has published ours

The pointer table is the in-process stand-in for their IPC mesh: peer
access is already on, so each rank stores the other ranks' ``data_ptr``.
A second flag kernel is not used. The kernel returns only after the
handshake, which is when this rank's buffer is complete.

Two logical ranks on one GPU do not take the handshake. Their spin
assumes the peer kernel is already running on another GPU; on one
stream it never starts. The copies still use this kernel.
"""

from __future__ import annotations

import torch

_EXT = None

# The pybind file is compiled separately from the .cu, so it needs a declaration.
_CPP = r"""
#include <torch/extension.h>
void hao_basic_launch(torch::Tensor x,
                      torch::Tensor buffer_ptrs,
                      torch::Tensor signal_ptrs,
                      torch::Tensor semaphore,
                      int64_t rank,
                      int64_t world,
                      int64_t batch_size,
                      int64_t max_batch,
                      int64_t hidden,
                      int64_t is_transpose,
                      int64_t handshake,
                      torch::Tensor mask,
                      torch::Tensor dst_rows,
                      int64_t has_mask,
                      int64_t has_dst_rows);
"""

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

__device__ __forceinline__ int4 ld_nc_int4(const int4* ptr) {
    int4 value;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr));
    return value;
}

__device__ __forceinline__ void st_na_int4(int4* ptr, int4 value) {
    asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
                 "r"(value.x),
                 "r"(value.y),
                 "r"(value.z),
                 "r"(value.w));
}

__device__ __forceinline__ void st_release_sys(int* ptr, int value) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys(const int* ptr) {
    int value;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

// Same remainder shape as DeepEP UNROLLED_WARP_COPY(..., 4, lane, n, ...).
__device__ __forceinline__ void warp_copy_int4(int n, int lane, int4* dst, const int4* src) {
    constexpr int kUnroll = 4;
    constexpr int kStride = 32 * kUnroll;
    int4 regs[kUnroll];
    const int full = (n / kStride) * kStride;
    for (int base = lane; base < full; base += kStride) {
        #pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
            regs[j] = ld_nc_int4(src + base + j * 32);
        }
        #pragma unroll
        for (int j = 0; j < kUnroll; ++j) {
            st_na_int4(dst + base + j * 32, regs[j]);
        }
    }
    const int tail = full + lane;
    #pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
        if (tail + j * 32 < n) {
            regs[j] = ld_nc_int4(src + tail + j * 32);
        }
    }
    #pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
        if (tail + j * 32 < n) {
            st_na_int4(dst + tail + j * 32, regs[j]);
        }
    }
}

template <int kNumWarps, bool kUseDstRowIndices>
__global__ void __launch_bounds__(kNumWarps * 32, 1)
intranode_alltoall_kernel(const void* x,
                          void** buffer_ptr,
                          void** signal_ptr,
                          int* local_semaphore,
                          int local_rank,
                          int world_size,
                          int batch_size,
                          int max_batch_size,
                          int hidden_dim,
                          int itemsize,
                          int is_transpose,
                          int handshake,
                          const int* mask,
                          const int* dst_row_indices) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int dst_rank = blockIdx.x % world_size;
    const int num_blocks_for_rank = gridDim.x / world_size;
    const int group_idx = blockIdx.x / world_size;

    const int tokens_per_block = (batch_size + num_blocks_for_rank - 1) / num_blocks_for_rank;
    const int start_token = group_idx * tokens_per_block;
    int end_token = start_token + tokens_per_block;
    if (end_token > batch_size) {
        end_token = batch_size;
    }

    const uint32_t bytes_per_token = static_cast<uint32_t>(hidden_dim) * static_cast<uint32_t>(itemsize);
    const int ints_per_token = static_cast<int>(bytes_per_token / sizeof(int4));
    const bool target_major = !kUseDstRowIndices && (is_transpose || mask == nullptr);
    const int4* src_base = reinterpret_cast<const int4*>(x);
    if (target_major && batch_size > 0) {
        src_base += static_cast<uint64_t>(dst_rank) * static_cast<uint64_t>(batch_size) * ints_per_token;
    }

    int4* dst_base = nullptr;
    if constexpr (!kUseDstRowIndices) {
        auto* dst_bytes = reinterpret_cast<int8_t*>(buffer_ptr[dst_rank]);
        dst_bytes += static_cast<uint64_t>(local_rank) * static_cast<uint64_t>(batch_size) * bytes_per_token;
        dst_base = reinterpret_cast<int4*>(dst_bytes);
    }

    if (start_token < batch_size) {
        for (int token = start_token; token < end_token; ++token) {
            int4* token_dst = nullptr;
            if constexpr (kUseDstRowIndices) {
                const int dst_row = __ldg(dst_row_indices + dst_rank * max_batch_size + token);
                if (dst_row < 0 || dst_row >= world_size * max_batch_size) {
                    continue;
                }
                auto* dst_bytes = reinterpret_cast<int8_t*>(buffer_ptr[dst_rank]);
                dst_bytes += static_cast<uint64_t>(dst_row) * bytes_per_token;
                token_dst = reinterpret_cast<int4*>(dst_bytes);
            } else {
                if (mask != nullptr && __ldg(mask + dst_rank * batch_size + token) == 0) {
                    continue;
                }
                token_dst = dst_base + static_cast<uint64_t>(token) * ints_per_token;
            }

            const int ints_per_warp = (ints_per_token + kNumWarps - 1) / kNumWarps;
            const int warp_offset = warp_id * ints_per_warp;
            int warp_len = ints_per_token - warp_offset;
            if (warp_len > ints_per_warp) {
                warp_len = ints_per_warp;
            }
            if (warp_len < 0) {
                warp_len = 0;
            }
            if (warp_len > 0) {
                const int4* warp_src =
                    src_base + static_cast<uint64_t>(token) * ints_per_token + warp_offset;
                warp_copy_int4(warp_len, lane_id, token_dst + warp_offset, warp_src);
            }
        }
    }

    __syncthreads();
    if (handshake && tid == 0) {
        __threadfence_block();
        const int finished = atomicAdd(local_semaphore + dst_rank, 1);
        if (finished == num_blocks_for_rank - 1) {
            __threadfence_block();
            // Their source fences the block only. A block fence does not
            // publish these stores to the peer GPU, so the flag is released
            // at system scope after the int4 writes.
            __threadfence_system();
            int* remote = reinterpret_cast<int*>(signal_ptr[dst_rank]) + local_rank;
            st_release_sys(remote, 1);
            int* mine = reinterpret_cast<int*>(signal_ptr[local_rank]) + dst_rank;
            while (ld_acquire_sys(mine) == 0) {
            }
            *mine = 0;
            __threadfence_system();
        }
    }
}

void hao_basic_launch(torch::Tensor x,
                      torch::Tensor buffer_ptrs,
                      torch::Tensor signal_ptrs,
                      torch::Tensor semaphore,
                      int64_t rank,
                      int64_t world,
                      int64_t batch_size,
                      int64_t max_batch,
                      int64_t hidden,
                      int64_t is_transpose,
                      int64_t handshake,
                      torch::Tensor mask,
                      torch::Tensor dst_rows,
                      int64_t has_mask,
                      int64_t has_dst_rows) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "hao_basic source must be contiguous CUDA");
    TORCH_CHECK(buffer_ptrs.is_cuda() && signal_ptrs.is_cuda(), "pointer tables must be CUDA");
    TORCH_CHECK(semaphore.is_cuda(), "semaphore must be CUDA");
    TORCH_CHECK(world > 0 && rank >= 0 && rank < world, "rank outside world");
    TORCH_CHECK(batch_size >= 0 && hidden > 0, "batch and hidden must be valid");
    const int64_t row_bytes = hidden * x.element_size();
    TORCH_CHECK(row_bytes % 16 == 0, "row bytes must be a multiple of 16");

    constexpr int num_warps = 16;
    int blocks_per_rank = 16;
    if (blocks_per_rank > batch_size) {
        blocks_per_rank = std::max<int64_t>(1, batch_size);
    }
    const int grid = blocks_per_rank * static_cast<int>(world);
    const int block = num_warps * 32;
    auto stream = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(semaphore.data_ptr<int>(), 0, world * sizeof(int), stream));

    const int* mask_ptr = has_mask ? mask.data_ptr<int>() : nullptr;
    const int* row_ptr = has_dst_rows ? dst_rows.data_ptr<int>() : nullptr;
    auto* buffers = reinterpret_cast<void**>(buffer_ptrs.data_ptr<int64_t>());
    auto* signals = reinterpret_cast<void**>(signal_ptrs.data_ptr<int64_t>());
    if (has_dst_rows) {
        intranode_alltoall_kernel<num_warps, true><<<grid, block, 0, stream>>>(
            x.data_ptr(), buffers, signals, semaphore.data_ptr<int>(),
            static_cast<int>(rank), static_cast<int>(world), static_cast<int>(batch_size),
            static_cast<int>(max_batch), static_cast<int>(hidden), static_cast<int>(x.element_size()),
            static_cast<int>(is_transpose), static_cast<int>(handshake), nullptr, row_ptr);
    } else {
        intranode_alltoall_kernel<num_warps, false><<<grid, block, 0, stream>>>(
            x.data_ptr(), buffers, signals, semaphore.data_ptr<int>(),
            static_cast<int>(rank), static_cast<int>(world), static_cast<int>(batch_size),
            static_cast<int>(max_batch), static_cast<int>(hidden), static_cast<int>(x.element_size()),
            static_cast<int>(is_transpose), static_cast<int>(handshake), mask_ptr, nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="hao_basic_intranode",
            cpp_sources=[_CPP],
            cuda_sources=[_SRC],
            functions=["hao_basic_launch"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def _resident(cache: dict, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Upload a route table once. The bench reuses the same CPU table every step."""
    key = ("table", int(tensor.data_ptr()), tuple(tensor.shape), device.type, device.index)
    hit = cache.get(key)
    if hit is not None:
        return hit
    host = tensor if tensor.is_cuda or tensor.is_pinned() else tensor.pin_memory()
    uploaded = host.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    cache[key] = uploaded
    cache[("keep", key)] = host
    return uploaded


def _empty_i32(cache: dict, device: torch.device) -> torch.Tensor:
    key = ("empty", device.type, device.index)
    hit = cache.get(key)
    if hit is not None:
        return hit
    hit = torch.empty(0, dtype=torch.int32, device=device)
    cache[key] = hit
    return hit


def _ptrs(cache: dict, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Device pointer table. Built once, with a stream-ordered copy.

    ``torch.tensor(..., device=cuda)`` copies through a synchronizing
    memcpy. The next channel's table would then wait for the kernel that
    is already spinning on the peer flag, and that peer has not launched.
    """
    key = ("ptrs", tuple(int(t.data_ptr()) for t in tensors), device.index)
    hit = cache.get(key)
    if hit is not None:
        return hit
    host = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)
    if device.type == "cuda":
        host = host.pin_memory()
        table = torch.empty(host.shape, dtype=torch.int64, device=device)
        table.copy_(host, non_blocking=True)
    else:
        table = host
    cache[key] = table
    cache[("ptrhost", key)] = host
    return table


def prepare_tables(buffers: list[torch.Tensor], flags: list[torch.Tensor],
                   cache: dict, device: torch.device) -> None:
    """Upload pointer tables before any rank enters the handshake."""
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    _ptrs(cache, buffers, device)
    _ptrs(cache, flags, device)
    _empty_i32(cache, device)
    torch.cuda.current_stream(device).synchronize()


def hao_launch(
    src: torch.Tensor,
    dst_buffers: list[torch.Tensor],
    flag_tensors: list[torch.Tensor],
    semaphore: torch.Tensor,
    rank: int,
    world: int,
    batch_size: int,
    max_batch: int,
    is_transpose: bool,
    mask: torch.Tensor | None,
    dst_row_indices: torch.Tensor | None,
    cache: dict,
    handshake: bool,
) -> list[torch.Tensor]:
    """Queue one rank's Basic all-to-all on the current stream.

    ``mask`` writes ``[rank, token]`` in the destination buffer. That layout
    matches ``[world, max_batch]`` only when ``batch_size == max_batch``.
    ``dst_row_indices`` is a flat row in the destination buffer, stride
    ``max_batch``, and negative rows are skipped. The two are exclusive.
    ``batch_size == 0`` only enters the handshake.
    """
    if mask is not None and dst_row_indices is not None:
        raise ValueError("hao_basic takes mask or dst_row_indices, not both")
    if is_transpose and dst_row_indices is not None:
        raise ValueError("dst_row_indices is not a transpose send")
    if mask is not None and batch_size != max_batch:
        raise ValueError(
            f"mask batch {batch_size} != max_batch {max_batch}; the Basic kernel strides by batch"
        )
    if dst_row_indices is not None and tuple(dst_row_indices.shape) != (world, max_batch):
        raise ValueError(
            f"dst_row_indices {tuple(dst_row_indices.shape)} != {(world, max_batch)}"
        )
    if src.shape[0] == 0:
        hidden = 1
        for extent in src.shape[1:]:
            hidden *= extent
        flat = src.reshape(0, hidden)
    else:
        flat = src.reshape(src.shape[0], -1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if batch_size == 0:
        if flat.shape[0] != 0:
            raise ValueError("an empty all-to-all needs a zero-row source")
    elif is_transpose:
        if flat.shape[0] != world * batch_size:
            raise ValueError(
                f"transpose source rows {flat.shape[0]} != {world * batch_size}"
            )
    elif flat.shape[0] != batch_size:
        raise ValueError(f"source rows {flat.shape[0]} != batch {batch_size}")
    row_bytes = flat.shape[1] * flat.element_size()
    if row_bytes % 16 != 0:
        raise ValueError(f"hao_basic row is {row_bytes} bytes; need a multiple of 16")
    device = flat.device
    torch.cuda.set_device(device)
    empty = _empty_i32(cache, device)
    mask_gpu = _resident(cache, mask, device) if mask is not None else empty
    rows_gpu = _resident(cache, dst_row_indices, device) if dst_row_indices is not None else empty
    dst_ptrs = _ptrs(cache, dst_buffers, device)
    flag_ptrs = _ptrs(cache, flag_tensors, device)
    _ext().hao_basic_launch(
        flat, dst_ptrs, flag_ptrs, semaphore,
        int(rank), int(world), int(batch_size), int(max_batch), int(flat.shape[1]),
        int(is_transpose), int(handshake),
        mask_gpu, rows_gpu, int(mask is not None), int(dst_row_indices is not None),
    )
    return [flat, mask_gpu, rows_gpu, dst_ptrs, flag_ptrs, semaphore]
