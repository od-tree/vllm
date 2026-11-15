"""Indexed batch copy used before NanoCP's return all-to-all.

Same kernels as ``nanodeploy/kernels/copy.py``: one autotuned kernel for
``D > 1`` (partial O) and a scalar kernel for ``D == 1`` (LSE).
"""

from __future__ import annotations

import torch

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_D": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_D": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_D": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        ],
        key=["D", "H"],
    )
    @triton.jit
    def _copy_d(
        src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr,
        B_src, B_dst, H, D, M,
        stride_b_src, stride_h_src, stride_d_src,
        stride_b_dst, stride_h_dst, stride_d_dst,
        dtype: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
        dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)
        row_mask = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)
        active = (
            (row_mask == 1) & (src_idx >= 0) & (src_idx < B_src)
            & (dst_idx >= 0) & (dst_idx < B_dst)
        )
        valid = mask_m & active
        src_base = (src_idx * stride_b_src + pid_h * stride_h_src)[:, None]
        dst_base = (dst_idx * stride_b_dst + pid_h * stride_h_dst)[:, None]
        full = D // BLOCK_D
        for block in range(full):
            offs_d = block * BLOCK_D + tl.arange(0, BLOCK_D)
            src = src_ptr + src_base + offs_d[None, :] * stride_d_src
            dst = dst_ptr + dst_base + offs_d[None, :] * stride_d_dst
            take = valid[:, None]
            tl.store(dst, tl.load(src, mask=take, other=0.0), mask=take)
        if full * BLOCK_D < D:
            offs_d = full * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            src = src_ptr + src_base + offs_d[None, :] * stride_d_src
            dst = dst_ptr + dst_base + offs_d[None, :] * stride_d_dst
            take = valid[:, None] & mask_d[None, :]
            tl.store(dst, tl.load(src, mask=take, other=0.0), mask=take)

    @triton.jit
    def _copy_d1(
        src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr,
        B_src, B_dst, H, M,
        stride_b_src, stride_h_src, stride_d_src,
        stride_b_dst, stride_h_dst, stride_d_dst,
        dtype: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % grid_m
        pid_h = pid // grid_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
        dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)
        row_mask = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)
        valid = (
            mask_m & (row_mask == 1) & (src_idx >= 0) & (src_idx < B_src)
            & (dst_idx >= 0) & (dst_idx < B_dst)
        )
        src_off = src_idx * stride_b_src + pid_h * stride_h_src
        dst_off = dst_idx * stride_b_dst + pid_h * stride_h_dst
        tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off, mask=valid, other=0.0), mask=valid)

    _KERNELS = (_copy_d, _copy_d1, tl)
    return _KERNELS


def copy_batch_indexed_triton(src, dst, src_idx, dst_idx, mask) -> None:
    """Copy ``src[src_idx]`` to ``dst[dst_idx]`` where ``mask == 1``. Both are ``[rows, H, D]``."""
    copy_d, copy_d1, tl = _kernels()
    if src.ndim != 3 or dst.ndim != 3:
        raise ValueError(f"indexed copy expects [rows, H, D], got {tuple(src.shape)} -> {tuple(dst.shape)}")
    b_src, heads, width = src.shape
    b_dst = dst.shape[0]
    count = src_idx.numel()
    triton_dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}.get(
        src.dtype, tl.float32,
    )
    if width == 1:
        block_m = 128
        grid = (triton_cdiv(count, block_m) * heads,)
        copy_d1[grid](
            src, dst, src_idx, dst_idx, mask,
            b_src, b_dst, heads, count,
            src.stride(0), src.stride(1), src.stride(2),
            dst.stride(0), dst.stride(1), dst.stride(2),
            dtype=triton_dtype, BLOCK_M=block_m,
        )
        return

    def grid(meta):
        return (triton_cdiv(count, meta["BLOCK_M"]), heads)

    copy_d[grid](
        src, dst, src_idx, dst_idx, mask,
        b_src, b_dst, heads, width, count,
        src.stride(0), src.stride(1), src.stride(2),
        dst.stride(0), dst.stride(1), dst.stride(2),
        dtype=triton_dtype,
    )


def triton_cdiv(count: int, block: int) -> int:
    return (count + block - 1) // block
