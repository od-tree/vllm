"""LSE merge for decode-CP partials.

This file does not import or modify the head-TP operator. The timed path
merges every rank in one Triton kernel. If Triton is not installed and there
are exactly two partials, it calls vLLM ``merge_attn_states``. Otherwise the
same formula runs in fp32 PyTorch:

    m   = max(lse_a, lse_b)
    a   = exp(lse - m)          # +inf LSE is rewritten to -inf first
    O   = (a_a / s) * O_a + (a_b / s) * O_b
    lse = m + log(s)

If both LSEs are empty, O is 0 and the merged LSE is -inf.
Prefix is the first half of the sequence. Suffix is the second half.
"""

from __future__ import annotations

import torch

_vllm_merge = None
_vllm_checked = False


def _merge_from_vllm():
    global _vllm_merge, _vllm_checked
    if _vllm_checked:
        return _vllm_merge
    _vllm_checked = True
    try:
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
    except ImportError:
        _vllm_merge = None
    else:
        _vllm_merge = merge_attn_states
    return _vllm_merge


def normalize_lse_to_head_major(lse: torch.Tensor, num_heads: int, batch: int) -> torch.Tensor:
    """Return LSE as ``[num_heads, batch]`` float32."""
    if lse is None:
        raise RuntimeError("FA2 did not return an LSE")
    if lse.ndim == 2 and tuple(lse.shape) == (num_heads, batch):
        out = lse
    elif lse.ndim == 2 and tuple(lse.shape) == (batch, num_heads):
        out = lse.transpose(0, 1)
    elif lse.numel() == num_heads * batch:
        # FA2's GQA-decode return is head-major [H, B] after the host reshape.
        out = lse.reshape(num_heads, batch)
    else:
        raise RuntimeError(
            f"LSE shape {tuple(lse.shape)} is not [H, B]=[{num_heads}, {batch}] or [B, H]"
        )
    return out.float().contiguous()


def _negative_inf_only(lse: torch.Tensor) -> torch.Tensor:
    positive_inf = torch.isinf(lse) & (lse > 0)
    return torch.where(positive_inf, torch.full_like(lse, float("-inf")), lse)


def _merge_torch(out, o0, lse0, o1, lse1, out_lse) -> None:
    """Match ``_merge_attn_states_torch`` with every token merged."""
    maximum = torch.maximum(lse0, lse1)
    both_empty = maximum == float("-inf")
    maximum_safe = torch.where(both_empty, torch.zeros_like(maximum), maximum)
    weight0 = torch.exp(lse0 - maximum_safe)
    weight1 = torch.exp(lse1 - maximum_safe)
    weight0 = torch.where(both_empty, torch.zeros_like(weight0), weight0)
    weight1 = torch.where(both_empty, torch.zeros_like(weight1), weight1)
    denom = weight0 + weight1
    scale0 = torch.where(both_empty, torch.zeros_like(weight0), weight0 / denom)
    scale1 = torch.where(both_empty, torch.zeros_like(weight1), weight1 / denom)
    # [H, B] -> [B, H, 1]
    scale0 = scale0.transpose(0, 1).unsqueeze(-1)
    scale1 = scale1.transpose(0, 1).unsqueeze(-1)
    merged = scale0 * o0.float() + scale1 * o1.float()
    empty = both_empty.transpose(0, 1).unsqueeze(-1)
    merged = torch.where(empty, torch.zeros_like(merged), merged)
    out.copy_(merged.to(dtype=out.dtype))
    if out_lse is not None:
        merged_lse = maximum + torch.log(denom.clamp_min(torch.finfo(torch.float32).tiny))
        merged_lse = torch.where(both_empty, torch.full_like(merged_lse, float("-inf")), merged_lse)
        if tuple(out_lse.shape) != tuple(merged_lse.shape):
            raise ValueError(
                f"out_lse shape {tuple(out_lse.shape)} != [H, B] {tuple(merged_lse.shape)}"
            )
        out_lse.copy_(merged_lse)


_MAX_PARTIALS = 8


def _merge_torch_n(out, out_lse, o_stack, lse_stack) -> None:
    """N partials. Absent ranks are ``-inf`` in ``lse_stack`` and contribute nothing."""
    lse = _negative_inf_only(lse_stack.float())
    maximum = torch.amax(lse, dim=0)
    both_empty = maximum == float("-inf")
    maximum_safe = torch.where(both_empty, torch.zeros_like(maximum), maximum)
    weight = torch.exp(lse - maximum_safe.unsqueeze(0))
    weight = torch.where(both_empty.unsqueeze(0), torch.zeros_like(weight), weight)
    denom = weight.sum(dim=0)
    scale = torch.where(both_empty, torch.zeros_like(denom), weight / denom.unsqueeze(0))
    scale = scale.permute(0, 2, 1).unsqueeze(-1)
    merged = (scale * o_stack.float()).sum(dim=0)
    empty = both_empty.transpose(0, 1).unsqueeze(-1)
    merged = torch.where(empty, torch.zeros_like(merged), merged)
    out.copy_(merged.to(dtype=out.dtype))
    if out_lse is not None:
        merged_lse = maximum + torch.log(denom.clamp_min(torch.finfo(torch.float32).tiny))
        merged_lse = torch.where(both_empty, torch.full_like(merged_lse, float("-inf")), merged_lse)
        out_lse.copy_(merged_lse)


_TRITON_KERNEL = None


def _triton_merge_kernel():
    """Compile the N-way kernel once. Importing Triton stays inside this call."""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        o_ptr, lse_ptr, out_ptr, out_lse_ptr,
        world, batch, heads, dim,
        stride_ow, stride_ob, stride_oh, stride_od,
        stride_lw, stride_lh, stride_lb,
        stride_out_b, stride_out_h, stride_out_d,
        stride_lse_h, stride_lse_b,
        BLOCK: tl.constexpr,
        MAX_W: tl.constexpr,
        STORE_BF16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        token = pid // heads
        head = pid % heads
        w_ids = tl.arange(0, MAX_W)
        lse_off = w_ids * stride_lw + head * stride_lh + token * stride_lb
        lse = tl.load(lse_ptr + lse_off, mask=w_ids < world, other=float("-inf"))
        lse = tl.where(lse == float("inf"), float("-inf"), lse)
        maximum = tl.max(lse, axis=0)
        empty = maximum == float("-inf")
        maximum_safe = tl.where(empty, 0.0, maximum)
        weight = tl.where(w_ids < world, tl.exp(lse - maximum_safe), 0.0)
        weight = tl.where(empty, 0.0, weight)
        denom = tl.sum(weight, axis=0)
        scale = weight / tl.maximum(denom, 1.0e-20)
        scale = tl.where(empty, 0.0, scale)
        for block_id in tl.static_range(4):
            d_ids = block_id * BLOCK + tl.arange(0, BLOCK)
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for part in tl.static_range(MAX_W):
                part_scale = tl.sum(tl.where(w_ids == part, scale, 0.0), axis=0)
                o_off = (
                    part * stride_ow + token * stride_ob + head * stride_oh + d_ids * stride_od
                )
                values = tl.load(
                    o_ptr + o_off, mask=(part < world) & (d_ids < dim), other=0.0,
                ).to(tl.float32)
                acc += part_scale * values
            acc = tl.where(empty, 0.0, acc)
            stored = acc.to(tl.bfloat16) if STORE_BF16 else acc
            tl.store(
                out_ptr + token * stride_out_b + head * stride_out_h + d_ids * stride_out_d,
                stored, mask=d_ids < dim,
            )
        merged_lse = tl.where(
            empty, float("-inf"), maximum + tl.log(tl.maximum(denom, 1.0e-20)),
        )
        tl.store(out_lse_ptr + head * stride_lse_h + token * stride_lse_b, merged_lse)

    _TRITON_KERNEL = _kernel
    return _kernel


def _merge_triton(out, out_lse, o_stack, lse_stack) -> None:
    world, batch, heads, dim = o_stack.shape
    if world > _MAX_PARTIALS:
        raise ValueError(f"merge supports at most {_MAX_PARTIALS} partials, got {world}")
    if dim > 512:
        raise ValueError(f"merge kernel covers head dim up to 512, got {dim}")
    if batch == 0:
        return
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"merge kernel supports bf16 or fp32 output, got {out.dtype}")
    block = 128 if dim >= 128 else 64
    _triton_merge_kernel()[(batch * heads,)](
        o_stack, lse_stack, out, out_lse,
        world, batch, heads, dim,
        o_stack.stride(0), o_stack.stride(1), o_stack.stride(2), o_stack.stride(3),
        lse_stack.stride(0), lse_stack.stride(1), lse_stack.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        out_lse.stride(0), out_lse.stride(1),
        BLOCK=block, MAX_W=_MAX_PARTIALS, STORE_BF16=out.dtype == torch.bfloat16,
    )


def merge_partials(out, out_lse, o_stack, lse_stack) -> None:
    """Merge ``o_stack [W, B, H, D]`` and ``lse_stack [W, H, B]``.

    A rank that does not hold a row keeps LSE ``-inf``. CUDA uses one Triton
    kernel. Without Triton, two partials go through vLLM ``merge_attn_states``
    when that package is installed. CPU uses the same formula in PyTorch.
    """
    if o_stack.shape[0] != lse_stack.shape[0] or o_stack.shape[0] < 1:
        raise ValueError(f"partial count mismatch: O {tuple(o_stack.shape)} LSE {tuple(lse_stack.shape)}")
    if out_lse is None:
        raise ValueError("merge_partials writes the merged LSE")
    o_stack = o_stack.contiguous()
    lse_stack = lse_stack.contiguous()
    if out.is_cuda:
        try:
            _merge_triton(out, out_lse, o_stack, lse_stack)
            return
        except ImportError:
            if o_stack.shape[0] == 2:
                merge_cp_decode(
                    out, o_stack[0], lse_stack[0], o_stack[1], lse_stack[1], out_lse,
                )
                return
    _merge_torch_n(out, out_lse, o_stack, lse_stack)


def combine_kv(out: torch.Tensor, o_stack: torch.Tensor, lse_stack: torch.Tensor,
               context_lens: torch.Tensor) -> None:
    """Merge like ``inter_rank_gqa_fwd_batch_decode_combine_kv``.

    ``o_stack`` is ``[W, B, H, D]`` and ``lse_stack`` is ``[W, B, H]`` or
    ``[W, B, H, 1]``, already in the transpose-return buffer. ``context_lens``
    is ``[W, B]`` int32; a non-positive length skips that rank. The kernel
    writes O only. It does not write a merged LSE.
    """
    if o_stack.ndim != 4:
        raise ValueError(f"O stack must be [W, B, H, D], got {tuple(o_stack.shape)}")
    if context_lens.shape != o_stack.shape[:2]:
        raise ValueError(
            f"context_lens {tuple(context_lens.shape)} != [W, B] {tuple(o_stack.shape[:2])}"
        )
    if lse_stack.shape[:3] != o_stack.shape[:3]:
        raise ValueError(f"LSE stack {tuple(lse_stack.shape)} does not match O {tuple(o_stack.shape)}")
    if out.shape != o_stack.shape[1:]:
        raise ValueError(f"out {tuple(out.shape)} != [B, H, D] {tuple(o_stack.shape[1:])}")
    if o_stack.shape[1] == 0:
        return
    if o_stack.stride(-1) != 1:
        o_stack = o_stack.contiguous()
    if lse_stack.stride(-1) != 1:
        lse_stack = lse_stack.contiguous()
    context_lens = context_lens.to(dtype=torch.int32).contiguous()
    if out.is_cuda:
        try:
            _combine_triton(out, o_stack, lse_stack, context_lens)
            return
        except ImportError:
            pass
    _combine_online(out, o_stack, lse_stack, context_lens)


def _combine_online(out, o_stack, lse_stack, context_lens) -> None:
    """Same online update as the Triton kernel, for CPU and for a missing Triton."""
    world, batch, heads, _ = o_stack.shape
    lse = lse_stack[..., 0].float() if lse_stack.ndim == 4 else lse_stack.float()
    acc = torch.zeros(batch, heads, o_stack.shape[-1], dtype=torch.float32, device=out.device)
    e_sum = torch.zeros(batch, heads, dtype=torch.float32, device=out.device)
    e_max = torch.full((batch, heads), float("-inf"), dtype=torch.float32, device=out.device)
    for split in range(world):
        active = (context_lens[split] > 0).view(batch, 1)
        tv = o_stack[split].float()
        tlogic = lse[split].float()
        use_max = torch.where(active, torch.maximum(tlogic, e_max), e_max)
        old_scale = torch.where(active, torch.exp(e_max - use_max), torch.ones_like(e_max))
        exp_logic = torch.where(active, torch.exp(tlogic - use_max), torch.zeros_like(e_max))
        acc = acc * old_scale.unsqueeze(-1) + exp_logic.unsqueeze(-1) * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = use_max
    out.copy_((acc / e_sum.unsqueeze(-1)).to(dtype=out.dtype))


_COMBINE_KERNEL = None


def _combine_kernel():
    global _COMBINE_KERNEL
    if _COMBINE_KERNEL is not None:
        return _COMBINE_KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _kernel(
        mid_o, mid_lse, o_ptr, lens_ptr,
        batch,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lb, stride_lh, stride_ls,
        stride_out_b, stride_out_h,
        NUM_KV_SPLITS: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        LV: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < LV
        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
        offs_v = cur_batch * stride_ob + cur_head * stride_oh + offs_d * stride_od
        offs_lse = cur_batch * stride_lb + cur_head * stride_lh
        for split_kv_id in range(NUM_KV_SPLITS):
            effective_kv_len = tl.load(lens_ptr + cur_batch + split_kv_id * batch)
            if effective_kv_len > 0:
                tv = tl.load(
                    mid_o + offs_v + split_kv_id * stride_os, mask=mask_d, other=0.0,
                )
                tlogic = tl.load(mid_lse + offs_lse + split_kv_id * stride_ls)
                n_e_max = tl.maximum(tlogic, e_max)
                old_scale = libdevice.fast_expf(e_max - n_e_max)
                acc *= old_scale
                exp_logic = libdevice.fast_expf(tlogic - n_e_max)
                acc += exp_logic * tv
                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max
        # fp32 quotient stored through the output pointer. Triton narrows to bf16
        # when ``o`` is bf16, which is the store in their kernel.
        tl.store(
            o_ptr + cur_batch * stride_out_b + cur_head * stride_out_h + offs_d,
            acc / e_sum, mask=mask_d,
        )

    _COMBINE_KERNEL = _kernel
    return _kernel


def _combine_triton(out, o_stack, lse_stack, context_lens) -> None:
    world, batch, heads, dim = o_stack.shape
    if dim > 512:
        raise ValueError(f"combine kernel uses BLOCK_DV 512, got head dim {dim}")
    lse_rank = lse_stack.stride(0)
    lse_batch = lse_stack.stride(1)
    lse_head = lse_stack.stride(2)
    _combine_kernel()[(batch, heads)](
        o_stack, lse_stack, out, context_lens,
        batch,
        o_stack.stride(1), o_stack.stride(2), o_stack.stride(0), o_stack.stride(3),
        lse_batch, lse_head, lse_rank,
        out.stride(0), out.stride(1),
        NUM_KV_SPLITS=world, BLOCK_DV=512, LV=dim,
    )


def merge_cp_decode(
    out: torch.Tensor,
    o_prefix: torch.Tensor,
    lse_prefix: torch.Tensor,
    o_suffix: torch.Tensor,
    lse_suffix: torch.Tensor,
    out_lse: torch.Tensor | None = None,
) -> None:
    """Merge prefix/suffix partials into ``out`` ([B, H, D]).

    ``out_lse``, when given, is written as ``[H, B]`` float32.
    """
    if o_prefix.shape != o_suffix.shape or o_prefix.shape != out.shape:
        raise ValueError(
            f"output shapes differ: out {tuple(out.shape)} "
            f"prefix {tuple(o_prefix.shape)} suffix {tuple(o_suffix.shape)}"
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous [B, H, D]")
    o0 = o_prefix if o_prefix.is_contiguous() else o_prefix.contiguous()
    o1 = o_suffix if o_suffix.is_contiguous() else o_suffix.contiguous()
    if o0.stride(1) != o1.stride(1):
        o0 = o0.contiguous()
        o1 = o1.contiguous()
    batch, num_heads, _ = out.shape
    lse0 = _negative_inf_only(normalize_lse_to_head_major(lse_prefix, num_heads, batch))
    lse1 = _negative_inf_only(normalize_lse_to_head_major(lse_suffix, num_heads, batch))
    vllm_merge = _merge_from_vllm()
    if vllm_merge is not None:
        vllm_merge(
            out, o0, lse0, o1, lse1,
            output_lse=out_lse,
            prefill_tokens_with_context=None,
        )
        return
    _merge_torch(out, o0, lse0, o1, lse1, out_lse)
