"""Host-side checks for the LSE merge. No FA2, no head-TP."""

from __future__ import annotations

import torch

from layout_cp import page_split
from merge_cp import combine_kv, merge_cp_decode, merge_partials, normalize_lse_to_head_major


def _merge(o0, lse0, o1, lse1):
    out = torch.empty_like(o0)
    out_lse = torch.empty(lse0.shape[0], lse0.shape[1], dtype=torch.float32, device=o0.device)
    # Tests pass [H, B] already when constructed that way.
    merge_cp_decode(out, o0, lse0, o1, lse1, out_lse)
    return out, out_lse


def test_one_side_dominates():
    b, h, d = 2, 4, 8
    o0 = torch.randn(b, h, d)
    o1 = torch.randn(b, h, d)
    lse0 = torch.zeros(h, b)
    lse1 = torch.full((h, b), float("-inf"))
    out, lse = _merge(o0, lse0, o1, lse1)
    assert torch.allclose(out.float(), o0.float(), atol=1e-5)
    assert torch.allclose(lse, lse0)


def test_equal_lse_averages():
    b, h, d = 1, 2, 4
    o0 = torch.ones(b, h, d)
    o1 = torch.full((b, h, d), 3.0)
    lse0 = torch.zeros(h, b)
    lse1 = torch.zeros(h, b)
    out, lse = _merge(o0, lse0, o1, lse1)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5)
    assert torch.allclose(lse, torch.full_like(lse, 0.693147), atol=1e-4)


def test_positive_inf_is_treated_as_empty():
    b, h, d = 1, 1, 4
    o0 = torch.arange(d, dtype=torch.float32).view(1, 1, d)
    o1 = torch.zeros(1, 1, d)
    lse0 = torch.zeros(h, b)
    lse1 = torch.full((h, b), float("inf"))
    out, _ = _merge(o0, lse0, o1, lse1)
    assert torch.allclose(out.float(), o0, atol=1e-5)


def test_both_empty_is_zero():
    b, h, d = 1, 2, 4
    o0 = torch.ones(b, h, d)
    o1 = torch.ones(b, h, d)
    lse0 = torch.full((h, b), float("-inf"))
    lse1 = torch.full((h, b), float("inf"))
    out, lse = _merge(o0, lse0, o1, lse1)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


def test_batch_major_lse_is_accepted():
    raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [B=2, H=2]
    got = normalize_lse_to_head_major(raw, num_heads=2, batch=2)
    assert got.shape == (2, 2)
    assert torch.allclose(got, raw.transpose(0, 1))


def _fp64(o0, lse0, o1, lse1):
    """Independent formula, not a call into merge_cp."""
    a = torch.where(torch.isinf(lse0) & (lse0 > 0), torch.full_like(lse0, float("-inf")), lse0).double()
    b = torch.where(torch.isinf(lse1) & (lse1 > 0), torch.full_like(lse1, float("-inf")), lse1).double()
    maximum = torch.maximum(a, b)
    both = maximum == float("-inf")
    safe = torch.where(both, torch.zeros_like(maximum), maximum)
    wa = torch.where(both, torch.zeros_like(a), torch.exp(a - safe))
    wb = torch.where(both, torch.zeros_like(b), torch.exp(b - safe))
    denom = wa + wb
    scale_a = torch.where(both, torch.zeros_like(wa), wa / denom).transpose(0, 1).unsqueeze(-1)
    scale_b = torch.where(both, torch.zeros_like(wb), wb / denom).transpose(0, 1).unsqueeze(-1)
    out = scale_a * o0.double() + scale_b * o1.double()
    out = torch.where(both.transpose(0, 1).unsqueeze(-1), torch.zeros_like(out), out)
    return out


def test_matches_independent_fp64():
    b, h, d = 3, 4, 8
    gen = torch.Generator().manual_seed(7)
    o0 = torch.randn(b, h, d, generator=gen)
    o1 = torch.randn(b, h, d, generator=gen)
    lse0 = torch.randn(h, b, generator=gen)
    lse1 = torch.randn(h, b, generator=gen)
    lse1[0, 1] = float("-inf")
    lse0[1, 2] = float("inf")
    out, _ = _merge(o0, lse0, o1, lse1)
    ref = _fp64(o0, lse0, o1, lse1)
    assert torch.allclose(out.double(), ref, atol=1e-6, rtol=1e-6)


def test_three_partials_match_pairwise():
    b, h, d = 2, 3, 4
    gen = torch.Generator().manual_seed(3)
    parts = [torch.randn(b, h, d, generator=gen) for _ in range(3)]
    lses = [torch.randn(h, b, generator=gen) for _ in range(3)]
    lses[2][:, 0] = float("-inf")
    out = torch.empty_like(parts[0])
    out_lse = torch.empty(h, b)
    o_stack = torch.stack(parts, dim=0)
    lse_stack = torch.stack(lses, dim=0)
    merge_partials(out, out_lse, o_stack, lse_stack)
    acc, acc_lse = _merge(parts[0], lses[0], parts[1], lses[1])
    acc, acc_lse = _merge(acc, acc_lse, parts[2], lses[2])
    assert torch.allclose(out.float(), acc.float(), atol=1e-5)
    assert torch.allclose(out_lse, acc_lse, atol=1e-5)


def test_combine_skips_a_zero_length_rank():
    # Token 0 is on both ranks. Token 1 is only on rank 1, so rank 0 is skipped.
    o_stack = torch.zeros(2, 2, 2, 4)
    o_stack[0] = 1
    o_stack[1] = 3
    lse_stack = torch.zeros(2, 2, 2, 1)
    lens = torch.tensor([[4, 0], [4, 5]], dtype=torch.int32)
    out = torch.empty(2, 2, 4)
    combine_kv(out, o_stack, lse_stack, lens)
    assert torch.allclose(out[0].float(), torch.full((2, 4), 2.0), atol=1e-5)
    assert torch.allclose(out[1].float(), torch.full((2, 4), 3.0), atol=1e-5)


def test_page_split_stops_on_a_block():
    assert page_split(64) == (32, 32)
    assert page_split(96) == (48, 48)
    assert page_split(100) == (48, 52)
    assert page_split(17) == (0, 17)


def main():
    test_one_side_dominates()
    test_equal_lse_averages()
    test_positive_inf_is_treated_as_empty()
    test_both_empty_is_zero()
    test_batch_major_lse_is_accepted()
    test_matches_independent_fp64()
    test_three_partials_match_pairwise()
    test_combine_skips_a_zero_length_rank()
    test_page_split_stops_on_a_block()
    print("merge unit ok", flush=True)


if __name__ == "__main__":
    main()
