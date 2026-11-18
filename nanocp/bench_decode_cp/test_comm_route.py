"""CPU checks for the N-rank route. No FA2 and no peer access."""

from __future__ import annotations

import torch

from comm_cp import CpComm, mask_from_kv_ranks, return_mask


def _comm(world=4, max_batch=8, width=2):
    return CpComm(
        ["cpu"] * world, max_batch=max_batch, q_shape=(width,),
        lse_shape=(width,), q_dtype=torch.float32, o_dtype=torch.float32,
    )


def test_q_fans_out_by_mask():
    comm = _comm()
    q = torch.arange(3 * 2, dtype=torch.float32).view(3, 2)
    mask = mask_from_kv_ranks(4, [[0, 2], [1], [0, 1, 2, 3]])
    launch = comm.launch_q(0, q, mask=mask)
    assert torch.equal(comm.gather("q", 2, 0, launch.slots_at(2)), q[[0, 2]])
    assert torch.equal(comm.gather("q", 1, 0, launch.slots_at(1)), q[[1, 2]])
    assert torch.equal(comm.gather("q", 3, 0, launch.slots_at(3)), q[2:3])
    # Rank 0 is the source. Its rows stay in ``q``; the buffer is not the copy.
    assert torch.equal(launch.slots_at(0), torch.tensor([0, 2]))
    assert int(comm.arrived["q"][3][0]) == 1


def test_dst_row_indices_choose_the_slot():
    comm = _comm()
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    table = torch.full((4, 2), -1, dtype=torch.long)
    table[3, 0] = 5
    launch = comm.launch_q(1, q, dst_row_indices=table)
    assert torch.equal(comm.gather("q", 3, 1, launch.slots_at(3)), q[0:1])
    untouched = comm.buffers["q"][3][1][0]
    assert torch.count_nonzero(untouched) == 0
    try:
        comm.launch_q(1, q, mask=torch.ones(4, 2, dtype=torch.int32), dst_row_indices=table)
    except ValueError:
        return
    raise AssertionError("mask and dst_row_indices were both accepted")


def test_partial_returns_to_one_moe_rank():
    comm = _comm()
    partial = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    slots = torch.tensor([0, 2])
    comm.launch_partial("o", 2, partial, return_mask(4, 0, 2), slots)
    assert torch.equal(comm.gather("o", 0, 2, slots), partial)
    bad = torch.ones(4, 2, dtype=torch.int32)
    try:
        comm.launch_partial("o", 2, partial, bad, slots)
    except ValueError:
        return
    raise AssertionError("a partial with two MoE destinations was accepted")


def test_active_rows_pack_into_one_slice():
    from comm_cp import _active_rows, _contiguous_bounds

    mask = torch.zeros(2, 6, dtype=torch.int32)
    mask[0] = 1
    mask[1, :2] = 1
    src, dst = _active_rows(1, 6, mask, None, None, False)
    assert _contiguous_bounds(src) == (0, 2)
    assert _contiguous_bounds(dst) == (0, 2)
    src, dst = _active_rows(0, 6, mask, None, None, True)
    assert _contiguous_bounds(src) == (0, 6)
    assert torch.equal(dst, torch.arange(6))
    # Partner return reads the moe plane at the same slots.
    src, dst = _active_rows(1, 6, mask, None, None, True)
    assert _contiguous_bounds(src) == (6, 2)
    assert _contiguous_bounds(dst) == (0, 2)
    slots = torch.tensor([0, 2])
    partial = torch.zeros(2, 2, dtype=torch.int32)
    partial[0] = 1
    src, dst = _active_rows(0, 2, partial, None, slots, False)
    assert torch.equal(src, torch.tensor([0, 1]))
    assert torch.equal(dst, slots)
    assert _contiguous_bounds(dst) is None
    table = torch.full((2, 3), -1, dtype=torch.long)
    table[1, 0] = 5
    table[1, 2] = 1
    src, dst = _active_rows(1, 3, None, table, None, False)
    assert torch.equal(src, torch.tensor([0, 2]))
    assert torch.equal(dst, torch.tensor([5, 1]))


def main():
    test_q_fans_out_by_mask()
    test_dst_row_indices_choose_the_slot()
    test_partial_returns_to_one_moe_rank()
    test_active_rows_pack_into_one_slice()
    print("comm route ok", flush=True)


if __name__ == "__main__":
    main()
