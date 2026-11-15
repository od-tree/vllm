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


def main():
    test_q_fans_out_by_mask()
    test_dst_row_indices_choose_the_slot()
    test_partial_returns_to_one_moe_rank()
    print("comm route ok", flush=True)


if __name__ == "__main__":
    main()
