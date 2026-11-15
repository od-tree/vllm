"""Two-GPU decode CP through the routing path.

Master KV stays on GPU0 and includes short rows at full length plus long-row
prefixes, in one attention call. Partner KV is the long-row suffixes, placed
on GPU1 before the step. Short rows are mask 0 on the partner, so they are
not routed.
"""

from __future__ import annotations

from comm_cp import enable_peers
from decode_cp import run_routed
from layout_cp import build_segments, relocate_kv, segment_limits
from reference import reference_full

MAX_ABS = 1e-2


def _run(seqs, n_long, num_splits=0):
    full, master, partner, mask = build_segments(seqs, n_long, "cuda:0", seed=11)
    if partner.q.shape[0] != n_long:
        raise SystemExit(
            f"partner batch {partner.q.shape[0]} does not equal the long-row count {n_long}"
        )
    if int(mask[1, n_long:].sum()) != 0:
        raise SystemExit("short rows were marked for the partner")
    ref, _ = reference_full(full, num_splits=num_splits)
    partner = relocate_kv(partner, 1)
    master_lens, partner_lens = segment_limits(seqs, n_long)
    merged = run_routed(
        [master, partner], moe_rank=0, devices=[0, 1], mask=mask,
        num_splits=num_splits,
        max_ks=[max(master_lens), max(partner_lens)],
    )
    err = (merged.float() - ref.float()).abs()
    print(
        f"e2e {seqs} max_abs {float(err.max()):.6e} partner_batch {partner.q.shape[0]}",
        flush=True,
    )
    if float(err.max()) > MAX_ABS:
        raise SystemExit(f"two-gpu merge differs by {float(err.max()):.3e}")


def main():
    enable_peers([0, 1])
    _run([128], 1)
    _run([100], 1)
    _run([64, 96, 128, 17], 3)
    print("e2e ok", flush=True)


if __name__ == "__main__":
    main()
