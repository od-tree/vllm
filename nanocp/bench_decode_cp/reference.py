"""Single-card full-sequence reference for decode CP."""

from __future__ import annotations

from attn_local import attn_local


def reference_full(batch, num_splits: int = 0):
    return attn_local(batch.q, batch.k, batch.v, batch.table, batch.seqused, num_splits=num_splits)
