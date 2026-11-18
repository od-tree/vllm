"""Use the engine's CPU tile planner for each rank's actual paged layout."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# This benchmark also runs with only the FA2 extension installed.
def _load_planner():
    root = Path(os.environ.get("VLLM_ROOT", Path(__file__).resolve().parents[3] / "vllm"))
    path = root / "vllm/v1/decode_tiles.py"
    spec = importlib.util.spec_from_file_location("decode_tiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_planner = _load_planner()
pack_tile = _planner.pack_head_tp_tile


def plan_tiles(lengths, n_long, num_sms, num_kv_heads=2, *, first_head=0, long_heads=None):
    if not 0 <= n_long <= len(lengths):
        raise ValueError("invalid number of cooperative rows")
    count = num_kv_heads if long_heads is None else long_heads
    ranges = [(row, first_head, first_head + count) for row in range(n_long)]
    return _planner.plan_decode_tiles(
        lengths, ranges, list(range(n_long, len(lengths))), num_kv_heads, num_sms,
    )


def device_tiles(lengths, n_long, first_head, num_sms, device, *, long_heads=1):
    import torch

    tiles, splits, maximum = plan_tiles(
        lengths, n_long, num_sms, first_head=first_head, long_heads=long_heads,
    )
    return (torch.tensor(tiles, dtype=torch.int32, device=device),
            torch.tensor(splits, dtype=torch.int8, device=device), maximum)


def segment_tiles(segment):
    """CUDA tile plan for ``segment``, built once per device.

    ``None`` means this batch should use FA2's own split heuristic.
    """

    import torch

    device = segment.k.device
    if device.type != "cuda" or segment.q.shape[0] == 0 or not segment.lengths:
        return None
    cached = getattr(segment, "_tiles", None)
    if cached is not None and cached[0].device == device:
        return cached
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    uploaded = device_tiles(segment.lengths, segment.n_long, 0, num_sms, device, long_heads=2)
    segment._tiles = uploaded
    return uploaded
