"""Routing communication for decode CP.

Same contract as NanoCP's hao_basic all-to-all, for any number of ranks:

- Q-Route is one launch. A CPU mask ``[world, batch]`` or ``dst_row_indices``
  says which rows move. The payload is not transposed. The source rank is
  written into its own buffer as well; attention reads that buffer.
- Res-Route is two more launches, partial O then fp32 LSE. Each partial has
  exactly one MoE destination. ``slots[row]`` is the MoE row written at
  ``buffer[src, slot]``.
- Bytes land in a buffer allocated at init, ``[world, max_batch, ...]`` on the
  destination device. On CUDA each rank launches DLSlime's Basic kernel. The
  flag handshake is inside that kernel, so every rank has to enter it. CPU
  tests use the same placement with ``index_copy``.

KV is not moved. Masks stay on the CPU: the route is known when the batch is
built, and the device copy of a mask is kept after the first step.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import torch


def as_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def enable_peers(devices):
    """Enable peer access for every CUDA pair in ``devices``. One device is a no-op."""
    cuda_ids = []
    for device in devices:
        resolved = as_device(device)
        if resolved.type == "cuda":
            cuda_ids.append(resolved.index)
    if len(set(cuda_ids)) != len(cuda_ids):
        raise ValueError(f"duplicate CUDA devices in {cuda_ids}")
    if len(cuda_ids) < 2:
        return
    for src in cuda_ids:
        for dst in cuda_ids:
            if src != dst and not torch.cuda.can_device_access_peer(src, dst):
                raise SystemExit(f"cuda peer access {src}->{dst} is not available")
    lib = None
    for soname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError:
            continue
    if lib is None:
        raise SystemExit("libcudart was not found")
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    for src in cuda_ids:
        err = lib.cudaSetDevice(src)
        if err != 0:
            raise SystemExit(f"cudaSetDevice({src}) returned {err}")
        for dst in cuda_ids:
            if src == dst:
                continue
            err = lib.cudaDeviceEnablePeerAccess(dst, 0)
            if err not in (0, 704):
                raise SystemExit(f"peer access {src}->{dst} returned {err}")
    torch.cuda.set_device(cuda_ids[0])


def mask_from_kv_ranks(world: int, kv_ranks: list[list[int]]) -> torch.Tensor:
    """Q-Route table. ``mask[rank, row] == 1`` sends that row to ``rank``."""
    mask = torch.zeros(world, len(kv_ranks), dtype=torch.int32)
    for row, ranks in enumerate(kv_ranks):
        for rank in ranks:
            if rank < 0 or rank >= world:
                raise ValueError(f"kv rank {rank} is outside world {world}")
            mask[rank, row] = 1
    return mask


def return_mask(world: int, moe_rank: int, rows: int) -> torch.Tensor:
    """Res-Route table. Every partial row returns to the single MoE rank."""
    if moe_rank < 0 or moe_rank >= world:
        raise ValueError(f"moe rank {moe_rank} is outside world {world}")
    mask = torch.zeros(world, rows, dtype=torch.int32)
    if rows:
        mask[moe_rank] = 1
    return mask


@dataclass
class Transfer:
    src: int
    dst: int
    rows: torch.Tensor  # CPU long, indexes into the source batch
    slots: torch.Tensor  # CPU long, indexes into the destination buffer


@dataclass
class RouteLaunch:
    channel: str
    transfers: list[Transfer]
    src: int = 0
    epoch: int = 0
    written: set = field(default_factory=set)
    events: dict[int, torch.cuda.Event] = field(default_factory=dict)
    hold: list[torch.Tensor] = field(default_factory=list)

    def slots_at(self, dst: int) -> torch.Tensor | None:
        for transfer in self.transfers:
            if transfer.dst == dst:
                return transfer.slots
        return None


def _cpu_long(index: torch.Tensor) -> torch.Tensor:
    if index.device.type != "cpu":
        raise RuntimeError("routing tables must already be on CPU")
    return index.to(dtype=torch.long)


def _arange_covers(slots: torch.Tensor, rows: int) -> bool:
    if slots.numel() != rows:
        return False
    if rows == 0:
        return True
    expected = torch.arange(rows, dtype=torch.long)
    return torch.equal(slots, expected)


class CpComm:
    """Preallocated Q / O / LSE routes across ``devices``."""

    def __init__(self, devices, max_batch: int, q_shape, o_shape=None, lse_shape=(16,),
                 q_dtype=torch.bfloat16, o_dtype=None, lse_dtype=torch.float32):
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.devices = [as_device(device) for device in devices]
        if not self.devices:
            raise ValueError("devices must not be empty")
        self.world = len(self.devices)
        self.max_batch = max_batch
        self.shapes = {
            "q": tuple(q_shape),
            "o": tuple(o_shape if o_shape is not None else q_shape),
            "lse": tuple(lse_shape),
        }
        self.dtypes = {"q": q_dtype, "o": o_dtype if o_dtype is not None else q_dtype, "lse": lse_dtype}
        self.buffers = {
            name: [
                torch.zeros((self.world, max_batch, *shape), dtype=self.dtypes[name], device=device)
                for device in self.devices
            ]
            for name, shape in self.shapes.items()
        }
        self.arrived = {
            name: [torch.zeros(self.world, dtype=torch.int32, device=device) for device in self.devices]
            for name in self.shapes
        }
        # Per-rank block counters. The Basic kernel zeros this on every launch.
        self.semaphore = {
            name: [torch.zeros(self.world, dtype=torch.int32, device=device) for device in self.devices]
            for name in self.shapes
        }
        self.epoch = {name: 0 for name in self.shapes}
        self._gpu_tables = {}
        self._return_cache = {}
        self._dummies = {}
        # Persistent local_buffer. The return path fills this, then a fresh input buffer.
        self.staging = {
            name: [
                torch.empty((self.world * max_batch, *shape), dtype=self.dtypes[name], device=device)
                for device in self.devices
            ]
            for name, shape in self.shapes.items()
        }

    def launch_q(self, src: int, q: torch.Tensor, mask: torch.Tensor | None = None,
                 dst_row_indices: torch.Tensor | None = None, sync: bool = True) -> RouteLaunch:
        """Send Q. ``mask`` and ``dst_row_indices`` are mutually exclusive.

        ``mask[dst, row] != 0`` writes ``q[row]`` at slot ``row`` on ``dst``.
        ``dst_row_indices[dst, row] >= 0`` writes ``q[row]`` at that slot.
        A row may fan out to several ranks. On CUDA the source rank is written
        too, and the caller reads Q back from that buffer.
        """
        if mask is not None and dst_row_indices is not None:
            raise ValueError("Q route takes mask or dst_row_indices, not both")
        if mask is None and dst_row_indices is None:
            raise ValueError("Q route needs a mask or dst_row_indices")
        self._check_src(src, q)
        if mask is not None:
            transfers = self._from_mask(src, mask, q.shape[0], allow_fanout=True)
            for transfer in transfers:
                transfer.slots = transfer.rows
        else:
            transfers = self._from_dst_rows(src, dst_row_indices, q.shape[0])
        if q.is_cuda:
            launch = self._device_launch(
                "q", src, q, transfers, mask, dst_row_indices, None, is_transpose=False,
            )
            # Only the MoE rank has Q. The other ranks still have to enter the
            # kernel, or the handshake never finishes.
            self._participate_others("q", src)
            if sync:
                self._sync_cuda()
            return launch
        # CPU checks do not model the self-write. Callers keep using ``q``.
        remote = [transfer for transfer in transfers if transfer.dst != src]
        launch = self._launch("q", src, q, remote, sync)
        launch.transfers = transfers
        return launch

    def launch_partial(self, channel: str, src: int, value: torch.Tensor, mask: torch.Tensor,
                       slots: torch.Tensor, sync: bool = True) -> RouteLaunch:
        """Send one Res-Route payload. Exactly one destination per row."""
        if channel not in ("o", "lse"):
            raise ValueError(f"partial channel must be o or lse, got {channel}")
        self._check_src(src, value)
        transfers = self._from_mask(src, mask, value.shape[0], allow_fanout=False)
        slots = _cpu_long(slots)
        if slots.numel() != value.shape[0]:
            raise ValueError(
                f"partial slots {slots.numel()} != rows {value.shape[0]}"
            )
        for transfer in transfers:
            transfer.slots = slots.index_select(0, transfer.rows)
        if value.is_cuda:
            launch = self._device_launch(
                channel, src, value, transfers, mask, None, slots, is_transpose=False,
            )
            self._participate_others(channel, src)
            if sync:
                self._sync_cuda()
            return launch
        return self._launch(channel, src, value, transfers, sync)

    def wait(self, launch: RouteLaunch, dst: int) -> None:
        """No poll. The Basic kernel does not return until the peer has too.

        Later work on this rank's stream is already behind that kernel.
        """
        del launch, dst

    def buffer_rows(self, channel: str, dst: int, src: int, slots: torch.Tensor) -> torch.Tensor:
        """Rows of ``buffer[src]`` on ``dst``. A leading ``0..n-1`` is a view, not a gather."""
        table = self.buffers[channel][dst][src]
        slots = _cpu_long(slots)
        if slots.numel() and int(slots[0]) == 0 and _arange_covers(slots, slots.numel()):
            return table[: slots.numel()]
        return table.index_select(0, slots.to(table.device))

    def gather(self, channel: str, dst: int, src: int, slots: torch.Tensor,
               local: torch.Tensor | None = None) -> torch.Tensor:
        """Read rows that ``src`` wrote for ``dst``.

        Pass ``local`` for the Q that stayed on its source rank. That path
        does not read the receive buffer.
        """
        slots = _cpu_long(slots)
        if local is not None and dst == src and _arange_covers(slots, local.shape[0]):
            return local
        if local is not None and dst == src:
            return local.index_select(0, slots.to(local.device))
        buffer = self.buffers[channel][dst][src]
        return buffer.index_select(0, slots.to(buffer.device))

    def _check_src(self, src: int, value: torch.Tensor) -> None:
        if src < 0 or src >= self.world:
            raise ValueError(f"src {src} is outside world {self.world}")
        if value.device != self.devices[src]:
            raise ValueError(
                f"payload for rank {src} is on {value.device}, expected {self.devices[src]}"
            )
        tail = tuple(value.shape[1:])
        # Channel is checked by the caller via the buffer the launch writes.
        if value.shape[0] > self.max_batch:
            raise ValueError(f"batch {value.shape[0]} exceeds max_batch {self.max_batch}")
        del tail

    def _check_tail(self, channel: str, value: torch.Tensor) -> None:
        if tuple(value.shape[1:]) != self.shapes[channel]:
            raise ValueError(
                f"{channel} tail {tuple(value.shape[1:])} != {self.shapes[channel]}"
            )
        if value.dtype != self.dtypes[channel]:
            raise ValueError(f"{channel} dtype {value.dtype} != {self.dtypes[channel]}")

    def _from_mask(self, src: int, mask: torch.Tensor, rows: int, allow_fanout: bool) -> list[Transfer]:
        if mask.device.type != "cpu":
            raise RuntimeError("routing mask must already be on CPU")
        if tuple(mask.shape) != (self.world, rows):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != {(self.world, rows)}"
            )
        if not allow_fanout:
            counts = mask.sum(dim=0)
            if not torch.equal(counts, torch.ones(rows, dtype=counts.dtype)):
                raise ValueError("Res-Route requires exactly one MoE destination per row")
        transfers = []
        for dst in range(self.world):
            chosen = torch.nonzero(mask[dst] != 0, as_tuple=False).flatten().to(torch.long)
            if chosen.numel() == 0:
                continue
            transfers.append(Transfer(src, dst, chosen, chosen))
        return transfers

    def _from_dst_rows(self, src: int, dst_row_indices: torch.Tensor, rows: int) -> list[Transfer]:
        if dst_row_indices.device.type != "cpu":
            raise RuntimeError("dst_row_indices must already be on CPU")
        if tuple(dst_row_indices.shape) != (self.world, rows):
            raise ValueError(
                f"dst_row_indices shape {tuple(dst_row_indices.shape)} != {(self.world, rows)}"
            )
        table = dst_row_indices.to(dtype=torch.long)
        transfers = []
        for dst in range(self.world):
            chosen = torch.nonzero(table[dst] >= 0, as_tuple=False).flatten()
            if chosen.numel() == 0:
                continue
            slots = table[dst].index_select(0, chosen)
            if int(slots.max()) >= self.max_batch:
                raise ValueError(f"dst slot {int(slots.max())} exceeds max_batch {self.max_batch}")
            transfers.append(Transfer(src, dst, chosen, slots))
        return transfers

    def _launch(self, channel: str, src: int, value: torch.Tensor,
                transfers: list[Transfer], sync: bool) -> RouteLaunch:
        self._check_tail(channel, value)
        self.epoch[channel] += 1
        launch = RouteLaunch(
            channel, transfers, src=src, epoch=self.epoch[channel],
            written={transfer.dst for transfer in transfers},
        )
        for transfer in transfers:
            event, held = self._put(channel, value, transfer)
            launch.hold.extend(held)
            if event is not None:
                launch.events[transfer.dst] = event
            if self.devices[transfer.dst].type != "cuda":
                self.arrived[channel][transfer.dst][transfer.src] = self.epoch[channel]
        if sync:
            for event in launch.events.values():
                event.synchronize()
            for transfer in transfers:
                if self.devices[transfer.dst].type == "cuda":
                    self.arrived[channel][transfer.dst][transfer.src] = self.epoch[channel]
        return launch

    def launch_return(self, channel: str, src: int, packed: torch.Tensor, mask: torch.Tensor,
                      sync: bool = False) -> RouteLaunch:
        """Transpose all-to-all. ``packed`` is ``[world * max_batch, ...]`` on ``src``.

        ``mask[dst, row]`` sends ``packed[dst, row]`` to ``buffer[src, row]`` on ``dst``.
        ``sync`` is ignored: this kernel waits for the other ranks, so the
        caller has to queue those launches before any synchronize.
        """
        dests = [dst for dst in range(self.world) if mask[dst].numel() and int(mask[dst].max()) > 0]
        empty = torch.empty(0, dtype=torch.long)
        transfers = [Transfer(src, dst, empty, empty) for dst in dests]
        del sync
        return self._device_launch(
            channel, src, packed, transfers, mask, None, None, is_transpose=True,
        )

    def participate(self, channel: str, rank: int) -> None:
        """Enter the all-to-all with no rows.

        Ranks that do not launch a real send still have to run the kernel
        when the handshake is on, or the ranks that do send never leave it.
        One GPU does not handshake, so this is a no-op there.
        """
        if not self._cross_device() or self.devices[rank].type != "cuda":
            return
        from hao_basic import hao_launch

        device = self.devices[rank]
        torch.cuda.set_device(device)
        hao_launch(
            self._dummy(channel, device), self.buffers[channel], self.arrived[channel],
            self.semaphore[channel][rank], rank=rank, world=self.world, batch_size=0,
            max_batch=self.max_batch, is_transpose=False, mask=None, dst_row_indices=None,
            cache=self._gpu_tables, handshake=True,
        )

    def _participate_others(self, channel: str, src: int) -> None:
        for rank in range(self.world):
            if rank != src:
                self.participate(channel, rank)

    def _cross_device(self) -> bool:
        ids = [device.index for device in self.devices if device.type == "cuda"]
        return len(set(ids)) > 1

    def _sync_cuda(self) -> None:
        seen = set()
        for device in self.devices:
            if device.type != "cuda" or device.index in seen:
                continue
            seen.add(device.index)
            torch.cuda.synchronize(device)

    def _dummy(self, channel: str, device: torch.device) -> torch.Tensor:
        key = (channel, device.type, device.index)
        hit = self._dummies.get(key)
        if hit is not None:
            return hit
        msg = 1
        for extent in self.shapes[channel]:
            msg *= extent
        hit = torch.empty((0, msg), dtype=self.dtypes[channel], device=device)
        self._dummies[key] = hit
        return hit

    def _table_slots_to_rows(self, src: int, table: torch.Tensor) -> torch.Tensor:
        """Slot numbers become flat rows in ``[src, slot]``."""
        rows = table.shape[1]
        flat = torch.full((self.world, self.max_batch), -1, dtype=torch.int32)
        base = src * self.max_batch
        shifted = torch.where(
            table >= 0,
            table.to(dtype=torch.int32) + base,
            torch.full(table.shape, -1, dtype=torch.int32),
        )
        flat[:, :rows] = shifted
        return flat

    def _mask_slots_to_rows(self, src: int, mask: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        rows = slots.shape[0]
        flat = torch.full((self.world, self.max_batch), -1, dtype=torch.int32)
        base = src * self.max_batch
        slot_i = slots.to(dtype=torch.int32)
        for dst in range(self.world):
            active = mask[dst, :rows] != 0
            if not bool(active.any()):
                continue
            dest = torch.full((rows,), -1, dtype=torch.int32)
            dest[active] = slot_i[active] + base
            flat[dst, :rows] = dest
        return flat

    def _device_launch(self, channel: str, src: int, value: torch.Tensor, transfers: list[Transfer],
                       mask: torch.Tensor | None, dst_row_indices: torch.Tensor | None,
                       slots: torch.Tensor | None, is_transpose: bool = False) -> RouteLaunch:
        from hao_basic import hao_launch

        self._check_tail(channel, value)
        self.epoch[channel] += 1
        epoch = self.epoch[channel]
        launch = RouteLaunch(
            channel, transfers, src=src, epoch=epoch,
            written={transfer.dst for transfer in transfers},
        )
        if slots is not None:
            dst_row_indices = self._mask_slots_to_rows(src, mask, slots)
            mask = None
        elif dst_row_indices is not None:
            dst_row_indices = self._table_slots_to_rows(src, dst_row_indices)
        rows = mask.shape[1] if is_transpose else value.shape[0]
        torch.cuda.set_device(self.devices[src])
        launch.hold.extend(hao_launch(
            value, self.buffers[channel], self.arrived[channel], self.semaphore[channel][src],
            rank=src, world=self.world, batch_size=rows, max_batch=self.max_batch,
            is_transpose=is_transpose, mask=mask, dst_row_indices=dst_row_indices,
            cache=self._gpu_tables, handshake=self._cross_device(),
        ))
        return launch

    def _put(self, channel: str, value: torch.Tensor, transfer: Transfer):
        """CPU placement. CUDA goes through the Basic kernel."""
        buffer = self.buffers[channel][transfer.dst][transfer.src]
        taken = value.index_select(0, transfer.rows.to(value.device))
        buffer.index_copy_(0, transfer.slots.to(dtype=torch.long), taken)
        return None, [taken]
