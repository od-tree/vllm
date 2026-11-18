"""Logical requests resident on each DP rank before any collaboration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    owner: tuple[int, ...]
    n_long: int
    partner: tuple[int, ...] = ()

    def __post_init__(self):
        if not 1 <= self.n_long <= len(self.owner):
            raise ValueError("the owner must have at least one cooperative request")
        if min(self.owner + self.partner) < 1:
            raise ValueError("decode sequences must contain the current token")


QUICK = (
    Workload("n4_1024", (1024,) * 4, 4),
    Workload("mix_4x128_1x1024", (1024,) + (128,) * 4, 1),
    Workload("n1_8192", (8192,), 1),
    Workload("paired_smoke", (65, 96, 128, 17), 3, (33, 129)),
)

CASES = tuple(
    Workload(f"{count}x{length}", (length,) * count, count)
    for count, length in ((20, 32768), (128, 8192), (128, 32768),
                          (128, 65536), (256, 32768))
) + (
    Workload("mix_128x256_2x128k_1x256k", (131072, 131072, 262144) + (256,) * 128, 3),
    Workload("mix_64x1024_4x128k", (131072,) * 4 + (1024,) * 64, 4),
    Workload("mix_32x512_2x256k", (262144,) * 2 + (512,) * 32, 2),
    Workload("mix_512x512_16x256k", (262144,) * 16 + (512,) * 512, 16),
    Workload("mix_1024x1024_4x256k", (262144,) * 4 + (1024,) * 1024, 4, (1024,) * 1024),
)
