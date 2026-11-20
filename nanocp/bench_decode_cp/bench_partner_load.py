"""Hold the owner's batch fixed while varying the partner's resident rows."""
import argparse
from dataclasses import replace

import torch

from bench_perf import run_case
from comm_cp import enable_peers
from workloads import Workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--profile-rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cp-owner-percent", type=int, nargs="+",
                        help="Run only CP with these owner shares, in percent.")
    parser.add_argument("--equal-tokens", action="store_true",
                        help="Hold partner KV at 131072 tokens and vary its request count.")
    args = parser.parse_args()
    if min(args.rounds, args.profile_rounds, args.repeats) < 1:
        parser.error("rounds and repeats must be positive")
    if args.cp_owner_percent and any(not 0 < share < 100 for share in args.cp_owner_percent):
        parser.error("CP owner shares must lie strictly between 0 and 100")
    args.check_only = False
    owner = Workload("background", (65536,) * 4 + (256,) * 128, 4)
    methods = ("dp", "head", "cp")
    counts = (64, 256, 1024) if args.equal_tokens else (0, 128, 256, 384, 512, 1024, 2048)
    enable_peers([0, 1])
    with torch.inference_mode():
        for repeat in range(args.repeats):
            order = repeat % len(methods)
            for count in counts if repeat % 2 == 0 else reversed(counts):
                length = 131072 // count if args.equal_tokens else 512
                series = "equal_tokens" if args.equal_tokens else "background"
                spec = replace(owner, name=f"{series}_{count}_rep_{repeat + 1}",
                               partner=(length,) * count)
                args.check = repeat == 0 and count == (256 if args.equal_tokens else 512)
                if args.cp_owner_percent:
                    shares = args.cp_owner_percent
                    order = repeat % len(shares)
                    for share in shares[order:] + shares[:order]:
                        run_case(spec, ("cp",), args, compute_only=False, cp_owner_percent=share)
                else:
                    run_case(spec, methods[order:] + methods[:order], args, compute_only=False)


if __name__ == "__main__":
    main()
