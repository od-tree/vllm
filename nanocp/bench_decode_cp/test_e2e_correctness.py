"""Validate complete graphed DP/head-TP/CP outputs against the same FP32 reference."""
import unittest
from types import SimpleNamespace

import torch

from bench_perf import run_case
from comm_cp import enable_peers
from workloads import QUICK, Workload


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestDecodeOutputs(unittest.TestCase):
    def test_owner_and_partner_outputs(self):
        enable_peers([0, 1])
        args = SimpleNamespace(check=True, check_only=True, rounds=1, profile_rounds=1)
        cases = (QUICK[-1], Workload("single_128", (128,), 1),
                 Workload("partial_page", (100,), 1), Workload("empty_prefix", (17,), 1))
        with torch.inference_mode():
            for case in cases:
                with self.subTest(case=case.name):
                    run_case(case, ("dp", "head", "cp"), args, compute_only=False)


if __name__ == "__main__":
    unittest.main()
