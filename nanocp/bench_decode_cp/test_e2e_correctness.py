"""Validate complete graphed DP/head-TP/CP outputs against the same FP32 reference."""
import unittest
from types import SimpleNamespace

import torch

from bench_perf import full_batch, head_step, run_case
from comm_cp import enable_peers
from cuda_ipc_p2p import capture_graph
from measurement import ignore_stage
from workloads import QUICK, Workload


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestDecodeOutputs(unittest.TestCase):
    def test_head_tp_returns_current_partner_output_exactly(self):
        # A combine CTA publishes values stored by other threads in the block.
        # Change Q between replays so returning a previous value is observable.
        spec = Workload("return_exact", (65536,) * 4 + (256,) * 128, 4, (512,) * 512)
        with torch.inference_mode():
            inputs = [full_batch(spec.owner, 11), full_batch(spec.partner, 29)]
            work, streams = head_step(spec, inputs, compute_only=False)
            case = work.__self__
            graph = capture_graph(streams, lambda: work(ignore_stage))
            for _ in range(100):
                with torch.cuda.stream(streams[0]):
                    case.owner[0].neg_()
                    graph.replay()
                streams[0].synchronize()
                got = case.owner_out[:spec.n_long, 8:].cpu()
                expected = case.partner_out[:spec.n_long, 8:].cpu()
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

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
