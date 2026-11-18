"""Matched two-stream CUDA graph timing. Diagnostic replays are separate."""
from collections import defaultdict
from statistics import median

import torch

from cuda_ipc_p2p import capture_graph, graph_streams_active, replay_samples


def ignore_stage(name):
    pass


def kernel_times(graph, streams, rounds):
    from torch.profiler import ProfilerActivity, profile

    captures = []
    for _ in range(rounds):
        launches = defaultdict(list)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with torch.cuda.stream(streams[0]):
                graph.replay()
            streams[0].synchronize()
        for event in prof.events():
            if event.device_type != torch.autograd.DeviceType.CUDA:
                continue
            launches[(event.device_index, event.name)].append(event.device_time_total / 1000)
        if not launches:
            raise RuntimeError("CUPTI returned no CUDA activities for the graph replay")
        captures.append(launches)
    results = []
    for rank, name in sorted(set().union(*(capture.keys() for capture in captures))):
        runs = [capture.get((rank, name), []) for capture in captures]
        counts = [len(run) for run in runs]
        result = {"rank": rank, "kernel": name, "launch_counts": counts,
                  "median_total_ms": median(sum(run) for run in runs)}
        if min(counts) == max(counts):
            result["per_launch_median_ms"] = [median(column) for column in zip(*runs)]
        results.append(result)
    return results


def stage_times(work, streams, rounds):
    # External event nodes remain observable after replay. Internal events
    # used for cross-stream dependencies do not have this property.
    boundaries = []

    def mark(name):
        if graph_streams_active():
            events = [torch.cuda.Event(enable_timing=True, external=True) for _ in streams]
            for event, stream in zip(events, streams):
                event.record(stream)
            boundaries.append((name, events))

    def diagnostic():
        mark("start")
        work(mark)

    graph = capture_graph(streams, diagnostic)
    samples = defaultdict(list)
    for _ in range(rounds):
        with torch.cuda.stream(streams[0]):
            graph.replay()
        streams[0].synchronize()
        for (_, starts), (name, ends) in zip(boundaries, boundaries[1:]):
            for rank, (start, end) in enumerate(zip(starts, ends)):
                samples[rank, name].append(start.elapsed_time(end))
    return [{"rank": rank, "stage": name, "median_ms": median(values)}
            for (rank, name), values in samples.items()]


def measure(work, streams, rounds, profile_rounds, check=None, check_only=False):
    """work(mark) returns the two logical owners' outputs when requested."""
    outputs = None

    def once():
        nonlocal outputs
        outputs = work(ignore_stage)

    graph = capture_graph(streams, once)
    with torch.cuda.stream(streams[0]):
        graph.replay()
    streams[0].synchronize()
    if check is not None:
        check(outputs)
    if check_only:
        return {}
    samples, host_samples = replay_samples(graph, streams[0], rounds)
    kernels = kernel_times(graph, streams, profile_rounds)
    del graph, outputs
    return {"graph_median_ms": median(samples), "graph_samples_ms": samples,
            "host_median_ms": median(host_samples), "host_samples_ms": host_samples,
            "kernels": kernels, "stages": stage_times(work, streams, profile_rounds)}
