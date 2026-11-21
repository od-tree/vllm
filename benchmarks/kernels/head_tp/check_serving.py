"""Compare two runs of the same DP+EP server using fixed IDs and greedy sampling."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def generate(url, model, rank, prompt):
    request = Request(
        url + "/v1/completions",
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
        data=json.dumps(
            dict(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=12,
                ignore_eos=True,
                logprobs=1,
                return_tokens_as_token_ids=True,
            )
        ).encode(),
    )
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    assert result["usage"]["completion_tokens"] == 12
    data = result["choices"][0]["logprobs"]
    return {"tokens": data["tokens"], "logprobs": data["token_logprobs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:18489")
    parser.add_argument("--model", default="qwen-2507")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prompt-length", type=int, default=253)
    parser.add_argument("--logprob-atol", type=float, default=0.005)
    parser.add_argument(
        "--check-contention",
        action="store_true",
        help="check both ranks after a streaming disconnect (requires --reference)",
    )
    args = parser.parse_args()
    if args.check_contention and not args.reference:
        parser.error("--check-contention requires a matching native --reference")

    # The first request crosses a 256-token page boundary during decode.
    prompts = [[1000 + (i % 7) for i in range(args.prompt_length)], [2000, 2001] * 13]
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(generate, args.url, args.model, i, p)
            for i, p in enumerate(prompts)
        ]
        records = [f.result() for f in futures]
    # Exercise an entirely idle partner in each direction.
    for rank in (1, 0):
        records.append(generate(args.url, args.model, rank, prompts[rank]))
    args.save.write_text(json.dumps(records, indent=2))
    if args.reference:
        baseline = json.loads(args.reference.read_text())
        worst = 0.0
        for run, ref in zip(records, baseline, strict=True):
            assert run["tokens"] == ref["tokens"], (run, ref)
            assert all(math.isfinite(x) for x in run["logprobs"] + ref["logprobs"])
            worst = max(
                worst,
                *(
                    abs(x - y)
                    for x, y in zip(run["logprobs"], ref["logprobs"], strict=True)
                ),
            )
        assert worst <= args.logprob_atol, worst
        print(f"Greedy tokens match; maximum logprob difference {worst:.6g}")

        if args.check_contention:
            # Close the stream while it owns a live packed cache. Both ranks
            # must subsequently make progress, even if only one lease fits.
            req = Request(
                args.url + "/v1/completions",
                data=json.dumps(
                    dict(
                        model=args.model,
                        prompt=prompts[0],
                        stream=True,
                        ignore_eos=True,
                        temperature=0,
                        max_tokens=650,
                    )
                ).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-data-parallel-rank": "0",
                },
            )
            with urlopen(req, timeout=120) as stream:
                assert any(line.startswith(b"data:") for line in stream)
            with ThreadPoolExecutor(2) as pool:
                futures = [
                    pool.submit(generate, args.url, args.model, rank, prompts[0])
                    for rank in (0, 1)
                ]
                for future in futures:
                    actual = future.result()
                    expected = baseline[-1]  # the isolated long request
                    assert actual["tokens"] == expected["tokens"]
                    assert all(
                        abs(a - b) <= args.logprob_atol
                        for a, b in zip(
                            actual["logprobs"], expected["logprobs"], strict=True
                        )
                    )
            print("Both competing long requests matched after stream disconnect.")


if __name__ == "__main__":
    main()
