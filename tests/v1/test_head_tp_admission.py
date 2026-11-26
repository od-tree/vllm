# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest

from vllm.v1.head_tp_admission import AdmissionRow, LoadAdmission


def row(name, length, **kwargs):
    return replace(AdmissionRow(name, length, 0, 142, True), **kwargs)


@pytest.mark.parametrize("owner", [0, 1])
def test_split_only_when_the_critical_rank_benefits(owner):
    policy = LoadAdmission({}, 1024)
    long = row("long", 141698)
    loads = [[], []]
    loads[owner] = [long] + [row(f"bg{i}", 16384) for i in range(4)]
    loads[1 - owner] = [row(f"bg{i}", 16384) for i in range(4)]
    assert policy.choose(loads, [None, None]) == (owner, "long")
    loads[1 - owner].append(row("other", 141698))
    assert policy.choose(loads, [None, None]) is None


def test_short_and_short_lived_rows_stay_local():
    policy = LoadAdmission({}, 1024)
    for r in (
        row("short", 513),
        row("almost_done", 141698, history=141690, remaining=2),
        row("out_of_capacity", 141698, eligible=False),
    ):
        assert policy.choose([[r], []], [None, None]) is None


def test_migration_changes_admission_and_retention():
    policy = LoadAdmission({}, 1024)
    r = row("long", 141698)
    assert policy.choose([[r], []], [None, None]) == (0, "long")
    existing = replace(r, history=r.length)
    assert policy.choose([[existing], []], [None, None]) is None
    assert policy.choose([[existing], []], ["long", None]) == (0, "long")
    # A short remaining horizon cannot amortize even an unprofitable exit.
    assert policy.choose([[existing], [row("peer", r.length)]], ["long", None]) == (
        0,
        "long",
    )
    patient = LoadAdmission({"horizon": 512}, 1024)
    assert (
        patient.choose(
            [[replace(existing, remaining=480)], [row("peer", r.length)]],
            ["long", None],
        )
        is None
    )


def test_best_candidate_can_be_shorter_than_the_longest_request():
    policy = LoadAdmission({"return_fraction": 0}, 1024)
    batches = [
        [row("larger", 200000), row("better", 160000)],
        [row("bg", 200000, eligible=False)],
    ]
    assert policy.choose(batches, [None, None]) == (0, "better")


@pytest.mark.parametrize("owner", [0, 1])
def test_frontend_uses_context_work_for_the_selected_owner(owner):
    policy = LoadAdmission({}, 1024)
    contexts = [0, 0]
    assert policy.suggest(contexts, owner, 109042, 120)
    contexts[1 - owner] = 109042
    assert not policy.suggest(contexts, owner, 109042, 120)
    assert not policy.suggest([0, 0], owner, 256, 120)


def test_helper_rank_zero_survives_engine_request_conversion():
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request

    message = EngineCoreRequest(
        request_id="req",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=1,
        head_tp_helper_rank=0,
    )
    assert Request.from_engine_core_request(message, block_hasher=None).head_tp_proposed
    message.head_tp_helper_rank = None
    assert not Request.from_engine_core_request(message, block_hasher=None).head_tp_proposed
