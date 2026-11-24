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
