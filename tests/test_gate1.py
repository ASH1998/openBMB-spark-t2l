import json
from pathlib import Path

import pytest

from minicpm_t2l import gate1

STUDY = Path("studies/task_pool")


@pytest.fixture(scope="module")
def splits():
    m = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    h = json.loads((STUDY / "heldout_2026-11-19.json").read_text(encoding="utf-8"))
    return gate1.make_splits(m, h)


def test_split_sizes_match_protocol(splits):
    assert len(splits.test) == 78 and len(splits.dev) == 32 and len(splits.train) == 333
    assert not set(splits.train) & set(splits.dev) and not set(splits.train) & set(splits.test)
    assert all(t.startswith(("syn_gym_membership_", "syn_utility_bill_")) for t in splits.dev)


def test_dev_domains_are_the_seeded_draw():
    import random

    from minicpm_t2l import synth
    from minicpm_t2l.synth_fields import DOMAINS

    train_domains = sorted(set(DOMAINS) - set(synth.heldout_domains()))
    assert tuple(sorted(random.Random(gate1.DEV_SEED).sample(train_domains, 2))) == gate1.DEV_DOMAINS


def test_support_draws_are_nested_and_fixed():
    ids = [f"d{i}" for i in range(50)]
    s1, s4, s8 = (gate1.support_for(ids, k) for k in (1, 4, 8))
    assert s8[:4] == s4 and s4[:1] == s1 and len(set(s8)) == 8
    assert gate1.support_for(ids, 4) == s4
    with pytest.raises(ValueError):
        gate1.support_draw(ids[:5])


def test_training_samples_disjoint_and_deterministic():
    sizes = {f"t{i}": 30 for i in range(20)}
    a = gate1.training_samples(sizes, 50, seed=3)
    assert a == gate1.training_samples(sizes, 50, seed=3)
    for step in a:
        assert len(step) == gate1.TASKS_PER_STEP and len({s.task for s in step}) == gate1.TASKS_PER_STEP
        for s in step:
            assert s.k in gate1.KS_TRAIN and len(s.support) == s.k and s.query not in s.support
