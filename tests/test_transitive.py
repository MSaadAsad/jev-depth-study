import random
import pytest
from jevprobe.data import vocab
from jevprobe.data.transitive import (
    MAX_D, N_DISTRACT, N_MAIN, N_PREMISES, Example, build_dataset, load_jsonl, sample_example, save_jsonl,
)


def test_name_pools_disjoint_and_sized():
    train, test = vocab.name_pool("train"), vocab.name_pool("test")
    assert len(train) == 400 and len(test) == 200
    assert not set(train) & set(test)


def test_relations_and_templates_disjoint():
    assert not {r.rel for r in vocab.RELATIONS["train"]} & {r.rel for r in vocab.RELATIONS["test"]}
    assert not set(vocab.PREMISE_TEMPLATES["train"]) & set(vocab.PREMISE_TEMPLATES["test"])


@pytest.mark.parametrize("d", range(1, MAX_D + 1))
@pytest.mark.parametrize("answer", [True, False])
def test_example_invariants(d, answer):
    ex = sample_example(d, answer, "train", random.Random(d * 10 + answer), "u")
    assert len(ex.premises) == N_PREMISES
    assert N_MAIN == 24 and N_DISTRACT == 3 and N_PREMISES == 25
    assert len(set(ex.chain + ex.distractors)) == N_MAIN + N_DISTRACT
    expected = {(ex.chain[k], ex.chain[k + 1]) for k in range(N_MAIN - 1)} | {
        (ex.distractors[k], ex.distractors[k + 1]) for k in range(N_DISTRACT - 1)
    }
    assert {(p.greater, p.lesser) for p in ex.premises} == expected
    assert abs(ex.rank_x - ex.rank_y) == d
    assert (ex.rank_x < ex.rank_y) == answer
    assert len(ex.bridges) == d - 1


def test_bridges_ordered_from_x_and_same_set_when_flipped():
    ex = sample_example(4, True, "train", random.Random(1), "u")
    assert ex.bridges == ex.chain[ex.rank_x + 1 : ex.rank_y]
    flipped = Example.from_dict({**ex.to_dict(), "x": ex.y, "y": ex.x, "answer": False})
    assert flipped.bridges == ex.bridges[::-1]


def test_invalid_d_rejected():
    with pytest.raises(ValueError):
        sample_example(0, True, "train", random.Random(0), "u")
    with pytest.raises(ValueError):
        sample_example(MAX_D + 1, True, "train", random.Random(0), "u")


def test_dataset_balanced_and_split_names():
    ds = build_dataset(20, "test", seed=0)
    assert len(ds) == 120
    for d in range(1, 7):
        sub = [e for e in ds if e.d == d]
        assert len(sub) == 20 and sum(e.answer for e in sub) == 10
    pool = set(vocab.name_pool("test"))
    assert all(set(e.chain + e.distractors) <= pool for e in ds)
    assert all(e.rel in {r.rel for r in vocab.RELATIONS["test"]} for e in ds)


def test_odd_n_rejected():
    with pytest.raises(ValueError):
        build_dataset(3, "train", seed=0)


def test_deterministic_and_jsonl_roundtrip(tmp_path):
    a, b = build_dataset(4, "train", seed=7), build_dataset(4, "train", seed=7)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]
    p = tmp_path / "x.jsonl"
    save_jsonl(a, p)
    assert [e.to_dict() for e in load_jsonl(p)] == [e.to_dict() for e in a]


@pytest.mark.parametrize("d", range(1, MAX_D + 1))
def test_query_entities_beyond_endpoint_reach(d):
    """Any endpoint-anchored shortcut must cost more hops than the path itself (review C1)."""
    for seed in range(40):
        for answer in (True, False):
            ex = sample_example(d, answer, "test", random.Random(seed), "u")
            dist = min(min(r, N_MAIN - 1 - r) for r in (ex.rank_x, ex.rank_y))
            assert dist >= d + 1, (ex.rank_x, ex.rank_y)
