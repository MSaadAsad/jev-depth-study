import pytest
from jevprobe.tokens import last_token_index

OFFSETS = [(0, 0), (0, 4), (4, 5), (5, 10), (10, 11)]  # special, "Mara", " ", " Tolen" (leading space), "."


def test_last_token_overlapping_span_end():
    assert last_token_index(OFFSETS, 0, 4) == 1
    assert last_token_index(OFFSETS, 6, 10) == 3  # token includes the leading space
    assert last_token_index(OFFSETS, 6, 11) == 4


def test_zero_width_tokens_never_selected():
    with pytest.raises(ValueError):
        last_token_index([(0, 0), (0, 0)], 0, 0)


def test_no_overlap_raises():
    with pytest.raises(ValueError):
        last_token_index(OFFSETS, 20, 25)


import random
import re
from dataclasses import replace

from jevprobe.data.counterfactual import make_counterfactual
from jevprobe.data.transitive import sample_example
from jevprobe.tokens import token_aligned


class ChunkTok:
    """Splits words into 3-char chunks, so a 4-letter and a 7-letter name differ in token count."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        spans = [m.span() for m in re.finditer(r"\w{1,3}|[^\w\s]|\s+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def _pair_with_partner_name(name_len):
    ex = sample_example(2, True, "test", random.Random(3), "u")
    cf = make_counterfactual(ex, random.Random(0))
    partner = ex.chain[cf.rank_y]
    new = "Q" * name_len
    ren = lambda e: replace(e, chain=[new if n == partner else n for n in e.chain],
                            premises=[replace(p, greater=new if p.greater == partner else p.greater,
                                              lesser=new if p.lesser == partner else p.lesser) for p in e.premises])
    return ren(ex), ren(cf), len(ex.y)


def test_token_aligned_detects_length_mismatch():
    aligned = token_aligned(ChunkTok())
    clean, cf, ylen = _pair_with_partner_name(5)
    same, diff = (ylen + 2) // 3 * 3, (ylen + 2) // 3 * 3 + 3  # same / one more 3-char chunk than Y
    c1, f1, _ = _pair_with_partner_name(same)
    c2, f2, _ = _pair_with_partner_name(diff)
    assert aligned(c1, f1)
    assert not aligned(c2, f2)
