import pytest
import torch
from jevprobe.models import parse_model_ref
from jevprobe.readout import letter_ids, true_minus_false


class FakeTok:
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return self.table[text]


def test_letter_ids_single_tokens():
    assert letter_ids(FakeTok({"A": [32], "B": [33]})) == (32, 33)


def test_letter_ids_rejects_multi_token():
    with pytest.raises(ValueError):
        letter_ids(FakeTok({"A": [32], "B": [33, 1]}))


def test_true_minus_false():
    logits = torch.zeros(40)
    logits[32], logits[33] = 2.5, -1.0
    assert true_minus_false(logits, (32, 33)) == pytest.approx(3.5)


def test_parse_model_ref():
    assert parse_model_ref("alibiserikbay/JevK5@v0.2") == ("alibiserikbay/JevK5", "v0.2")
    assert parse_model_ref("crh225/plumb-4b") == ("crh225/plumb-4b", None)
