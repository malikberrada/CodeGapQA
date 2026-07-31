import numpy as np
from codegap_qa.qpu_counts import (
    choose_count_convention,
    counts_to_samples,
    normalize_count_key,
)


def test_hex_and_qiskit_order():
    assert normalize_count_key('0x3', 4) == '0011'
    samples = counts_to_samples({'0011': 2}, 4, 'qiskit')
    assert samples.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]


def test_choose_convention():
    counts = {'probe': {'0011': 100}}
    expected = {'probe': (1, 1, 0, 0)}
    convention, scores = choose_count_convention(counts, expected)
    assert convention == 'qiskit'
    assert scores['qiskit'] == 1.0
