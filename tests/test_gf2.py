import numpy as np
from codegap_qa.gf2 import commute, css_k, css_distance_at_least


def test_small_css():
    h_x = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    h_z = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    assert commute(h_x, h_z)
    assert css_k(h_x, h_z) == 2
    ok, certificate = css_distance_at_least(h_z, h_x, 2)
    assert not ok
    assert certificate["witness_weight"] == 1
