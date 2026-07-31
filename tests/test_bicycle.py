from codegap_qa.bicycle import BicycleFamilySpec, build_bicycle_css
from codegap_qa.gf2 import commute


def test_bicycle_commutes():
    spec = BicycleFamilySpec(
        l=2,
        m=3,
        support_a=((0, 0), (1, 0)),
        support_b=((0, 1), (1, 2)),
    )
    h_x, h_z = build_bicycle_css(spec)
    assert h_x.shape == (6, 12)
    assert h_z.shape == (6, 12)
    assert commute(h_x, h_z)
