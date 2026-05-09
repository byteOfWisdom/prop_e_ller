import propeller as p
import random

# def test_format_ev():
#     # assert p.ev(1.5, 0.3).format() == "1.5(3)"
#     assert p.ev(15, 3).format() == "15(3)"
#     # assert p.ev(15, 30).format() == ""


def test_from_string():
    for e in [1, 10, 0.1, 0.01, 1e-4]:
        test_value = p.ev(1, e)
        assert p.value(test_value) == p.value(p.from_string(test_value.format()))
        assert p.error(test_value) == p.error(p.from_string(test_value.format()))
        assert p.within_sigma(test_value, p.from_string(test_value.format()))
