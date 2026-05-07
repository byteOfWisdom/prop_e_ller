import propeller as p
import random

# def test_format_ev():
#     # assert p.ev(1.5, 0.3).format() == "1.5(3)"
#     assert p.ev(15, 3).format() == "15(3)"
#     # assert p.ev(15, 30).format() == ""


def test_from_string():
    for _ in range(10000):
        a, b = random.random() * 1e3, random.random() * 1e3
        test_value = p.ev(a, b)
        assert p.within_sigma(test_value, p.from_string(test_value.format()))
