import propeller as p
import numpy as np


def default_values():
    a = p.ev(1., 2.)
    b = p.ev(2., 3.)
    c = p.ev(-1., 2.)
    d = p.ev(0., 0.5)
    return a, b, c, d


def test_make_ev():
    a = p.ev(1, 2)
    assert a.value == 1
    assert a.error == 2


def test_make_arr():
    a = p.ev(np.linspace(0, 1, 10), 2)
    assert isinstance(p.value(a), np.ndarray)


def test_addition():
    a, b, c, d = default_values()

    assert (a + b).value == 3.
    assert (a + d).value == a.value
    assert a + c == 0.


def test_subtraction():
    a, b, c, d = default_values()

    assert (a - b).value == -1.
    assert (a - d).value == a.value
    assert a - c == 2.


def test_negation():
    a, _, c, _ = default_values()
    assert -a == c
    
