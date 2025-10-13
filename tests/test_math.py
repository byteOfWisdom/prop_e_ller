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


def dont_test_make_arr():
    a = p.ev(np.linspace(0, 1, 10), 2)
    assert isinstance(p.value(a), np.ndarray)


def dont_test_expr_builder():
    a, b, c, d = default_values()

    exp = a + b + c + d
    print(list(map(str, exp._vars())))
    print(str(exp))
    assert False


def test_addition():
    a, b, c, d = default_values()

    assert isinstance(a + b, p.GenericOp)
    assert float(a + d) == a.value
    assert float(a + c) == 0.
    assert float(a + b + c + d) == 2.


def test_addition_errors():
    a, b, c, d = default_values()

    assert p.error(a + b) == p.error(b + a)
    assert p.error(a + b + c + d) == p.error((b + d) + (c + a))


def test_subtraction():
    a, b, c, d = default_values()

    assert float(a - a) == 0.
    assert float(a - b) == float(d - (b - a))


def test_log_exp():
    a, b, c, d = default_values()

    assert float(np.exp(a)) == np.exp(1.)
    assert float(np.exp(np.log(b))) == float(b)
    assert float(np.log(b)) == np.log(2.)
    assert float(np.log10(b)) == np.log10(float(b))
    # todo: assert correct errors


def test_trig_funcs():
    a, b, c, d = default_values()

    assert float(np.sin(d)) == 0.
    assert float(np.cos(d)) == 1.
    assert float(np.tan(d)) == float(np.sin(d) / np.cos(d))
    # todo: assert correct errors


def test_negation():
    a, _, c, _ = default_values()
    assert float(-a) == float(c)


def test_mul():
    a, b, c, d = default_values()

    assert float(a * d) == 0.
    assert float(a * b) == float(b * a)
    assert p.error(a * b) == p.error(b * a)
    

def test_ops_with_floats():
    a, b, c, d = default_values()
    n = 2.0
    assert float(a * n) == n
    assert float(a + n) == float(n + a)
    assert p.error(a + n) == p.error(n + a)
    

def test_div():
    a, b, c, d = default_values()

    assert float(b / a) == float(b)
    assert float(b / c) == float(1 / (c / b))
    # todo: assert error correctnes 


def test_associativity():
    a, b, c, d = default_values()
    assert float(a * (b * c)) == float((a * b) * c)
    assert float(a + (b + c)) == float((a + b) + c)
    assert p.error(a * (b * c)) == p.error((a * b) * c)
    assert p.error(a + (b + c)) == p.error((a + b) + c)
    

def test_distributivity():
    a, b, c, d = default_values()

    assert float(a * (b + c)) == float(a * b + a * c)
    assert p.error(a * (b + c)) == p.error(a * b + a * c)
    

def test_vector_ops():
    a, b, c, d = default_values()
    many_ones = np.zeros(10)
    many_ones += 1.
    v = p.ev(np.linspace(0.1, 1, 10), 1)
    differing_errors = p.ev(many_ones, np.linspace(0, 1, 10))
    va = ~(v * a)
    res, err = p.ve(v * differing_errors)
    assert isinstance(err, np.ndarray)
    assert (res == va).all()
    assert isinstance(va, np.ndarray)
    assert (v * a == a * v).all()
    assert (v + a == a + v).all()
    assert ((v / a) - (1. / (a / v)) < 0.001).all()
    assert (p.value(v * b) == v * p.value(b)).all()
