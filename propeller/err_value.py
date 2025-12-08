import copy
import sympy
import math
from propeller.util import list_like
import numpy as np


def varname(i: int):
    # lut = "abcdefghij"
    # return "".join([lut[int(n)] for n in str(i)])
    return "x" + str(i)


def ev(value, error):
    if list_like(value) and not list_like(error):
        return np.array([ev(x, error) for x in value])
    elif list_like(value):
        return np.array([ev(x, y) for x, y in zip(value, error)])
    else:
        return ErrVal(value, error)


def ve(generic_expr):
    if list_like(generic_expr):
        errors = np.array([e for _, e in map(ve, generic_expr)])
        values = np.array([v for v, _ in map(ve, generic_expr)])
        return values, errors
    return generic_expr(), generic_expr._cal_error()


def error(x):
    return x._cal_error()


def value(x):
    # return float(x)
    return ~x


def sq(x):
    return x * x


def within_sigma(x, y, n_sigma=1):
    return np.abs(~x - ~y) < n_sigma * (error(x) + error(y))


def is_primitive_num(x):
    return isinstance(x, float) or isinstance(x, int)


class GenericOp:
    id = 0

    def __float__(self):
        return self._eval()

    def __call__(self):
        return self._eval()

    # syntactic sugar... maybe reconsider
    # but maybe i like this
    def __invert__(self):
        return self._eval()

    def all_vars(self):
        return self._vars()

    def _cal_error(self):
        sym_eq, vars = self._to_symbolic_eq()
        error_sq = 0.

        sub_vars = [(sympy.Symbol(str(v)), float(v.value)) for v in vars]
        for var in vars:
            partial = sympy.diff(sym_eq, str(var))
            # print(partial)
            # print(sub_vars)
            partial = partial.subs(sub_vars)

            # try:
            error_sq += sq(float(partial)) * sq(var.error)
            # except TypeError:
                # print(partial)

        return np.sqrt(error_sq)

    def _to_symbolic_eq(self):
        # eq = sympy.parsing.sympy_parser.parse_expr(str(self), evaluate=False)
        sym_eq = sympy.sympify(str(self), evaluate=True)
        vars = self.all_vars()

        # numbers without error can just be substituted in
        # saves derivations
        for v in vars:
            if isinstance(v, Number):
                sym_eq = sym_eq.subs(str(v), float(v))
                vars.remove(v)

        # every occurance of a variable gets a different name
        # these have to be deduplicated in order to achieve correct results
        dedup_vars = []
        for v in vars:
            if vars.count(v) > 1:
                alt_names = [str(vn) for vn in filter(lambda x: x == v, vars)]
                for a_name in alt_names:
                    sym_eq = sym_eq.subs(a_name, str(v))

            dedup_vars.append(v)

        return sym_eq, dedup_vars

    def __int__(self):
        return int(float(self))

    def __str__(self):
        return "not implemented!!!"

    def __repr__(self):
        return str(self)

    def format(self, strict=True):
        value, error = ve(self)
        oom = -math.floor(math.log10(value))
        oom_e = -math.floor(math.log10(error))
        # while round(error * 10**oom) == 0 or round(value * 10**oom) == 0:
            # oom += 1
        value *= 10**oom
        error *= 10**oom
        rv = 0
        while round(error, rv) == 0:
            rv += 1

        if rv == 0:
            return f"{round(value)}({round(error)})e{-oom}"
        return f"{round(value, rv)}({round(error, rv)})e{-oom}"

    def _comp(self, other, op):
        if not isinstance(other, GenericOp):
            return op(~self, other)
        return op(~self, ~other)

    def __eq__(self, other):
        return self._comp(other, lambda x, y: x == y)

    def __neq__(self, other):
        return self._comp(other, lambda x, y: x != y)

    def __gt__(self, other):
        return self._comp(other, lambda x, y: x > y)

    def __lt__(self, other):
        return self._comp(other, lambda x, y: x < y)

    def __ge__(self, other):
        return self._comp(other, lambda x, y: x >= y)

    def __le__(self, other):
        return self._comp(other, lambda x, y: x <= y)

    def _rop(self, op, other):
        if is_primitive_num(other):
            return op(Number(other), self)
        if list_like(other):
            return np.array([op(elem, self) for elem in other])
        return op(other, self)

    def _lop(self, op, other):
        if is_primitive_num(other):
            return op(self, Number(other))
        if list_like(other):
            return np.array([op(self, elem) for elem in other])
        return op(self, other)

    def __add__(self, other):
        return self._lop(Addition, other)

    def __radd__(self, other):
        return self._rop(Addition, other)

    def __mul__(self, other):
        return self._lop(Multiplication, other)

    def __rmul__(self, other):
        return self._rop(Multiplication, other)

    def __sub__(self, other):
        return self._lop(Subtraction, other)

    def __rsub__(self, other):
        return self._rop(Subtraction, other)

    def __truediv__(self, other):
        return self._lop(Division, other)

    def __rtruediv__(self, other):
        return self._rop(Division, other)

    def __pow__(self, other):
        return self._lop(Power, other)

    def __rpow__(self, other):
        return self._rop(Power, other)

    def __neg__(self):
        return Number(0) - self

    def exp(self):
        return Exp(self)

    def sqrt(self):
        return self ** 0.5

    def log(self):
        return Log(self)

    def log10(self):
        return Log10(self)

    def sin(self):
        return Sin(self)

    def cos(self):
        return Cos(self)

    def tan(self):
        return Tan(self)

    def arctan(self):
        return Arctan(self)

    def __abs__(self):
        return Abs(self)

    def _inc_ids(self, n):
        self.id += n


class DualOp(GenericOp):
    def __init__(self, a, b):
        self.a = copy.deepcopy(a)
        self.b = copy.deepcopy(b)
        self.b._inc_ids(self.a._varcount())

    def _eval(self):
        return 0.0

    def _vars(self):
        return self.a._vars() + self.b._vars()

    def __str__(self):
        return f"({str(self.a)} {self.op_type} {str(self.b)})"

    def _varcount(self):
        return self.a._varcount() + self.b._varcount()

    def _inc_ids(self, n):
        self.a._inc_ids(n)
        self.b._inc_ids(n)


class SingularOp(GenericOp):
    def __init__(self, a):
        self.a = copy.deepcopy(a)

    def __str__(self):
        return f"{self.op_type}({str(self.a)})"

    def _vars(self):
        return self.a._vars()

    def _varcount(self):
        return self.a._varcount()

    def _inc_ids(self, n):
        self.a._inc_ids(n)


class Addition(DualOp):
    op_type = "+"

    def _eval(self):
        return self.a._eval() + self.b._eval()


class Multiplication(DualOp):
    op_type = "*"

    def _eval(self):
        return self.a._eval() * self.b._eval()


class Subtraction(DualOp):
    op_type = "-"

    def _eval(self):
        return self.a._eval() - self.b._eval()


class Division(DualOp):
    op_type = "/"

    def _eval(self):
        return self.a._eval() / self.b._eval()


class Exp(SingularOp):
    op_type = "exp"

    def _eval(self):
        return math.exp(self.a._eval())


class Log(SingularOp):
    op_type = "log"

    def _eval(self):
        return math.log(self.a._eval())


class Log10(SingularOp):
    op_type = "log10"

    def _eval(self):
        return math.log10(self.a._eval())


class Sin(SingularOp):
    op_type = "sin"

    def _eval(self):
        return math.sin(self.a._eval())


class Cos(SingularOp):
    op_type = "cos"

    def _eval(self):
        return math.cos(self.a._eval())


class Tan(SingularOp):
    op_type = "tan"

    def _eval(self):
        return math.tan(self.a._eval())


class Arctan(SingularOp):
    op_type = "atan"

    def _eval(self):
        return math.atan(self.a._eval())


class Power(DualOp):
    op_type = "^"

    def _eval(self):
        return self.a._eval() ** self.b._eval()


class Abs(SingularOp):
    op_type = "abs"

    def _eval(self):
        return math.fabs(self.a._eval())


class LiteralContainer(GenericOp):
    def __init__(self, value: float, error: float):
        super().__init__()
        # if list_like(value):
        #     pass
        self.value = value
        self.error = np.abs(error)

    # def vec(self):
    #     return list_like(self.value)

    def _eval(self):
        return self.value

    def __eq__(self, other):
        return (self.value == other.value) and (self.error == other.error)


class ErrVal(LiteralContainer):
    def __str__(self):
        return varname(self.id)

    def _vars(self):
        return [self]

    def _varcount(self):
        return 1


class Number(LiteralContainer):
    def __init__(self, value: float):
        super().__init__(value, 0)

    def __str__(self):
        return str(self.value)

    def _vars(self):
        return []

    def _varcount(self):
        return 0
