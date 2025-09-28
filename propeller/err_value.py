import copy
import sympy
import math


def varname(i: int):
    lut = "abcdefghij"
    return "".join([lut[int(n)] for n in str(i)])


def ev(value, error):
    return ErrVal(value, error)


def isclose(a, b):
    return True


def error(x):
    return x._cal_error()


def value(x):
    return float(x)


def sq(x):
    return x * x

class GenericOp:
    id = 0

    def __float__(self):
        return self._eval()


    def all_vars(self):
        return self._vars()


    def _cal_error(self):
        sym_eq = self._to_symbolic_eq()
        vars = self.all_vars()

        print(sym_eq)

        error_sq = 0.0
        for var in vars:
            if isinstance(var, Number):
                continue

            partial = sympy.diff(sym_eq, str(var))
            for v in vars:
                partial.subs(str(v), v.value)
            error_sq += sq(float(partial)) * sq(var.error)

        return math.sqrt(error_sq)


    def _to_symbolic_eq(self):
        # eq = sympy.parsing.sympy_parser.parse_expr(str(self), evaluate=False)
        eq = sympy.sympify(str(self), evaluate=True)
        return eq

    
    def __int__(self):
        return int(float(self))


    def __str__(self):
        return "not implemented!!!"


    def __add__(self, other):
        return Addition(self, other)

    def _inc_ids(self, n):
        self.id += n

    
class DualOp(GenericOp):
    def __init__(self, a, b):
        print("making new dual op")
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


class ErrVal(GenericOp):
    def __init__(self, value: float, error: float):
        super().__init__()
        self.value = value
        self.error = error

    def _eval(self):
        return self.value

    def __str__(self):
        return varname(self.id)

    def _vars(self):
        return [self]

    def _varcount(self):
        return 1


class Number(GenericOp):
    def __init__(self, value: float, error: float):
        super().__init__()
        self.value = value
        self.error = 0

    def _eval(self):
        return self.value

    def __str__(self):
        return varname(self.id)

    def _vars(self):
        return [self]

    def _varcount(self):
        return 1

