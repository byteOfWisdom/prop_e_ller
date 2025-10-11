import copy
import sympy
import math
from propeller.util import list_like
import numpy as np


def varname(i: int):
    lut = "abcdefghij"
    return "".join([lut[int(n)] for n in str(i)])


def ev(value, error):
    if list_like(value) and not list_like(error):
        return np.array([ev(x, error) for x in value])
    elif list_like(value):
        return np.array([ev(x, y) for x, y in zip(error, value)])
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
    return x()


def sq(x):
    return x * x


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
            
        sub_vars = [(sympy.Symbol(str(v)), v.value) for v in vars]
        for var in vars:
            partial = sympy.diff(sym_eq, str(var))
            partial = partial.subs(sub_vars)
            error_sq += sq(float(partial)) * sq(var.error)

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
            print(vars.count(v))
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


    def __eq__(self, other):
        return ~self == ~other


    def __neq__(self, other):
        return ~self != ~other


    def __gt__(self, other):
        return ~self > ~other

    def __lt__(self, other):
        return ~self < ~other


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
        if is_primitive_num(other):
            return Addition(self, Number(other))
        return Addition(self, other)

    def __radd__(self, other):
        return self._rop(Addition, other)
        if is_primitive_num(other):
            return Addition(Number(other), self)
        return Addition(other, self)

    def __mul__(self, other):
        return self._lop(Multiplication, other)
        if is_primitive_num(other):
            return Multiplication(self, Number(other))
        return Multiplication(self, other)

    def __rmul__(self, other):
        return self._rop(Multiplication, other)
        if is_primitive_num(other):
            return Multiplication(Number(other), self)
        return Multiplication(other, self)

    def __sub__(self, other):
        return self._lop(Subtraction, other)
        if is_primitive_num(other):
            return Subtraction(self, Number(other))
        return Subtraction(self, other)
    
    def __rsub__(self, other):
        return self._lop(Subtraction, other)
        if is_primitive_num(other):
            return Subtraction(Number(other), self)
        return Subtraction(other, self)

    def __truediv__(self, other):
        return self._lop(Division, other)
        if is_primitive_num(other):
            return Division(self, Number(other))
        return Division(self, other)

    def __rtruediv__(self, other):
        return self._rop(Division, other)
        if is_primitive_num(other):
            return Division(Number(other), self)
        return Division(other, self)

    def __neg__(self):
        return Number(0) - self

    def exp(self):
        return Exp(self)

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


class LiteralContainer(GenericOp):
    def __init__(self, value: float, error: float):
        super().__init__()
        # if list_like(value):
        #     pass
        self.value = value
        self.error = error

    # def vec(self):
    #     return list_like(self.value)

    def _eval(self):
        return self.value

    def __eq__(self, other):
        # if self.vec() or other.vec():
        #     return (self.value == other.value).all() and (self.error == other.error).all()
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

