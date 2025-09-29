import copy
import sympy
import math


def varname(i: int):
    lut = "abcdefghij"
    return "".join([lut[int(n)] for n in str(i)])


def ev(value, error):
    return ErrVal(value, error)


def error(x):
    return x._cal_error()


def value(x):
    return float(x)


def sq(x):
    return x * x


def is_primitive_num(x):
    return isinstance(x, float) or isinstance(x, int)


class GenericOp:
    id = 0

    def __float__(self):
        return self._eval()


    def all_vars(self):
        return self._vars()


    def _cal_error(self):
        sym_eq = self._to_symbolic_eq()
        vars = self.all_vars()

        error_sq = 0.0

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

        vars = dedup_vars
            
        sub_vars = [(sympy.Symbol(str(v)), v.value) for v in vars]
        for var in vars:
            if isinstance(var, Number):
                continue

            partial = sympy.diff(sym_eq, str(var))
            partial = partial.subs(sub_vars)
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
        if is_primitive_num(other):
            return Addition(self, Number(other))
        return Addition(self, other)

    def __radd__(self, other):
        if is_primitive_num(other):
            return Addition(Number(other), self)
        return Addition(other, self)

    def __mul__(self, other):
        if is_primitive_num(other):
            return Multiplication(self, Number(other))
        return Multiplication(self, other)

    def __rmul__(self, other):
        if is_primitive_num(other):
            return Multiplication(Number(other), self)
        return Multiplication(other, self)

    def __sub__(self, other):
        if is_primitive_num(other):
            return Subtraction(self, Number(other))
        return Subtraction(self, other)
    
    def __rsub__(self, other):
        if is_primitive_num(other):
            return Subtraction(Number(other), self)
        return Subtraction(other, self)

    def __truediv__(self, other):
        if is_primitive_num(other):
            return Division(self, Number(other))
        return Division(self, other)

    def __rtruediv__(self, other):
        if is_primitive_num(other):
            return Division(Number(other), self)
        return Division(other, self)

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


class Log(SingularOp):
    op_type = "log"

    def _eval(self):
        return math.log(self.a._eval())


class Sin(SingularOp):
    op_type = "sin"

    def _eval(self):
        return math.sin(self.a._eval())


class ErrVal(GenericOp):
    def __init__(self, value: float, error: float):
        super().__init__()
        self.value = value
        self.error = error

    def _eval(self):
        return self.value

    def __str__(self):
        return varname(self.id)

    def __eq__(self, other):
        return self.value == other.value and self.error == other.error

    def _vars(self):
        return [self]

    def _varcount(self):
        return 1


class Number(GenericOp):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.error = 0

    def _eval(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other.value and self.error == other.error

    def _vars(self):
        return []

    def _varcount(self):
        return 0

