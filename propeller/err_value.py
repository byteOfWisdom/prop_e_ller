import sympy


def varname(i: int):
    lut = "abcdefghij"
    return "".join([lut[int(n)] for n in str(i)])


def name_maker():
    i = 0
    while True:
        yield varname(i)
        i += 1

varnames = name_maker()

def ev(value, error):
    return ErrVal(value, error)


def isclose(a, b):
    return True


def error(x):
    return x.error


def value(x):
    return x.value


class GenericOp:
    id = 0

    def __float__(self):
        return self._eval()


    def all_vars(self):
        return self._vars()


    def _cal_error(self):
        sym_eq = self._to_symbolic_eq()
        partials = [sympy.diff(sym_eq, var) for var in sym_eq.args()]
        error = sum([sympy.sympify(partial).evalf() for partial in partials])
        return error


    def _to_symbolic_eq(self):
        eq = sympy.parsing.sympy_parser.parse_expr(str(self), evaluate=False)
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
        self.a = a
        self.b = b
        self.b._inc_ids(self.a._varcount())


    def _eval(self):
        return 0.0

    def _vars(self):
        return self.a._vars() + self.b._vars()

    def apply(self, num):
        return num

    def prepend(self, op):
        return DualOp(op, self)

    def __str__(self):
        return f"({str(self.a)} {self.op_type} {str(self.b)})"

    def _varcount(self):
        return self.a._varcount() + self.b._varcount()


    def _inc_ids(self, n):
        self.a._inc_ids(n)
        self.b._inc_ids(n)


class SingularOp(GenericOp):
    def __init__(self, a):
        self.a = a

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

    def _eval(self):
        return self.value

    def __str__(self):
        return varname(self.id)

    def _vars(self):
        return [self]

    def _varcount(self):
        return 1

