"""How each NumPy ufunc treats physical dimensions.

``RULES`` maps every ufunc QuantSI supports to a `Rule`. It is keyed by the ufunc
object itself rather than by its name: aliases such as ``np.divide`` and
``np.true_divide`` are the same object and cannot disagree, and a lookup is a
single dictionary access instead of scanning lists of names.
"""

import enum

import numpy as np


class Rule(enum.Enum):
    """What a ufunc does with the dimensions of its arguments."""

    PRESERVE = "keeps the dimensions of its argument"
    MATCH = "needs arguments with matching dimensions, keeps them"
    COMPARE = "needs arguments with matching dimensions, returns plain booleans"
    UNITLESS_RESULT = "accepts any dimensions, returns a plain (unitless) result"
    MULTIPLY = "multiplies the dimensions"
    DIVIDE = "divides the dimensions"
    SQRT = "takes the square root of the dimensions"
    SQUARE = "squares the dimensions"
    RECIPROCAL = "inverts the dimensions"
    POWER = "raises the dimensions to a dimensionless, scalar exponent"
    DIMENSIONLESS = "needs a dimensionless argument"
    DIMENSIONLESS_BOTH = "needs two dimensionless arguments"
    INTEGER_ONLY = "only works on integers, never on quantities"


#: The rule for each supported ufunc.
RULES = {
    np.absolute: Rule.PRESERVE,
    np.rint: Rule.PRESERVE,
    np.negative: Rule.PRESERVE,
    np.positive: Rule.PRESERVE,
    np.conjugate: Rule.PRESERVE,
    np.floor: Rule.PRESERVE,
    np.ceil: Rule.PRESERVE,
    np.trunc: Rule.PRESERVE,
    np.add: Rule.MATCH,
    np.subtract: Rule.MATCH,
    np.maximum: Rule.MATCH,
    np.minimum: Rule.MATCH,
    np.remainder: Rule.MATCH,
    np.fmod: Rule.MATCH,
    np.less: Rule.COMPARE,
    np.less_equal: Rule.COMPARE,
    np.greater: Rule.COMPARE,
    np.greater_equal: Rule.COMPARE,
    np.equal: Rule.COMPARE,
    np.not_equal: Rule.COMPARE,
    np.logical_and: Rule.UNITLESS_RESULT,
    np.logical_or: Rule.UNITLESS_RESULT,
    np.logical_xor: Rule.UNITLESS_RESULT,
    np.logical_not: Rule.UNITLESS_RESULT,
    np.isfinite: Rule.UNITLESS_RESULT,
    np.isinf: Rule.UNITLESS_RESULT,
    np.isnan: Rule.UNITLESS_RESULT,
    np.sign: Rule.UNITLESS_RESULT,
    np.multiply: Rule.MULTIPLY,
    np.matmul: Rule.MULTIPLY,
    np.divide: Rule.DIVIDE,
    np.floor_divide: Rule.DIVIDE,
    np.sqrt: Rule.SQRT,
    np.square: Rule.SQUARE,
    np.reciprocal: Rule.RECIPROCAL,
    np.power: Rule.POWER,
    np.sin: Rule.DIMENSIONLESS,
    np.sinh: Rule.DIMENSIONLESS,
    np.arcsin: Rule.DIMENSIONLESS,
    np.arcsinh: Rule.DIMENSIONLESS,
    np.cos: Rule.DIMENSIONLESS,
    np.cosh: Rule.DIMENSIONLESS,
    np.arccos: Rule.DIMENSIONLESS,
    np.arccosh: Rule.DIMENSIONLESS,
    np.tan: Rule.DIMENSIONLESS,
    np.tanh: Rule.DIMENSIONLESS,
    np.arctan: Rule.DIMENSIONLESS,
    np.arctanh: Rule.DIMENSIONLESS,
    np.log: Rule.DIMENSIONLESS,
    np.log2: Rule.DIMENSIONLESS,
    np.log10: Rule.DIMENSIONLESS,
    np.log1p: Rule.DIMENSIONLESS,
    np.exp: Rule.DIMENSIONLESS,
    np.exp2: Rule.DIMENSIONLESS,
    np.expm1: Rule.DIMENSIONLESS,
    np.logaddexp: Rule.DIMENSIONLESS_BOTH,
    np.logaddexp2: Rule.DIMENSIONLESS_BOTH,
    np.arctan2: Rule.DIMENSIONLESS_BOTH,
    np.hypot: Rule.DIMENSIONLESS_BOTH,
    np.bitwise_and: Rule.INTEGER_ONLY,
    np.bitwise_or: Rule.INTEGER_ONLY,
    np.bitwise_xor: Rule.INTEGER_ONLY,
    np.invert: Rule.INTEGER_ONLY,
    np.left_shift: Rule.INTEGER_ONLY,
    np.right_shift: Rule.INTEGER_ONLY,
}


# ------------------------------------------------------------------------------
# The old name lists. Code outside QuantSI imports them (Brian2's test suite
# does), so they stay exactly as they were. Dispatching doesn't use them: RULES is
# what counts, and test_ufunc_rules.py checks that the two agree. Some names never
# matched on NumPy 2 (true_divide, mod, conj and isreal are aliases or not ufuncs,
# and dot is a function).
# ------------------------------------------------------------------------------

# Note: A list of numpy ufuncs can be found here:
# http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs

#: ufuncs that work on all dimensions and preserve the dimensions, e.g. abs
UFUNCS_PRESERVE_DIMENSIONS = [
    "absolute",
    "rint",
    "negative",
    "positive",
    "conj",
    "conjugate",
    "floor",
    "ceil",
    "trunc",
]

#: ufuncs that work on all dimensions but change the dimensions, e.g. square
UFUNCS_CHANGE_DIMENSIONS = [
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "sqrt",
    "square",
    "reciprocal",
    "dot",
    "matmul",
]

#: ufuncs that work with matching dimensions, e.g. add
UFUNCS_MATCHING_DIMENSIONS = [
    "add",
    "subtract",
    "maximum",
    "minimum",
    "remainder",
    "mod",
    "fmod",
]

#: ufuncs that compare values, i.e. work only with matching dimensions but do
#: not result in a value with dimensions, e.g. equals
UFUNCS_COMPARISONS = [
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
]

#: Logical operations that work on all quantities and return boolean arrays
UFUNCS_LOGICAL = [
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "isreal",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
]

#: ufuncs that only work on dimensionless quantities
UFUNCS_DIMENSIONLESS = [
    "sin",
    "sinh",
    "arcsin",
    "arcsinh",
    "cos",
    "cosh",
    "arccos",
    "arccosh",
    "tan",
    "tanh",
    "arctan",
    "arctanh",
    "log",
    "log2",
    "log10",
    "log1p",
    "exp",
    "exp2",
    "expm1",
]

#: ufuncs that only work on two dimensionless quantities
UFUNCS_DIMENSIONLESS_TWOARGS = ["logaddexp", "logaddexp2", "arctan2", "hypot"]

#: ufuncs that only work on integers and therefore never on quantities
UFUNCS_INTEGERS = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]
