"""The Quantity class: a NumPy array that carries a physical dimension.

Values are always stored in base SI units, and ``.dim`` points to the shared
`Dimension` object. NumPy calls ``Quantity.__array_ufunc__`` for arithmetic, and
that's where dimensions are checked and combined.
"""

import numbers
import operator
import sys
from warnings import warn

import numpy as np
from numpy.exceptions import VisibleDeprecationWarning

from ._dimension import (
    DIMENSIONLESS,
    Dimension,
    fail_for_dimension_mismatch,
    get_dimensions,
    get_or_create_dimension,
    is_scalar_type,
)
from ._errors import DimensionMismatchError
from ._registry import additional_unit_register, standard_unit_register, user_unit_register
from ._utils import _flatten, set_module

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


def wrap_function_keep_dimensions(func):
    """
    Returns a new function that wraps the given function `func` so that it
    keeps the dimensions of its input. Quantities are transformed to
    unitless numpy arrays before calling `func`, the output is a quantity
    with the original dimensions re-attached.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched, allowing to work functions like
    ``sum`` to work as expected with additional ``axis`` etc. arguments.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        return Quantity(func(np.asarray(x), *args, **kwds), dim=x.dim)

    f._arg_units = [None]
    f._return_unit = lambda u: u
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


@set_module("QuantSI.fundamentalunits")
def quantity_with_dimensions(floatval, dims):
    """
    Create a new `Quantity` with the given dimensions. Calls
    `get_or_create_dimensions` with the dimension tuple of the `dims`
    argument to make sure that unpickling (which calls this function) does not
    accidentally create new Dimension objects which should instead refer to
    existing ones.

    Parameters
    ----------
    floatval : `float`
        The floating point value of the quantity.
    dims : `Dimension`
        The physical dimensions of the quantity.

    Returns
    -------
    q : `Quantity`
        A quantity with the given dimensions.

    Examples
    --------
    >>> from QuantSI import *
    >>> quantity_with_dimensions(0.001, volt.dim)
    1. * mvolt

    See Also
    --------
    get_or_create_dimensions
    """
    return Quantity(floatval, get_or_create_dimension(dims._dims))


class Quantity(np.ndarray):
    """
    A number with an associated physical dimension. In most cases, it is not
    necessary to create a Quantity object by hand, instead use multiplication
    and division of numbers with the constant unit names ``second``,
    ``kilogram``, etc.

    Notes
    -----
    The `Quantity` class defines arithmetic operations which check for
    consistency of dimensions and raise the DimensionMismatchError exception
    if they are inconsistent. It also defines default and other representations
    for a number for printing purposes.

    See the documentation on the Unit class for more details
    about the available unit names like mvolt, etc.

    *Casting rules*

    The rules that define the casting operations for
    Quantity object are:

    1. Quantity op Quantity = Quantity
       Performs dimension checking if appropriate
    2. (Scalar or Array) op Quantity = Quantity
       Assumes that the scalar or array is dimensionless

    There is one exception to the above rule, the number ``0`` is interpreted
    as having "any dimension".

    Examples
    --------
    >>> from QuantSI import *
    >>> I = 3 * amp  # I is a Quantity object
    >>> R = 2 * ohm  # same for R
    >>> I * R
    6. * volt
    >>> (I * R).in_unit(mvolt)
    '6000. mV'
    >>> (I * R) / mvolt
    6000.0
    >>> X = I + R  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Addition, dimensions were (A) (m^2 kg s^-3 A^-2)
    >>> Is = np.array([1, 2, 3]) * amp
    >>> Is * R
    array([2., 4., 6.]) * volt
    >>> np.asarray(Is * R)  # gets rid of units
    array([2., 4., 6.])

    See also
    --------
    Unit

    Attributes
    ----------
    dimensions
    is_dimensionless
    dim : Dimensions
        The physical dimensions of this quantity.

    Methods
    -------
    with_dimensions
    has_same_dimensions
    in_unit
    in_best_unit
    """

    __slots__ = ["dim"]

    __array_priority__ = 1000

    # ==========================================================================
    # Construction and handling of numpy ufuncs
    # ==========================================================================
    def __new__(cls, arr, dim=None, dtype=None, copy=False, force_quantity=False):
        # Do not create dimensionless quantities, use pure numpy arrays instead
        if dim is DIMENSIONLESS and not force_quantity:
            if copy:
                arr = np.array(arr, dtype=dtype)
            else:
                arr = np.asarray(arr, dtype=dtype)
            if arr.shape == ():
                # For scalar values, return a simple Python object instead of
                # a numpy scalar
                return arr.item()
            return arr

        # All np.ndarray subclasses need something like this, see
        # http://www.scipy.org/Subclasses
        if copy:
            subarr = np.array(arr, dtype=dtype).view(cls)
        else:
            subarr = np.asarray(arr, dtype=dtype).view(cls)
        # We only want numerical datatypes
        if not issubclass(np.dtype(subarr.dtype).type, (np.number, np.bool_)):
            raise TypeError("Quantities can only be created from numerical data.")

        # If a dimension is given, force this dimension
        if dim is not None:
            subarr.dim = dim
            return subarr

        # Use the given dimension or the dimension of the given array (if any)
        try:
            subarr.dim = arr.dim
        except AttributeError:
            if not isinstance(arr, (np.ndarray, np.number, numbers.Number)):
                # check whether it is an iterable containing Quantity objects
                try:
                    is_quantity = [isinstance(x, Quantity) for x in _flatten(arr)]
                except TypeError:
                    # Not iterable
                    is_quantity = [False]
                if len(is_quantity) == 0:
                    # Empty list
                    subarr.dim = DIMENSIONLESS
                elif all(is_quantity):
                    dims = [x.dim for x in _flatten(arr)]
                    one_dim = dims[0]
                    for d in dims:
                        if d != one_dim:
                            raise DimensionMismatchError(
                                "Mixing quantities with different dimensions is not allowed",
                                d,
                                one_dim,
                            )
                    subarr.dim = dims[0]
                elif any(is_quantity):
                    raise TypeError("Mixing quantities and non-quantities is not allowed.")

        return subarr

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, "dim", DIMENSIONLESS)

    def __array_ufunc__(self, uf, method, *inputs, **kwargs):
        if method not in ("__call__", "reduce"):
            return NotImplemented
        uf_method = getattr(uf, method)
        if "out" in kwargs:
            # In contrast to numpy, we will not change a scalar value in-place,
            # i.e. a scalar Quantity will act like a Python float and not like
            # a numpy scalar in that regard.
            if self.ndim == 0:
                del kwargs["out"]
            else:
                # The output needs to be an array to avoid infinite recursion
                # Note that it is also part of the input arguments, so we don't
                # need to check its dimensions
                assert len(kwargs["out"]) == 1
                kwargs["out"] = (np.asarray(kwargs["out"][0]),)
        if uf.__name__ in (UFUNCS_LOGICAL + ["sign", "ones_like"]):
            # do not touch return value
            return uf_method(*[np.asarray(a) for a in inputs], **kwargs)
        elif uf.__name__ in UFUNCS_PRESERVE_DIMENSIONS:
            return Quantity(
                uf_method(*[np.asarray(a) for a in inputs], **kwargs),
                dim=self.dim,
            )
        elif uf.__name__ in UFUNCS_CHANGE_DIMENSIONS + ["power"]:
            if uf.__name__ == "sqrt":
                dim = self.dim**0.5
            elif uf.__name__ == "power":
                fail_for_dimension_mismatch(
                    inputs[1],
                    error_message=(
                        "The exponent for a power operation has to be dimensionless but was {value}"
                    ),
                    value=inputs[1],
                )
                if np.asarray(inputs[1]).size != 1:
                    raise TypeError(
                        "Only length-1 arrays can be used as an exponent for quantities."
                    )
                dim = get_dimensions(inputs[0]) ** np.asarray(inputs[1])
            elif uf.__name__ == "square":
                dim = self.dim**2
            elif uf.__name__ in ("divide", "true_divide", "floor_divide"):
                dim = get_dimensions(inputs[0]) / get_dimensions(inputs[1])
            elif uf.__name__ == "reciprocal":
                dim = get_dimensions(inputs[0]) ** -1
            elif uf.__name__ in ("multiply", "dot", "matmul"):
                if method == "__call__":
                    dim = get_dimensions(inputs[0]) * get_dimensions(inputs[1])
                else:
                    dim = get_dimensions(inputs[0])
            else:
                return NotImplemented
            return Quantity(uf_method(*[np.asarray(a) for a in inputs], **kwargs), dim=dim)
        elif uf.__name__ in UFUNCS_INTEGERS:
            # Numpy should already raise a TypeError by itself
            raise TypeError(f"{uf.__name__} cannot be used on quantities.")
        elif uf.__name__ in UFUNCS_MATCHING_DIMENSIONS + UFUNCS_COMPARISONS:
            # Ok if dimension of arguments match (for reductions, they always do)
            if method == "__call__":
                fail_for_dimension_mismatch(
                    inputs[0],
                    inputs[1],
                    error_message=("Cannot calculate {val1} %s {val2}, the units do not match")
                    % uf.__name__,
                    val1=inputs[0],
                    val2=inputs[1],
                )
            if uf.__name__ in UFUNCS_COMPARISONS:
                return uf_method(*[np.asarray(i) for i in inputs], **kwargs)
            else:
                return Quantity(
                    uf_method(*[np.asarray(i) for i in inputs], **kwargs),
                    dim=self.dim,
                )
        elif uf.__name__ in UFUNCS_DIMENSIONLESS:
            # Ok if argument is dimensionless
            fail_for_dimension_mismatch(
                inputs[0],
                error_message="%s expects a dimensionless argument but got {value}" % uf.__name__,
                value=inputs[0],
            )
            return uf_method(np.asarray(inputs[0]), *inputs[1:], **kwargs)
        elif uf.__name__ in UFUNCS_DIMENSIONLESS_TWOARGS:
            # Ok if both arguments are dimensionless
            fail_for_dimension_mismatch(
                inputs[0],
                error_message=(
                    'Both arguments for "%s" should be dimensionless but first argument was {value}'
                )
                % uf.__name__,
                value=inputs[0],
            )
            fail_for_dimension_mismatch(
                inputs[1],
                error_message=(
                    "Both arguments for "
                    '"%s" should be '
                    "dimensionless but "
                    "second argument was "
                    "{value}"
                )
                % uf.__name__,
                value=inputs[1],
            )
            return uf_method(
                np.asarray(inputs[0]),
                np.asarray(inputs[1]),
                *inputs[2:],
                **kwargs,
            )
        else:
            return NotImplemented

    def __deepcopy__(self, memo):
        return Quantity(self, copy=True)

    # ==============================================================================
    # Quantity-specific functions (not existing in ndarray)
    # ==============================================================================
    @staticmethod
    def with_dimensions(value, *args, **keywords):
        """
        Create a `Quantity` object with dim.

        Parameters
        ----------
        value : {array_like, number}
            The value of the dimension
        args : {`Dimension`, sequence of float}
            Either a single argument (a `Dimension`) or a sequence of 7 values.
        kwds
            Keywords defining the dim, see `Dimension` for details.

        Returns
        -------
        q : `Quantity`
            A `Quantity` object with the given dim

        Examples
        --------
        All of these define an equivalent `Quantity` object:

        >>> from QuantSI import *
        >>> Quantity.with_dimensions(2, get_or_create_dimension(length=1))
        2. * metre
        >>> Quantity.with_dimensions(2, length=1)
        2. * metre
        >>> 2 * metre
        2. * metre
        """
        if len(args) and isinstance(args[0], Dimension):
            dimensions = args[0]
        else:
            dimensions = get_or_create_dimension(*args, **keywords)
        return Quantity(value, dim=dimensions)

    ### ATTRIBUTES ###
    is_dimensionless = property(
        lambda self: self.dim.is_dimensionless,
        doc="Whether this is a dimensionless quantity.",
    )

    @property
    def dimensions(self):
        """
        The physical dimensions of this quantity.
        """
        return self.dim

    @dimensions.setter
    def dimensions(self, dim):
        self.dim = dim

    #### METHODS ####

    def has_same_dimensions(self, other):
        """
        Return whether this object has the same dimensions as another.

        Parameters
        ----------
        other : {`Quantity`, array-like, number}
            The object to compare the dimensions against.

        Returns
        -------
        same : `bool`
            ``True`` if `other` has the same dimensions.
        """
        other_dim = get_dimensions(other)
        return (self.dim is other_dim) or (self.dim == other_dim)

    def in_unit(self, u, precision=None, python_code=False):
        """
        Represent the quantity in a given unit. If `python_code` is ``True``,
        this will return valid python code, i.e. a string like ``5.0 * um ** 2``
        instead of ``5.0 um^2``

        Parameters
        ----------
        u : {`Quantity`, `Unit`}
            The unit in which to show the quantity.
        precision : `int`, optional
            The number of digits of precision (in the given unit, see Examples).
            If no value is given, numpy's `get_printoptions` value is used.
        python_code : `bool`, optional
            Whether to return valid python code (``True``) or a human readable
            string (``False``, the default).

        Returns
        -------
        s : `str`
            String representation of the object in unit `u`.

        Examples
        --------
        >>> from QuantSI.allunits import *
        >>> from QuantSI.stdunits import *
        >>> x = 25.123456 * mV
        >>> x.in_unit(volt)
        '0.02512346 V'
        >>> x.in_unit(volt, 3)
        '0.025 V'
        >>> x.in_unit(mV, 3)
        '25.123 mV'

        See Also
        --------
        in_unit
        """

        from ._unit import Unit

        fail_for_dimension_mismatch(self, u, 'Non-matching unit for method "in_unit"')

        value = np.asarray(self / u)
        # numpy uses the printoptions setting only in arrays, not in array
        # scalars, so we use this hackish way of turning the scalar first into
        # an array, then removing the square brackets from the output
        if value.shape == ():
            s = np.array_str(np.array([value]), precision=precision)
            s = s.replace("[", "").replace("]", "").strip()
        else:
            if python_code:
                s = np.array_repr(value, precision=precision)
            else:
                s = np.array_str(value, precision=precision)

        if not u.is_dimensionless:
            if isinstance(u, Unit):
                if python_code:
                    s += f" * {repr(u)}"
                else:
                    s += f" {str(u)}"
            else:
                if python_code:
                    s += f" * {repr(u.dim)}"
                else:
                    s += f" {str(u.dim)}"
        elif python_code:  # Make a quantity without unit recognisable
            return f"{self.__class__.__name__}({s.strip()})"
        return s.strip()

    def get_best_unit(self, *regs):
        """
        Return the best unit for this `Quantity`.

        Parameters
        ----------
        regs : any number of `UnitRegistry` objects
            The registries that are searched for units. If none are provided, it
            will check the standard, user and additional unit registers in turn.

        Returns
        -------
            u : `Quantity` or `Unit`
                The best-fitting unit for the quantity `x`.
        """
        from ._unit import Unit

        if self.is_dimensionless:
            return Unit(1)
        if len(regs):
            for r in regs:
                try:
                    return r[self]
                except KeyError:
                    pass
            return Quantity(1, self.dim)
        else:
            return self.get_best_unit(
                standard_unit_register, user_unit_register, additional_unit_register
            )

    def _get_best_unit(self, *regs):
        warn(
            "Quantity._get_best_unit has been renamed to Quantity.get_best_unit.",
            VisibleDeprecationWarning,
        )
        return self.get_best_unit(*regs)

    def in_best_unit(self, precision=None, python_code=False, *regs):
        """
        Represent the quantity in the "best" unit.

        Parameters
        ----------
        python_code : `bool`, optional
            If set to ``False`` (the default), will return a string like
            ``5.0 um^2`` which is not a valid Python expression. If set to
            ``True``, it will return ``5.0 * um ** 2`` instead.
        precision : `int`, optional
            The number of digits of precision (in the best unit, see
            Examples). If no value is given, numpy's
            `get_printoptions` value is used.
        regs : `UnitRegistry` objects
            The registries where to search for units. If none are given, the
            standard, user-defined and additional registries are searched in
            that order.

        Returns
        -------
        representation : `str`
            A string representation of this `Quantity`.

        Examples
        --------
        >>> from QuantSI.allunits import *

        >>> x = 0.00123456 * volt

        >>> x.in_best_unit()
        '1.23456 mV'

        >>> x.in_best_unit(3)
        '1.235 mV'

        See Also
        --------
        in_best_unit
        """
        u = self.get_best_unit(*regs)
        return self.in_unit(u, precision=precision, python_code=python_code)

    # ==============================================================================
    # Overwritten ndarray methods
    # ==============================================================================

    #### Setting/getting items ####
    def __getitem__(self, key):
        """Overwritten to assure that single elements (i.e., indexed with a
        single integer or a tuple of integers) retain their unit.
        """
        return Quantity(np.ndarray.__getitem__(self, key), self.dim)

    def item(self, *args):
        """Overwritten to assure that the returned element retains its unit."""
        return Quantity(np.ndarray.item(self, *args), self.dim)

    def __setitem__(self, key, value):
        fail_for_dimension_mismatch(self, value, "Inconsistent units in assignment")
        return super().__setitem__(key, value)

    def tolist(self):
        """
        Convert the array into a list.

        Returns
        -------
        l : list of `Quantity`
            A (possibly nested) list equivalent to the original array.
        """

        def replace_with_quantity(seq, dim):
            """
            Replace all the elements in the list with an equivalent `Quantity`
            with the given `dim`.
            """
            # No recursion needed for single values
            if not isinstance(seq, list):
                return Quantity(seq, dim)

            def top_replace(s):
                """
                Recursivley descend into the list.
                """
                for i in s:
                    if not isinstance(i, list):
                        yield Quantity(i, dim)
                    else:
                        yield type(i)(top_replace(i))

            return type(seq)(top_replace(seq))

        return replace_with_quantity(np.asarray(self).tolist(), self.dim)

    #### COMPARISONS ####
    def _comparison(self, other, operator_str, operation):
        is_scalar = is_scalar_type(other)
        if not is_scalar and not isinstance(other, np.ndarray):
            return NotImplemented
        if not is_scalar or not np.isinf(other):
            message = (
                "Cannot perform comparison {value1} %s {value2}, units do not match" % operator_str
            )
            fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)
        return operation(np.asarray(self), np.asarray(other))

    def __lt__(self, other):
        return self._comparison(other, "<", operator.lt)

    def __le__(self, other):
        return self._comparison(other, "<=", operator.le)

    def __gt__(self, other):
        return self._comparison(other, ">", operator.gt)

    def __ge__(self, other):
        return self._comparison(other, ">=", operator.ge)

    def __eq__(self, other):
        return self._comparison(other, "==", operator.eq)

    def __ne__(self, other):
        return self._comparison(other, "!=", operator.ne)

    #### MAKE QUANTITY PICKABLE ####
    def __reduce__(self):
        return quantity_with_dimensions, (np.asarray(self), self.dim)

    #### REPRESENTATION ####
    def __repr__(self):
        return self.in_best_unit(python_code=True)

    def _latex(self, expr):
        """
        Translates a scalar, 1-d or 2-d array into a LaTeX representation. Will be called
        by ``sympy``'s `~sympy.latex` function and used as a "rich representation" in e.g.
        jupyter notebooks.
        The values in the array will be formatted with `numpy.array2string` and will
        therefore observe ``numpy``'s "print options" such as ``precision``. Including
        all numbers in the LaTeX output will rarely be useful for large arrays; this
        function will therefore apply a ``threshold`` value divided by 100 (the default
        ``threshold`` value is 1000, this function hence applies 10). Note that the
        ``max_line_width`` print option is ignored.
        """
        from ._unit import Unit

        best_unit = self.get_best_unit()
        if isinstance(best_unit, Unit):
            best_unit_latex = best_unit._latex()
        else:  # A quantity
            best_unit_latex = best_unit.dimensions._latex()
        unitless = np.asarray(self / best_unit)
        threshold = np.get_printoptions()["threshold"] // 100
        if unitless.ndim == 0:
            sympy_quantity = float(unitless)
        elif unitless.ndim == 1:
            array_str = np.array2string(
                unitless,
                separator=" & ",
                threshold=threshold,
                max_line_width=sys.maxsize,
            )
            # Replace [ and ]
            sympy_quantity = (
                r"\left[\begin{matrix}"
                + array_str[1:-1].replace("...", r"\dots")
                + r"\end{matrix}\right]"
            )
        elif unitless.ndim == 2:
            array_str = np.array2string(
                unitless,
                separator=" & ",
                threshold=threshold,
                max_line_width=sys.maxsize,
            )
            array_str = array_str[1:-1].replace("...", r"\dots")
            array_str = array_str.replace("[", "").replace("] &", r"\\").replace("]", "\n")
            lines = array_str.split("\n")
            n_cols = lines[0].count("&") + 1
            new_lines = []
            for line in lines:
                if line.strip() == r"\dots &":
                    new_lines.append(" & ".join([r"\vdots"] * n_cols) + r"\\")
                else:
                    new_lines.append(line)
            sympy_quantity = (
                r"\left[\begin{matrix}" + "\n" + "\n".join(new_lines) + r"\end{matrix}\right]"
            )
        else:
            raise NotImplementedError(
                f"Cannot create a LaTeX representation for a {unitless.ndim}-d matrix."
            )
        return f"{sympy_quantity}\\,{best_unit_latex}"

    def _repr_latex_(self):
        return f"${self._latex(None)}$"

    def __str__(self):
        return self.in_best_unit()

    def __format__(self, format_spec):
        # Avoid that formatted strings like f"{q}" use floating point formatting for the
        # quantity, i.e. discard the unit
        if format_spec == "":
            return str(self)
        else:
            return super().__format__(format_spec)

    #### Mathematic methods ####
    cumsum = wrap_function_keep_dimensions(np.ndarray.cumsum)
    trace = wrap_function_keep_dimensions(np.trace)

    def fill(self, values):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, "fill")
        super().fill(values)

    fill.__doc__ = np.ndarray.fill.__doc__
    fill._do_not_run_doctests = True

    def put(self, indices, values, *args, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, values, "fill")
        super().put(indices, values, *args, **kwds)

    put.__doc__ = np.ndarray.put.__doc__
    put._do_not_run_doctests = True

    def clip(self, a_min, a_max, *args, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, a_min, "clip")
        fail_for_dimension_mismatch(self, a_max, "clip")
        return Quantity(
            np.clip(
                np.asarray(self),
                np.asarray(a_min),
                np.asarray(a_max),
                *args,
                **kwds,
            ),
            self.dim,
        )

    clip.__doc__ = np.ndarray.clip.__doc__
    clip._do_not_run_doctests = True

    def dot(self, other, **kwds):  # pylint: disable=C0111
        return Quantity(
            np.array(self).dot(np.array(other), **kwds),
            self.dim * get_dimensions(other),
        )

    dot.__doc__ = np.ndarray.dot.__doc__
    dot._do_not_run_doctests = True

    def searchsorted(self, v, **kwds):  # pylint: disable=C0111
        fail_for_dimension_mismatch(self, v, "searchsorted")
        return super().searchsorted(np.asarray(v), **kwds)

    searchsorted.__doc__ = np.ndarray.searchsorted.__doc__
    searchsorted._do_not_run_doctests = True

    def prod(self, *args, **kwds):  # pylint: disable=C0111
        prod_result = super().prod(*args, **kwds)
        # Calculating the correct dimensions is not completly trivial (e.g.
        # like doing self.dim**self.size) because prod can be called on
        # multidimensional arrays along a certain axis.
        # Our solution: Use a "dummy matrix" containing a 1 (without units) at
        # each entry and sum it, using the same keyword arguments as provided.
        # The result gives the exponent for the dimensions.
        # This relies on sum and prod having the same arguments, which is true
        # now and probably remains like this in the future
        dim_exponent = np.ones_like(self).sum(*args, **kwds)
        # The result is possibly multidimensional but all entries should be
        # identical
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[0]
        return Quantity(np.asarray(prod_result), self.dim**dim_exponent)

    prod.__doc__ = np.ndarray.prod.__doc__
    prod._do_not_run_doctests = True

    def cumprod(self, *args, **kwds):  # pylint: disable=C0111
        if not self.is_dimensionless:
            raise TypeError(
                "cumprod over array elements on quantities with dimensions is not possible."
            )
        return Quantity(np.asarray(self).cumprod(*args, **kwds))

    cumprod.__doc__ = np.ndarray.cumprod.__doc__
    cumprod._do_not_run_doctests = True
