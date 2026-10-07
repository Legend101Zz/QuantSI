"""
Defines physical units and quantities

=====================  ========  ======
Quantity               Unit      Symbol
---------------------  --------  ------
Length                 metre     m
Mass                   kilogram  kg
Time                   second    s
Electric current       ampere    A
Temperature            kelvin    K
Quantity of substance  mole      mol
Luminosity             candle    cd
=====================  ========  ======
"""

from typing import Callable

import numpy as np
from sympy import latex

from ._dimension import (
    DIMENSIONLESS as DIMENSIONLESS,
    Dimension as Dimension,
    _di as _di,
    _dimensions as _dimensions,
    _iclass_label as _iclass_label,
    _ilabel as _ilabel,
    _siprefixes as _siprefixes,
    fail_for_dimension_mismatch as fail_for_dimension_mismatch,
    get_dimensions as get_dimensions,
    get_or_create_dimension as get_or_create_dimension,
    have_same_dimensions as have_same_dimensions,
    is_dimensionless as is_dimensionless,
    is_scalar_type as is_scalar_type,
)
from ._errors import DimensionMismatchError as DimensionMismatchError
from ._formatting import in_best_unit as in_best_unit, in_unit as in_unit
from ._quantity import (
    UFUNCS_CHANGE_DIMENSIONS as UFUNCS_CHANGE_DIMENSIONS,
    UFUNCS_COMPARISONS as UFUNCS_COMPARISONS,
    UFUNCS_DIMENSIONLESS as UFUNCS_DIMENSIONLESS,
    UFUNCS_DIMENSIONLESS_TWOARGS as UFUNCS_DIMENSIONLESS_TWOARGS,
    UFUNCS_INTEGERS as UFUNCS_INTEGERS,
    UFUNCS_LOGICAL as UFUNCS_LOGICAL,
    UFUNCS_MATCHING_DIMENSIONS as UFUNCS_MATCHING_DIMENSIONS,
    UFUNCS_PRESERVE_DIMENSIONS as UFUNCS_PRESERVE_DIMENSIONS,
    Quantity as Quantity,
    quantity_with_dimensions as quantity_with_dimensions,
    wrap_function_keep_dimensions as wrap_function_keep_dimensions,
)
from ._registry import (
    UnitRegistry as UnitRegistry,
    additional_unit_register as additional_unit_register,
    get_unit as get_unit,
    get_unit_for_display as get_unit_for_display,
    register_new_unit as register_new_unit,
    standard_unit_register as standard_unit_register,
    user_unit_register as user_unit_register,
)
from ._utils import _flatten as _flatten, _short_str as _short_str

__all__ = [
    "DimensionMismatchError",
    "get_or_create_dimension",
    "get_dimensions",
    "is_dimensionless",
    "have_same_dimensions",
    "in_unit",
    "in_best_unit",
    "Quantity",
    "Unit",
    "register_new_unit",
    "check_units",
    "is_scalar_type",
    "get_unit",
]


class Unit(Quantity):
    r"""
     A physical unit.

     Normally, you do not need to worry about the implementation of
     units. They are derived from the `Quantity` object with
     some additional information (name and string representation).

     Basically, a unit is just a number with given dimensions, e.g.
     mvolt = 0.001 with the dimensions of voltage. The units module
     defines a large number of standard units, and you can also define
     your own (see below).

     The unit class also keeps track of various things that were used
     to define it so as to generate a nice string representation of it.
     See below.

     When creating scaled units, you can use the following prefixes:

      ======     ======  ==============
      Factor     Name    Prefix
      ======     ======  ==============
      10^24      yotta   Y
      10^21      zetta   Z
      10^18      exa     E
      10^15      peta    P
      10^12      tera    T
      10^9       giga    G
      10^6       mega    M
      10^3       kilo    k
      10^2       hecto   h
      10^1       deka    da
      1
      10^-1      deci    d
      10^-2      centi   c
      10^-3      milli   m
      10^-6      micro   u (\mu in SI)
      10^-9      nano    n
      10^-12     pico    p
      10^-15     femto   f
      10^-18     atto    a
      10^-21     zepto   z
      10^-24     yocto   y
      ======     ======  ==============

    **Defining your own**

     It can be useful to define your own units for printing
     purposes. So for example, to define the newton metre, you
     write

     >>> from QuantSI import *
     >>> from QuantSI.allunits import newton
     >>> Nm = newton * metre

     You can then do

     >>> (1 * Nm).in_unit(Nm)
     '1. N m'

     New "compound units", i.e. units that are composed of other units will be
     automatically registered and from then on used for display. For example,
     imagine you define total conductance for a membrane, and the total area of
     that membrane:

     >>> conductance = 10.0 * nS
     >>> area = 20000 * um**2

     If you now ask for the conductance density, you will get an "ugly" display
     in basic SI dimensions, as Brian does not know of a corresponding unit:

     >>> conductance / area
     0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

     By using an appropriate unit once, it will be registered and from then on
     used for display when appropriate:

     >>> usiemens / cm**2
     usiemens / (cmetre ** 2)
     >>> conductance / area  # same as before, but now Brian knows about uS/cm^2
     50. * usiemens / (cmetre ** 2)

     Note that user-defined units cannot override the standard units (`volt`,
     `second`, etc.) that are predefined by Brian. For example, the unit
     ``Nm`` has the dimensions "length²·mass/time²", and therefore the same
     dimensions as the standard unit `joule`. The latter will be used for display
     purposes:

     >>> 3 * joule
     3. * joule
     >>> 3 * Nm
     3. * joule

    """

    __slots__ = ["dim", "scale", "_dispname", "_name", "_latexname", "iscompound"]

    __array_priority__ = 100

    automatically_register_units = True

    #### CONSTRUCTION ####
    def __new__(
        cls,
        arr,
        dim=None,
        scale=0,
        name=None,
        dispname=None,
        latexname=None,
        iscompound=False,
        dtype=None,
        copy=False,
    ):
        if dim is None:
            dim = DIMENSIONLESS
        obj = super().__new__(cls, arr, dim=dim, dtype=dtype, copy=copy, force_quantity=True)
        return obj

    def __array_finalize__(self, orig):
        self.dim = getattr(orig, "dim", DIMENSIONLESS)
        self.scale = getattr(orig, "scale", 0)
        self._name = getattr(orig, "_name", "")
        self._dispname = getattr(orig, "_dispname", "")
        self._latexname = getattr(orig, "_latexname", "")
        self.iscompound = getattr(orig, "_iscompound", False)
        return self

    def __init__(
        self,
        value,
        dim=None,
        scale=0,
        name=None,
        dispname=None,
        latexname="",
        iscompound=False,
    ):
        if value != 10.0**scale:
            raise AssertionError(f"Unit value has to be 10**scale (scale={scale}, value={value})")
        if dim is None:
            dim = DIMENSIONLESS
        self.dim = dim  #: The Dimensions of this unit

        #: The scale for this unit (as the integer exponent of 10), i.e.
        #: a scale of 3 means 10^3, e.g. for a "k" prefix.
        self.scale = scale
        if name is None:
            if dim is DIMENSIONLESS:
                name = "Unit(1)"
            else:
                name = repr(dim)
        if dispname is None:
            if dim is DIMENSIONLESS:
                dispname = "1"
            else:
                dispname = str(dim)
        #: The full name of this unit.
        self._name = name
        #: The display name of this unit.
        self._dispname = dispname
        #: A LaTeX expression for the name of this unit.
        self._latexname = latexname
        #: Whether this unit is a combination of other units.
        self.iscompound = iscompound

        if Unit.automatically_register_units:
            register_new_unit(self)

    @staticmethod
    def create(dim, name, dispname, latexname=None, scale=0):
        """
        Create a new named unit.

        Parameters
        ----------
        dim : `Dimension`
            The dimensions of the unit.
        name : `str`
            The full name of the unit, e.g. ``'volt'``
        dispname : `str`
            The display name, e.g. ``'V'``
        latexname : str, optional
            The name as a LaTeX expression (math mode is assumed, do not add
            $ signs or similar), e.g. ``'\\omega'``. If no `latexname` is
            specified, `dispname` will be used.
        scale : int, optional
            The scale of this unit as an exponent of 10, e.g. -3 for a unit that
            is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = str(name)
        dispname = str(dispname)
        if latexname is None:
            latexname = f"\\mathrm{{{dispname}}}"

        u = Unit(
            10.0**scale,
            dim=dim,
            scale=scale,
            name=name,
            dispname=dispname,
            latexname=latexname,
        )

        return u

    @staticmethod
    def create_scaled_unit(baseunit, scalefactor):
        """
        Create a scaled unit from a base unit.

        Parameters
        ----------
        baseunit : `Unit`
            The unit of which to create a scaled version, e.g. ``volt``,
            ``amp``.
        scalefactor : `str`
            The scaling factor, e.g. ``"m"`` for mvolt, mamp

        Returns
        -------
        u : `Unit`
            The new unit.
        """
        name = scalefactor + baseunit.name
        dispname = scalefactor + baseunit.dispname
        scale = _siprefixes[scalefactor] + baseunit.scale
        if scalefactor == "u":
            scalefactor = r"\mu"
        latexname = f"\\mathrm{{{scalefactor}}}{baseunit.latexname}"

        u = Unit(
            10.0**scale,
            dim=baseunit.dim,
            name=name,
            dispname=dispname,
            latexname=latexname,
            scale=scale,
        )

        return u

    #### METHODS ####

    name = property(fget=lambda self: self._name, doc="The name of the unit")

    dispname = property(
        fget=lambda self: self._dispname,
        doc="The display name of the unit",
    )

    latexname = property(
        fget=lambda self: self._latexname,
        doc="The LaTeX name of the unit",
    )

    #### REPRESENTATION ####
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.dispname

    def _latex(self, *args):
        return self.latexname

    def _repr_latex_(self):
        return f"${latex(self)}$"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        if ufunc.__name__ == "multiply":
            first, second = inputs
            if isinstance(first, Unit) and isinstance(second, Unit):
                name = f"{first.name} * {second.name}"
                dispname = f"{self.dispname} {second.dispname}"
                latexname = f"{first.latexname}\\,{second.latexname}"
                scale = first.scale + second.scale
                u = Unit(
                    10.0**scale,
                    dim=first.dim * second.dim,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    iscompound=True,
                    scale=scale,
                )
                return u
            else:
                return ufunc(
                    *[Quantity(i, dim=getattr(i, "dim", DIMENSIONLESS)) for i in inputs],
                    **kwargs,
                )
        elif ufunc.__name__ == "divide":
            first, second = inputs
            if isinstance(first, Unit) and isinstance(second, Unit):
                if first.iscompound:
                    dispname = f"({self.dispname})"
                    name = f"({self.name})"
                else:
                    dispname = self.dispname
                    name = self.name
                dispname += "/"
                name += " / "
                if second.iscompound:
                    dispname += f"({second.dispname})"
                    name += f"({second.name})"
                else:
                    dispname += second.dispname
                    name += second.name

                latexname = rf"\frac{{{first.latexname}}}{{{second.latexname}}}"
                scale = first.scale - second.scale
                u = Unit(
                    10.0**scale,
                    dim=first.dim / second.dim,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    scale=scale,
                    iscompound=True,
                )
                return u
            elif is_dimensionless(first) and np.array(first).shape == () and first == 1:
                return np.reciprocal(second)
            else:
                return ufunc(
                    *[Quantity(i, dim=getattr(i, "dim", DIMENSIONLESS)) for i in inputs],
                    **kwargs,
                )
        elif ufunc.__name__ == "power":
            first, second = inputs
            if is_scalar_type(second):
                if first.iscompound:
                    dispname = f"({first.dispname})"
                    name = f"({first.name})"
                    latexname = r"\left(%s\right)" % first.latexname
                else:
                    dispname = first.dispname
                    name = first.name
                    latexname = first.latexname
                dispname += f"^{str(second)}"
                name += f" ** {repr(second)}"
                latexname += "^{%s}" % latex(second)
                scale = first.scale * second
                u = Unit(
                    10.0**scale,
                    dim=first.dim**second,
                    name=name,
                    dispname=dispname,
                    latexname=latexname,
                    scale=scale,
                    iscompound=True,
                )  # To avoid issues with units like (second ** -1) ** -1
                return u
            else:
                return super().__pow__(second)
        elif ufunc.__name__ == "square":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^2"
            name += " ** 2"
            latexname += "^2"
            scale = self.scale * 2
            u = Unit(
                10.0**scale,
                dim=self.dim**2,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        elif ufunc.__name__ == "sqrt":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^0.5"
            name += " ** 0.5"
            latexname += "^0.5"
            scale = self.scale / 2
            u = Unit(
                10.0**scale,
                dim=self.dim**0.5,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        elif ufunc.__name__ == "reciprocal":
            if self.iscompound:
                dispname = f"({self.dispname})"
                name = f"({self.name})"
                latexname = r"\left(%s\right)" % self.latexname
            else:
                dispname = self.dispname
                name = self.name
                latexname = self.latexname
            dispname += "^-1"
            name += " ** -1"
            latexname += "^{-1}"
            scale = -self.scale
            u = Unit(
                10.0**scale,
                dim=self.dim**-1,
                name=name,
                dispname=dispname,
                latexname=latexname,
                scale=scale,
                iscompound=True,
            )
            return u
        else:
            # Treat the unit as a Quantity (e.g. meter + meter should not fail but give 2*meter)
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __iadd__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __isub__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __imul__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __itruediv__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __ifloordiv__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __imod__(self, other):
        raise TypeError("Units cannot be modified in-place")

    def __ipow__(self, other, modulo=None):
        raise TypeError("Units cannot be modified in-place")

    def __eq__(self, other):
        if isinstance(other, Unit):
            return other.dim is self.dim and other.scale == self.scale
        else:
            return Quantity.__eq__(self, other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.dim, self.scale))


#### DECORATORS


def check_units(**au):
    """Decorator to check units of arguments passed to a function

    Examples
    --------
    >>> from QuantSI import mV, nA
    >>> from QuantSI.allunits import *
    >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
    ... def getvoltage(I, R, **k):
    ...     return I * R

    You don't have to check the units of every variable in the function, and
    you can define what the units should be for variables that aren't
    explicitly named in the definition of the function. For example, the code
    above checks that the variable wibble should be a length, so writing

    >>> getvoltage(1 * amp, 1 * ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

    fails, but

    >>> getvoltage(1 * amp, 1 * ohm, wibble=1 * metre)
    1. * volt

    passes. String arguments or ``None`` are not checked

    >>> getvoltage(1 * amp, 1 * ohm, wibble="hello")
    1. * volt

    By using the special name ``result``, you can check the return value of the
    function.

    You can also use ``1`` or ``bool`` as a special value to check for a
    unitless number or a boolean value, respectively:

    >>> @check_units(value=1, absolute=bool, result=bool)
    ... def is_high(value, absolute=False):
    ...     if absolute:
    ...         return abs(value) >= 5
    ...     else:
    ...         return value >= 5

    This will then again raise an error if the argument if not of the expected
    type:

    >>> is_high(7)
    True
    >>> is_high(-7, True)
    True
    >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

    If the return unit depends on the unit of an argument, you can also pass
    a function that takes the units of all the arguments as its inputs (in the
    order specified in the function header):

    >>> @check_units(result=lambda d: d**2)
    ... def square(value):
    ...     return value**2

    If several arguments take arbitrary units but they have to be
    consistent among each other, you can state the name of another argument as
    a string to state that it uses the same unit as that argument.

    >>> @check_units(summand_1=None, summand_2="summand_1")
    ... def multiply_sum(multiplicand, summand_1, summand_2):
    ...     "Calculates multiplicand*(summand_1 + summand_2)"
    ...     return multiplicand * (summand_1 + summand_2)
    >>> multiply_sum(3, 4 * mV, 5 * mV)
    27. * mvolt
    >>> multiply_sum(3 * nA, 4 * mV, 5 * mV)
    27. * pwatt
    >>> multiply_sum(3 * nA, 4 * mV, 5 * nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    QuantSI.fundamentalunits.DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

    Raises
    ------

    DimensionMismatchError
        In case the input arguments or the return value do not have the
        expected dimensions.
    TypeError
        If an input argument or return value was expected to be a boolean but
        is not.

    Notes
    -----
    This decorator will destroy the signature of the original function, and
    replace it with the signature ``(*args, **kwds)``. Other decorators will
    do the same thing, and this decorator critically needs to know the signature
    of the function it is acting on, so it is important that it is the first
    decorator to act on a function. It cannot be used in combination with
    another decorator that also needs to know the signature of the function.

    Note that the ``bool`` type is "strict", i.e. it expects a proper
    boolean value and does not accept 0 or 1. This is not the case the other
    way round, declaring an argument or return value as "1" *does* allow for a
    ``True`` or ``False`` value.
    """

    def do_check_units(f):
        def new_f(*args, **kwds):
            newkeyset = kwds.copy()
            arg_names = f.__code__.co_varnames[0 : f.__code__.co_argcount]
            for n, v in zip(arg_names, args[0 : f.__code__.co_argcount]):
                if not isinstance(v, (Quantity, str, bool, np.bool_)) and v is not None and n in au:
                    try:
                        # allow e.g. to pass a Python list of values
                        v = Quantity(v)
                    except TypeError:
                        if have_same_dimensions(au[n], 1):
                            raise TypeError(f"Argument {n} is not a unitless value/array.")
                        else:
                            raise TypeError(
                                f"Argument '{n}' is not a quantity, "
                                "expected a quantity with dimensions "
                                f"{au[n]}"
                            )
                newkeyset[n] = v

            for k in newkeyset:
                # string variables are allowed to pass, the presumption is they
                # name another variable. None is also allowed, useful for
                # default parameters
                if (
                    k in au
                    and not isinstance(newkeyset[k], str)
                    and newkeyset[k] is not None
                    and au[k] is not None
                ):
                    if au[k] in (bool, np.bool_):
                        if not isinstance(newkeyset[k], (bool, np.bool_)):
                            value = newkeyset[k]
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected a boolean value "
                                f"for argument '{k}' but got "
                                f"'{value}'"
                            )
                            raise TypeError(error_message)
                    elif isinstance(au[k], str):
                        if au[k] not in newkeyset:
                            error_message = (
                                f"Function '{f.__name__}' "
                                "expected its argument to have the "
                                f"same units as argument '{k}', but "
                                "there is no argument of that name"
                            )
                            raise TypeError(error_message)
                        if not have_same_dimensions(newkeyset[k], newkeyset[au[k]]):
                            d1 = get_dimensions(newkeyset[k])
                            d2 = get_dimensions(newkeyset[au[k]])
                            error_message = (
                                f"Function '{f.__name__}' expected "
                                f"the argument '{k}' to have the same "
                                f"units as argument '{au[k]}', but "
                                f"argument '{k}' has "
                                f"unit {get_unit_for_display(d1)}, "
                                f"while argument '{au[k]}' "
                                f"has unit {get_unit_for_display(d2)}."
                            )
                            raise DimensionMismatchError(error_message)
                    elif not have_same_dimensions(newkeyset[k], au[k]):
                        unit = repr(au[k])
                        value = newkeyset[k]
                        error_message = (
                            f"Function '{f.__name__}' "
                            "expected a quantity with unit "
                            f"{unit} for argument '{k}' but got "
                            f"'{value}'"
                        )
                        raise DimensionMismatchError(error_message, get_dimensions(newkeyset[k]))

            result = f(*args, **kwds)
            if "result" in au:
                if isinstance(au["result"], Callable) and au["result"] not in (
                    bool,
                    np.bool_,
                ):
                    expected_result = au["result"](*[get_dimensions(a) for a in args])
                else:
                    expected_result = au["result"]
                if au["result"] in (bool, np.bool_):
                    if not isinstance(result, (bool, np.bool_)):
                        error_message = (
                            "The return value of function "
                            f"'{f.__name__}' was expected to be "
                            "a boolean value, but was of type "
                            f"{type(result)}"
                        )
                        raise TypeError(error_message)
                elif not have_same_dimensions(result, expected_result):
                    unit = get_unit_for_display(expected_result)
                    error_message = (
                        "The return value of function "
                        f"'{f.__name__}' was expected to have "
                        f"unit {unit} but was "
                        f"'{result}'"
                    )
                    raise DimensionMismatchError(error_message, get_dimensions(result))
            return result

        new_f._orig_func = f
        new_f.__doc__ = f.__doc__
        new_f.__name__ = f.__name__
        # store the information in the function, necessary when using the
        # function in expressions or equations
        if hasattr(f, "_orig_arg_names"):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
        new_f._arg_names = arg_names
        new_f._arg_units = [au.get(name, None) for name in arg_names]
        return_unit = au.get("result", None)
        if return_unit is None:
            new_f._return_unit = None
        else:
            new_f._return_unit = return_unit
        if return_unit is bool:
            new_f._returns_bool = True
        else:
            new_f._returns_bool = False
        new_f._orig_arg_names = arg_names

        # copy any annotation attributes
        if hasattr(f, "_annotation_attributes"):
            for attrname in f._annotation_attributes:
                setattr(new_f, attrname, getattr(f, attrname))
        new_f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
            "_arg_units",
            "_arg_names",
            "_return_unit",
            "_orig_func",
            "_returns_bool",
        ]
        return new_f

    return do_check_units
