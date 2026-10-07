"""Turning quantities into text.

`format_quantity` is the one place that writes a quantity as text; ``str``,
``repr``, ``format``, ``in_unit`` and ``in_best_unit`` all go through it.
"""

import numpy as np

from ._dimension import fail_for_dimension_mismatch, is_dimensionless
from ._unit import Unit


def format_quantity(quantity, unit, precision=None, python_code=False):
    """Write ``quantity`` in ``unit`` (which must have the same dimensions).

    With ``python_code=False`` the text is for people (``'3. mV'``); with
    ``python_code=True`` it is an expression that evaluates back to the quantity
    (``'3. * mvolt'``). NumPy formats the numbers, following its print options;
    ``precision`` overrides the print option of that name.
    """
    value = np.asarray(quantity / unit)
    if value.shape == ():
        # A scalar is written as a one-element array without its brackets: NumPy
        # scalars ignore the print options, and 0-d arrays are written with the
        # scalar's repr in NumPy's legacy print mode (legacy="1.13", which Brian2's
        # tests use). One-element arrays follow the print options in every mode.
        s = np.array2string(value.reshape(1), precision=precision)[1:-1].strip()
    elif python_code:
        s = np.array_repr(value, precision=precision)
    else:
        s = np.array_str(value, precision=precision)

    if not unit.is_dimensionless:
        if isinstance(unit, Unit):
            if python_code:
                s += f" * {repr(unit)}"
            else:
                s += f" {str(unit)}"
        else:
            if python_code:
                s += f" * {repr(unit.dim)}"
            else:
                s += f" {str(unit.dim)}"
    elif python_code:  # Make a quantity without unit recognisable
        return f"{type(quantity).__name__}({s.strip()})"
    return s.strip()


def in_unit(x, u, precision=None):
    """
    Display a value in a certain unit with a given precision.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    u : {`Quantity`, `Unit`}
        The unit to display the value `x` in.
    precision : `int`, optional
        The number of digits of precision (in the given unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    s : `str`
        A string representation of `x` in units of `u`.

    Examples
    --------
    >>> from QuantSI import *
    >>> in_unit(3 * volt, mvolt)
    '3000. mV'
    >>> in_unit(123123 * msecond, second, 2)
    '123.12 s'
    >>> in_unit(10 * uA / cm**2, nA / um**2)
    '1.e-04 nA/(um^2)'
    >>> in_unit(10 * mV, ohm * amp)
    '0.01 ohm A'
    >>> in_unit(10 * nS, ohm)  # doctest: +NORMALIZE_WHITESPACE
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Non-matching unit for method "in_unit",
    dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

    See Also
    --------
    Quantity.in_unit
    """
    if is_dimensionless(x):
        fail_for_dimension_mismatch(x, u, 'Non-matching unit for function "in_unit"')
        return str(np.asarray(x / u))
    else:
        return x.in_unit(u, precision=precision)


def in_best_unit(x, precision=None):
    """
    Represent the value in the "best" unit.

    Parameters
    ----------
    x : {`Quantity`, array-like, number}
        The value to display
    precision : `int`, optional
        The number of digits of precision (in the best unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    representation : `str`
        A string representation of this `Quantity`.

    Examples
    --------
    >>> from QuantSI.allunits import *
    >>> in_best_unit(0.00123456 * volt)
    '1.23456 mV'
    >>> in_best_unit(0.00123456 * volt, 2)
    '1.23 mV'
    >>> in_best_unit(0.123456)
    '0.123456'
    >>> in_best_unit(0.123456, 2)
    '0.12'

    See Also
    --------
    Quantity.in_best_unit
    """
    if is_dimensionless(x):
        if precision is None:
            precision = np.get_printoptions()["precision"]
        return str(np.round(x, precision))

    u = x.get_best_unit()
    return x.in_unit(u, precision=precision)
