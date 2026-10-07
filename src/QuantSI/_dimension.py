"""Physical dimensions: the exponents of the seven SI base units.

A `Dimension` says what kind of quantity something is (a length, a voltage,
...), not how much of it there is. `get_or_create_dimension` makes sure there is
only one `Dimension` object for each combination of exponents, so checking
whether two things have the same dimensions is just ``dim1 is dim2``.
"""

import numpy as np
from sympy import latex

from ._utils import set_module

# The seven SI base dimensions (length, mass, time, current, temperature,
# amount of substance, luminosity) and various descriptions;
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = {
    "Length": 0,
    "length": 0,
    "metre": 0,
    "metres": 0,
    "meter": 0,
    "meters": 0,
    "m": 0,
    "Mass": 1,
    "mass": 1,
    "kilogram": 1,
    "kilograms": 1,
    "kg": 1,
    "Time": 2,
    "time": 2,
    "second": 2,
    "seconds": 2,
    "s": 2,
    "Electric Current": 3,
    "electric current": 3,
    "Current": 3,
    "current": 3,
    "ampere": 3,
    "amperes": 3,
    "A": 3,
    "Temperature": 4,
    "temperature": 4,
    "kelvin": 4,
    "kelvins": 4,
    "K": 4,
    "Quantity of Substance": 5,
    "Quantity of substance": 5,
    "quantity of substance": 5,
    "Substance": 5,
    "substance": 5,
    "mole": 5,
    "moles": 5,
    "mol": 5,
    "Luminosity": 6,
    "luminosity": 6,
    "candle": 6,
    "candles": 6,
    "cd": 6,
}

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole", "candle"]

# SI unit _prefixes as integer exponents of 10, see table at end of file.
_siprefixes = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}


class Dimension:
    """
    Stores the indices of the 7 basic SI unit dimension (length, mass, etc.).

    Provides a subset of arithmetic operations appropriate to dimensions:
    multiplication, division and powers, and equality testing.

    Parameters
    ----------
    dims : sequence of `float`
        The dimension indices of the 7 basic SI unit dimensions.

    Notes
    -----
    Users shouldn't use this class directly, it is used internally in Quantity
    and Unit. Even internally, never use ``Dimension(...)`` to create a new
    instance, use `get_or_create_dimension` instead. This function makes
    sure that only one Dimension instance exists for every combination of
    indices, allowing for a very fast dimensionality check with ``is``.
    """

    __slots__ = ["_dims"]

    __array_priority__ = 1000
    #### INITIALISATION ####

    def __init__(self, dims):
        self._dims = dims

    #### METHODS ####
    def get_dimension(self, d):
        """
        Return a specific dimension.

        Parameters
        ----------
        d : `str`
            A string identifying the SI basic unit dimension. Can be either a
            description like "length" or a basic unit like "m" or "metre".

        Returns
        -------
        dim : `float`
            The dimensionality of the dimension `d`.
        """
        return self._dims[_di[d]]

    @property
    def is_dimensionless(self):
        """
        Whether this Dimension is dimensionless.

        Notes
        -----
        Normally, instead one should check dimension for being identical to
        `DIMENSIONLESS`.
        """
        return all([x == 0 for x in self._dims])

    @property
    def dim(self):
        """
        Returns the `Dimension` object itself. This can be useful, because it
        allows to check for the dimension of an object by checking its ``dim``
        attribute -- this will return a `Dimension` object for `Quantity`,
        `Unit` and `Dimension`.
        """
        return self

    #### REPRESENTATION ####
    def _str_representation(self, python_code=False):
        """
        String representation in basic SI units, or ``"1"`` for dimensionless.
        Use ``python_code=False`` for display purposes and ``True`` for valid
        Python code.
        """

        if python_code:
            power_operator = " ** "
        else:
            power_operator = "^"

        parts = []
        for i in range(len(self._dims)):
            if self._dims[i]:
                if python_code:
                    s = _iclass_label[i]
                else:
                    s = _ilabel[i]
                if self._dims[i] != 1:
                    s += power_operator + str(self._dims[i])
                parts.append(s)
        if python_code:
            s = " * ".join(parts)
            if not len(s):
                return f"{self.__class__.__name__}()"
        else:
            s = " ".join(parts)
            if not len(s):
                return "1"
        return s.strip()

    def _latex(self, *args):
        parts = []
        for i in range(len(self._dims)):
            if self._dims[i]:
                s = _ilabel[i]
                if self._dims[i] != 1:
                    s += "^{%s}" % str(self._dims[i])
                parts.append(s)
        s = "\\,".join(parts)
        if not len(s):
            return "1"
        return s.strip()

    def _repr_latex(self):
        return f"${latex(self)}$"

    def __repr__(self):
        return self._str_representation(python_code=True)

    def __str__(self):
        return self._str_representation(python_code=False)

    #### ARITHMETIC ####
    # Note that none of the dimension arithmetic objects do sanity checking
    # on their inputs, although most will throw an exception if you pass the
    # wrong sort of input
    def __mul__(self, value):
        return get_or_create_dimension([x + y for x, y in zip(self._dims, value._dims)])

    def __div__(self, value):
        return get_or_create_dimension([x - y for x, y in zip(self._dims, value._dims)])

    def __truediv__(self, value):
        return self.__div__(value)

    def __pow__(self, value):
        value = np.asarray(value)
        if value.size > 1:
            raise TypeError("Too many exponents")
        return get_or_create_dimension([x * value for x in self._dims])

    def __imul__(self, value):
        raise TypeError("Dimension object is immutable")

    def __ifloordiv__(self, value):
        raise TypeError("Dimension object is immutable")

    def __itruediv__(self, value):
        raise TypeError("Dimension object is immutable")

    def __ipow__(self, value):
        raise TypeError("Dimension object is immutable")

    #### COMPARISON ####
    def __eq__(self, value):
        try:
            return np.allclose(self._dims, value._dims)
        except AttributeError:
            # Only compare equal to another Dimensions object
            return False

    def __ne__(self, value):
        return not self.__eq__(value)

    def __hash__(self):
        return hash(self._dims)

    #### MAKE DIMENSION PICKABLE ####
    def __reduce__(self):
        # Make sure that unpickling Dimension objects does not bypass the singleton system
        return (get_or_create_dimension, (self._dims,))

    ### Dimension objects are singletons and deepcopy is therefore not necessary
    def __deepcopy__(self, memodict):
        return self


#: The singleton object for dimensionless Dimensions.
DIMENSIONLESS = Dimension((0, 0, 0, 0, 0, 0, 0))

_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}


@set_module("QuantSI.fundamentalunits")
def get_or_create_dimension(*args, **kwds):
    """
    Create a new Dimension object or get a reference to an existing one.
    This function takes care of only creating new objects if they were not
    created before and otherwise returning a reference to an existing object.
    This allows to compare dimensions very efficiently using ``is``.

    Parameters
    ----------
    args : sequence of `float`
        A sequence with the indices of the 7 elements of an SI dimension.
    kwds : keyword arguments
        a sequence of ``keyword=value`` pairs where the keywords are the names of
        the SI dimensions, or the standard unit.

    Examples
    --------
    The following are all definitions of the dimensions of force

    >>> from QuantSI import *
    >>> get_or_create_dimension(length=1, mass=1, time=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension(m=1, kg=1, s=-2)
    metre * kilogram * second ** -2
    >>> get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
    metre * kilogram * second ** -2

    Notes
    -----
    The 7 units are (in order):

    Length, Mass, Time, Electric Current, Temperature,
    Quantity of Substance, Luminosity

    and can be referred to either by these names or their SI unit names,
    e.g. length, metre, and m all refer to the same thing here.
    """
    if len(args):
        # initialisation by list
        dims = args[0]
        try:
            if len(dims) != 7:
                raise TypeError()
        except TypeError:
            raise TypeError("Need a sequence of exactly 7 items")
    else:
        # initialisation by keywords
        dims = [0, 0, 0, 0, 0, 0, 0]
        for k in kwds:
            # _di stores the index of the dimension with name 'k'
            dims[_di[k]] = kwds[k]

    dims = tuple(dims)

    # check whether this Dimension object has already been created
    if dims in _dimensions:
        return _dimensions[dims]
    else:
        new_dim = Dimension(dims)
        _dimensions[dims] = new_dim
        return new_dim
