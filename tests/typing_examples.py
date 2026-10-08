"""The types of QuantSI's API, as a type checker sees them.

mypy checks this file in CI (``mypy``, see pyproject.toml); pytest does not
collect it. Note: results of arithmetic (``3 * mvolt``) are typed as NumPy
arrays, because NumPy's type stubs type all operators that way.
"""

from typing import assert_type

from QuantSI import (
    Quantity,
    Unit,
    get_dimensions,
    mvolt,
    parse_dimensions,
    uV,
    volt,
)
from QuantSI.fundamentalunits import Dimension, get_or_create_dimension

assert_type(volt, Unit)
assert_type(mvolt, Unit)
assert_type(uV, Unit)
assert_type(volt.dim, Dimension)
assert_type(get_dimensions(volt), Dimension)
assert_type(get_or_create_dimension(m=1), Dimension)
assert_type(parse_dimensions("3 * mV"), Dimension)
assert_type(Unit.create(volt.dim, "myvolt", "myV"), Unit)
assert_type(volt.in_unit(mvolt), str)
assert_type(volt.dim * volt.dim, Dimension)


def needs_a_quantity(q: Quantity) -> str:
    return q.in_best_unit()


needs_a_quantity(mvolt)  # a Unit is a Quantity
