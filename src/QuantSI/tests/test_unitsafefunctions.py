"""Unit-aware replacements for NumPy functions (from Brian2)."""

import numpy as np
import pytest

from QuantSI import DimensionMismatchError, unitsafefunctions as unitsafe
from QuantSI.allunits import metre, msecond, mvolt, second

DIMENSIONLESS_FUNCTIONS = [
    (unitsafe.sin, np.sin),
    (unitsafe.sinh, np.sinh),
    (unitsafe.arcsin, np.arcsin),
    (unitsafe.arcsinh, np.arcsinh),
    (unitsafe.cos, np.cos),
    (unitsafe.cosh, np.cosh),
    (unitsafe.arccos, np.arccos),
    (unitsafe.arccosh, np.arccosh),
    (unitsafe.tan, np.tan),
    (unitsafe.tanh, np.tanh),
    (unitsafe.arctan, np.arctan),
    (unitsafe.arctanh, np.arctanh),
    (unitsafe.log, np.log),
    (unitsafe.log10, np.log10),
    (unitsafe.exp, np.exp),
    (unitsafe.expm1, np.expm1),
    (unitsafe.log1p, np.log1p),
]


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@pytest.mark.parametrize(("function", "numpy_function"), DIMENSIONLESS_FUNCTIONS)
def test_functions_of_dimensionless_values(function, numpy_function):
    for value in [3 * mvolt, np.array([1, 2]) * mvolt]:
        with pytest.raises(DimensionMismatchError):
            function(value)
    for value in [0.3, np.array([0.1, 0.2]), np.ones((2, 2)) * 0.5]:
        np.testing.assert_allclose(function(value), numpy_function(value))


@pytest.mark.parametrize(("function", "numpy_function"), DIMENSIONLESS_FUNCTIONS)
def test_code_generation_metadata(function, numpy_function):
    # Brian2's code generation reads these attributes.
    assert function._arg_units == [1]
    assert function._return_unit == 1
    assert function.__name__ == numpy_function.__name__


def test_exprel():
    assert unitsafe.exprel(0) == 1.0
    np.testing.assert_allclose(
        unitsafe.exprel(np.array([1e-20, 1.0, 1000.0])), [1.0, np.e - 1, np.inf]
    )
    assert unitsafe.exprel._arg_units == [1] and unitsafe.exprel._return_unit == 1
    with pytest.raises(DimensionMismatchError):
        unitsafe.exprel(3 * mvolt)


def test_arange():
    # np.arange never sees the units of its arguments; this one does. With units,
    # the step needs them too (the default step is the plain number 1).
    result = unitsafe.arange(3 * msecond, step=1 * msecond)
    assert result.dim is second.dim
    np.testing.assert_allclose(np.asarray(result), [0, 0.001, 0.002])
    result = unitsafe.arange(1 * msecond, 3 * msecond, 1 * msecond)
    np.testing.assert_allclose(np.asarray(result), [0.001, 0.002])
    np.testing.assert_array_equal(unitsafe.arange(5), np.arange(5))
    for arguments in [(1 * msecond, 3 * metre), (1 * msecond, 3 * msecond), (1, 3 * msecond)]:
        with pytest.raises(DimensionMismatchError):
            unitsafe.arange(*arguments)


def test_linspace():
    result = unitsafe.linspace(0 * mvolt, 2 * mvolt, 3)
    assert result.dim is mvolt.dim
    np.testing.assert_allclose(np.asarray(result), [0, 0.001, 0.002])
    with pytest.raises(DimensionMismatchError):
        unitsafe.linspace(0 * mvolt, 2 * second, 3)


def test_where():
    condition = np.array([True, False])
    a, b = np.array([1.0, 2.0]) * mvolt, np.array([3.0, 4.0]) * mvolt
    result = unitsafe.where(condition, a, b)
    assert result.dim is mvolt.dim
    np.testing.assert_allclose(np.asarray(result), [0.001, 0.004])
    with pytest.raises(DimensionMismatchError):
        unitsafe.where(condition, a, np.ones(2) * second)


def test_unit_free_and_unit_keeping_helpers():
    q = np.arange(4.0).reshape(2, 2) * mvolt
    ones = unitsafe.ones_like(q)
    assert not hasattr(ones, "dim") and ones.shape == (2, 2)
    assert unitsafe.ptp(q).dim is mvolt.dim
    assert unitsafe.ravel(q).dim is mvolt.dim
    assert unitsafe.diagonal(q).dim is mvolt.dim
    assert unitsafe.trace(q).dim is mvolt.dim
    assert unitsafe.dot(q, q).dim is (mvolt**2).dim
