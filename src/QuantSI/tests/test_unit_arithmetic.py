"""Arithmetic with Unit operands: units combine into units, everything else into quantities."""

import numpy as np
import pytest

from QuantSI.allunits import metre, mvolt, ohm, second, volt
from QuantSI.fundamentalunits import Quantity, Unit

CASES = [
    # (expression, expected type, expected value in SI, expected dimension)
    (lambda: 3 * mvolt, Quantity, 0.003, mvolt.dim),
    (lambda: mvolt * 3, Quantity, 0.003, mvolt.dim),
    (lambda: np.array([1.0, 2.0]) * mvolt, Quantity, [0.001, 0.002], mvolt.dim),
    (lambda: (2 * ohm) * mvolt, Quantity, 0.002, (ohm * volt).dim),
    (lambda: 3 / mvolt, Quantity, 3000.0, (1 / volt).dim),
    (lambda: mvolt / 2, Quantity, 0.0005, mvolt.dim),
    (lambda: (4 * metre) / second, Quantity, 4.0, (metre / second).dim),
    (lambda: metre / second, Unit, 1.0, (metre / second).dim),
    (lambda: mvolt * ohm, Unit, 0.001, (volt * ohm).dim),
    (lambda: 1 / second, Unit, 1.0, (1 / second).dim),
]


@pytest.mark.parametrize(("expression", "kind", "value", "dim"), CASES)
def test_unit_arithmetic(expression, kind, value, dim):
    result = expression()
    assert type(result) is kind
    assert result.dim is dim
    np.testing.assert_allclose(np.asarray(result), value)
