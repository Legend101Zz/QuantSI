"""Dimension behaviour, including equality checked against the old implementation."""

import numpy as np
import pytest

from QuantSI.allunits import amp, metre, volt
from QuantSI.fundamentalunits import DIMENSIONLESS, get_or_create_dimension


def old_eq(dim1, dim2):
    """Dimension.__eq__ as it was before it was rewritten for speed."""
    try:
        return np.allclose(dim1._dims, dim2._dims)
    except AttributeError:
        return False


def dimension_pairs():
    rng = np.random.default_rng(1)
    pairs = [(volt.dim, volt.dim), (volt.dim, amp.dim), (DIMENSIONLESS, metre.dim)]
    # Floating-point drift: equal but not identical tuples.
    pairs.append((get_or_create_dimension(length=0.1 + 0.2), get_or_create_dimension(length=0.3)))
    for _ in range(300):
        base = rng.integers(-4, 5, size=7).astype(float) * rng.choice([1, 0.5, 1 / 3])
        # Offsets around the tolerance boundary of np.allclose, absolute and relative.
        offset = rng.choice([0, 1e-9, 1e-8, 2e-8, 1e-6, 1e-5, 1e-4]) * rng.choice([-1, 1])
        other = base + offset * rng.integers(0, 2, size=7)
        pairs.append((get_or_create_dimension(tuple(base)), get_or_create_dimension(tuple(other))))
    return pairs


@pytest.mark.parametrize(("dim1", "dim2"), dimension_pairs())
def test_equality_matches_old_implementation(dim1, dim2):
    assert (dim1 == dim2) == old_eq(dim1, dim2)
    assert (dim1 != dim2) == (not old_eq(dim1, dim2))


def test_equality_with_other_types():
    assert volt.dim != (2, 1, -3, -1, 0, 0, 0)
    assert volt.dim != volt
    assert not (volt.dim == None)  # noqa: E711


def test_interning_makes_equal_dimensions_identical():
    assert get_or_create_dimension(m=2, kg=1, s=-3, A=-1) is volt.dim
