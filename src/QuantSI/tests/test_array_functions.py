"""NumPy functions on quantities: each function in a bucket has a case here.

Each case calls a NumPy function twice, on quantities and on the same values as
plain arrays, and checks the outcome against an expectation:

``"m"``, ``"m2"``, ``"m s"``, ...  a Quantity with these dimensions whose values
                                   (in base SI units) equal the plain call's result
``"plain"``                        no Quantity anywhere in the result
``"none"``                         the function returns None (it modifies in place)
``"TypeError"``, ...               the call raises this exception
a tuple of the above               the function returns a tuple
"""

import pathlib
import types
import warnings

import numpy as np
import pytest

from QuantSI import DimensionMismatchError, _array_functions
from QuantSI.allunits import metre, second
from QuantSI.fundamentalunits import DIMENSIONLESS, Quantity

DIMS = {
    "m": metre.dim,
    "m2": (metre**2).dim,
    "m4": (metre**4).dim,
    "m s": (metre * second).dim,
    "s": second.dim,
    "1/m": (1 / metre).dim,
    "s/m": (second / metre).dim,
    "m/s": (metre / second).dim,
    "m^0.5": (metre**0.5).dim,
    "1": DIMENSIONLESS,
}
EXCEPTIONS = {"TypeError": TypeError, "DimensionMismatchError": DimensionMismatchError}


def without_deprecations(function, *args):
    """Call a NumPy function that newer NumPy versions deprecate (e.g. np.fix in 2.5)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return function(*args)


def namespace(length, time, tmp_path):
    ns = types.SimpleNamespace(unit=length, second=time, tmp=tmp_path)
    ns.m = np.array([1.0, 2.0, 3.0, 4.0]) * length
    ns.m2 = np.arange(1.0, 7.0).reshape(2, 3) * length
    ns.s = np.array([1.0, 2.0, 4.0, 8.0]) * time
    ns.M = np.array([[2.0, 1.0], [1.0, 3.0]]) * length
    ns.d = np.array([0.1, 0.2, 0.3, 0.4])
    ns.i = np.array([0, 1, 1, 3])
    ns.b = np.array([True, False, True, False])
    return ns


#: function name (as reached from numpy) -> [(label, call, expectation), ...]
CASES = {
    # ---- SUBCLASS_SAFE ----------------------------------------------------------
    "all": [("1d", lambda ns: np.all(ns.m > 0 * ns.unit), "plain")],
    "any": [("1d", lambda ns: np.any(ns.m > 0 * ns.unit), "plain")],
    "amax": [("1d", lambda ns: np.amax(ns.m), "m")],
    "amin": [("1d", lambda ns: np.amin(ns.m), "m")],
    "max": [("1d", lambda ns: np.max(ns.m), "m")],
    "min": [("1d", lambda ns: np.min(ns.m), "m")],
    "sum": [("1d", lambda ns: np.sum(ns.m), "m"), ("axis", lambda ns: np.sum(ns.m2, axis=0), "m")],
    "prod": [("1d", lambda ns: np.prod(ns.m), "m4")],
    "mean": [("1d", lambda ns: np.mean(ns.m), "m")],
    "std": [("1d", lambda ns: np.std(ns.m), "m")],
    "var": [("1d", lambda ns: np.var(ns.m), "m2")],
    "median": [("1d", lambda ns: np.median(ns.m), "m")],
    "percentile": [("1d", lambda ns: np.percentile(ns.m, 50), "m")],
    "quantile": [("1d", lambda ns: np.quantile(ns.m, 0.5), "m")],
    "average": [
        ("1d", lambda ns: np.average(ns.m), "m"),
        ("weights", lambda ns: np.average(ns.m, weights=ns.d), "m"),
    ],
    "ptp": [("1d", lambda ns: np.ptp(ns.m), "m")],
    "nanmax": [("1d", lambda ns: np.nanmax(ns.m), "m")],
    "nanmin": [("1d", lambda ns: np.nanmin(ns.m), "m")],
    "nansum": [("1d", lambda ns: np.nansum(ns.m), "m")],
    "nanprod": [("1d", lambda ns: np.nanprod(ns.m), "m4")],
    "nanmean": [("1d", lambda ns: np.nanmean(ns.m), "m")],
    "nanstd": [("1d", lambda ns: np.nanstd(ns.m), "m")],
    "nanvar": [("1d", lambda ns: np.nanvar(ns.m), "m2")],
    "nanmedian": [("1d", lambda ns: np.nanmedian(ns.m), "m")],
    "nanpercentile": [("1d", lambda ns: np.nanpercentile(ns.m, 50), "m")],
    "nanquantile": [("1d", lambda ns: np.nanquantile(ns.m, 0.5), "m")],
    "trace": [("2d", lambda ns: np.trace(ns.M), "m")],
    "linalg.trace": [("2d", lambda ns: np.linalg.trace(ns.M), "m")],
    "cumsum": [("1d", lambda ns: np.cumsum(ns.m), "m")],
    "cumulative_sum": [("1d", lambda ns: np.cumulative_sum(ns.m), "m")],
    "nancumsum": [("1d", lambda ns: np.nancumsum(ns.m), "m")],
    "cumulative_prod": [
        ("with dimensions", lambda ns: np.cumulative_prod(ns.m * ns.m), "TypeError")
    ],
    "diff": [("1d", lambda ns: np.diff(ns.m), "m")],
    "ediff1d": [("1d", lambda ns: np.ediff1d(ns.m), "m")],
    "trapezoid": [("1d", lambda ns: np.trapezoid(ns.m, ns.s), "m s")],
    "around": [("1d", lambda ns: np.around(ns.m, 1), "m")],
    "round": [("1d", lambda ns: np.round(ns.m, 1), "m")],
    "fix": [("1d", lambda ns: without_deprecations(np.fix, ns.m), "m")],
    "clip": [
        ("1d", lambda ns: np.clip(ns.m, 1 * ns.unit, 3 * ns.unit), "m"),
        (
            "mismatch",
            lambda ns: np.clip(ns.m, 1 * ns.second, 3 * ns.second),
            "DimensionMismatchError",
        ),
    ],
    "nan_to_num": [("1d", lambda ns: np.nan_to_num(ns.m), "m")],
    "real": [("1d", lambda ns: np.real(ns.m), "m")],
    "imag": [("1d", lambda ns: np.imag(ns.m), "m")],
    "real_if_close": [("1d", lambda ns: np.real_if_close(ns.m), "m")],
    "reshape": [("1d", lambda ns: np.reshape(ns.m, (2, 2)), "m")],
    "ravel": [("2d", lambda ns: np.ravel(ns.m2), "m")],
    "transpose": [("2d", lambda ns: np.transpose(ns.m2), "m")],
    "matrix_transpose": [("2d", lambda ns: np.matrix_transpose(ns.m2), "m")],
    "linalg.matrix_transpose": [("2d", lambda ns: np.linalg.matrix_transpose(ns.m2), "m")],
    "swapaxes": [("2d", lambda ns: np.swapaxes(ns.m2, 0, 1), "m")],
    "moveaxis": [("2d", lambda ns: np.moveaxis(ns.m2, 0, 1), "m")],
    "rollaxis": [("2d", lambda ns: np.rollaxis(ns.m2, 1), "m")],
    "squeeze": [("2d", lambda ns: np.squeeze(ns.m2[:1]), "m")],
    "expand_dims": [("1d", lambda ns: np.expand_dims(ns.m, 0), "m")],
    "atleast_1d": [("1d", lambda ns: np.atleast_1d(ns.m), "m")],
    "atleast_2d": [("1d", lambda ns: np.atleast_2d(ns.m), "m")],
    "atleast_3d": [("1d", lambda ns: np.atleast_3d(ns.m), "m")],
    "flip": [("1d", lambda ns: np.flip(ns.m), "m")],
    "fliplr": [("2d", lambda ns: np.fliplr(ns.m2), "m")],
    "flipud": [("2d", lambda ns: np.flipud(ns.m2), "m")],
    "rot90": [("2d", lambda ns: np.rot90(ns.m2), "m")],
    "roll": [("1d", lambda ns: np.roll(ns.m, 1), "m")],
    "repeat": [("1d", lambda ns: np.repeat(ns.m, 2), "m")],
    "tile": [("1d", lambda ns: np.tile(ns.m, 2), "m")],
    "diag": [("2d", lambda ns: np.diag(ns.m2), "m")],
    "diagflat": [("1d", lambda ns: np.diagflat(ns.m), "m")],
    "diagonal": [("2d", lambda ns: np.diagonal(ns.m2), "m")],
    "linalg.diagonal": [("2d", lambda ns: np.linalg.diagonal(ns.M), "m")],
    "take": [("1d", lambda ns: np.take(ns.m, [0, 1]), "m")],
    "take_along_axis": [("1d", lambda ns: np.take_along_axis(ns.m, np.array([0, 1]), 0), "m")],
    "compress": [("1d", lambda ns: np.compress(ns.b, ns.m), "m")],
    "extract": [("1d", lambda ns: np.extract(ns.b, ns.m), "m")],
    "delete": [("1d", lambda ns: np.delete(ns.m, 1), "m")],
    "trim_zeros": [("1d", lambda ns: np.trim_zeros(ns.m), "m")],
    "sort": [("1d", lambda ns: np.sort(ns.m), "m")],
    "partition": [("1d", lambda ns: np.partition(ns.m, 1), "m")],
    "searchsorted": [
        ("1d", lambda ns: np.searchsorted(ns.m, 2 * ns.unit), "plain"),
        ("mismatch", lambda ns: np.searchsorted(ns.m, 2 * ns.second), "DimensionMismatchError"),
    ],
    "put": [
        ("1d", lambda ns: np.put(ns.m.copy(), [0], 9 * ns.unit), "none"),
        ("mismatch", lambda ns: np.put(ns.m.copy(), [0], 9 * ns.second), "DimensionMismatchError"),
    ],
    "split": [("1d", lambda ns: np.split(ns.m, 2), ("m", "m"))],
    "array_split": [("1d", lambda ns: np.array_split(ns.m, 2), ("m", "m"))],
    "hsplit": [("1d", lambda ns: np.hsplit(ns.m, 2), ("m", "m"))],
    "vsplit": [("2d", lambda ns: np.vsplit(ns.m2, 2), ("m", "m"))],
    "dsplit": [("3d", lambda ns: np.dsplit(np.atleast_3d(ns.m2), 1), ("m",))],
    "unstack": [("2d", lambda ns: np.unstack(ns.m2), ("m", "m"))],
    "astype": [("1d", lambda ns: np.astype(ns.m, np.float32), "m")],
    "piecewise": [("1d", lambda ns: np.piecewise(ns.m, [ns.b], [lambda v: v]), "m")],
    "apply_along_axis": [("sum", lambda ns: np.apply_along_axis(np.sum, 0, ns.m2), "m")],
    "meshgrid": [("1d", lambda ns: np.meshgrid(ns.m, ns.s), ("m", "s"))],
    "linspace": [
        ("1d", lambda ns: np.linspace(1 * ns.unit, 4 * ns.unit, 4), "m"),
        (
            "mismatch",
            lambda ns: np.linspace(1 * ns.unit, 4 * ns.second, 4),
            "DimensionMismatchError",
        ),
    ],
    "unique": [("1d", lambda ns: np.unique(ns.m), "m")],
    "unique_values": [("1d", lambda ns: np.unique_values(ns.m), "m")],
    "unique_counts": [("1d", lambda ns: np.unique_counts(ns.m), ("m", "plain"))],
    "unique_inverse": [("1d", lambda ns: np.unique_inverse(ns.m), ("m", "plain"))],
    "empty_like": [("1d", lambda ns: np.empty_like(ns.m).shape, "plain")],
    "zeros_like": [("1d", lambda ns: np.zeros_like(ns.m), "m")],
    "ones_like": [("1d", lambda ns: np.ones_like(ns.m), "m")],
    "full_like": [("1d", lambda ns: np.full_like(ns.m, 1 * ns.unit), "m")],
    "kron": [("1d", lambda ns: np.kron(ns.m, ns.s), "m s")],
    "linalg.matmul": [("2d", lambda ns: np.linalg.matmul(ns.M, ns.M), "m2")],
    "linalg.matrix_power": [("2d", lambda ns: np.linalg.matrix_power(ns.M, 2), "m2")],
    "linalg.vecdot": [("1d", lambda ns: np.linalg.vecdot(ns.m, ns.s), "m s")],
    # ---- UNIT_FREE ------------------------------------------------------------------
    "argwhere": [("1d", lambda ns: np.argwhere(ns.m), "plain")],
    "flatnonzero": [("1d", lambda ns: np.flatnonzero(ns.m), "plain")],
    "nonzero": [("1d", lambda ns: np.nonzero(ns.m), ("plain",))],
    "lexsort": [("1d", lambda ns: np.lexsort([ns.m]), "plain")],
    "nanargmax": [("1d", lambda ns: np.nanargmax(ns.m), "plain")],
    "nanargmin": [("1d", lambda ns: np.nanargmin(ns.m), "plain")],
    "ix_": [("1d", lambda ns: np.ix_(ns.i, ns.i), ("plain", "plain"))],
    "diag_indices_from": [("2d", lambda ns: np.diag_indices_from(ns.M), ("plain", "plain"))],
    "tril_indices_from": [("2d", lambda ns: np.tril_indices_from(ns.M), ("plain", "plain"))],
    "triu_indices_from": [("2d", lambda ns: np.triu_indices_from(ns.M), ("plain", "plain"))],
    "unravel_index": [("1d", lambda ns: np.unravel_index(ns.i, (5, 5)), ("plain", "plain"))],
    "ravel_multi_index": [("1d", lambda ns: np.ravel_multi_index((ns.i, ns.i), (5, 5)), "plain")],
    "count_nonzero": [("1d", lambda ns: np.count_nonzero(ns.m), "plain")],
    "isneginf": [("1d", lambda ns: np.isneginf(ns.m), "plain")],
    "isposinf": [("1d", lambda ns: np.isposinf(ns.m), "plain")],
    "iscomplex": [("1d", lambda ns: np.iscomplex(ns.m), "plain")],
    "isreal": [("1d", lambda ns: np.isreal(ns.m), "plain")],
    "iscomplexobj": [("1d", lambda ns: np.iscomplexobj(ns.m), "plain")],
    "isrealobj": [("1d", lambda ns: np.isrealobj(ns.m), "plain")],
    "may_share_memory": [("1d", lambda ns: np.may_share_memory(ns.m, ns.m), "plain")],
    "shares_memory": [("1d", lambda ns: np.shares_memory(ns.m, ns.m), "plain")],
    "corrcoef": [("1d", lambda ns: np.corrcoef(ns.m, ns.m), "plain")],
    "angle": [("1d", lambda ns: np.angle(ns.m), "plain")],
    "linalg.cond": [("2d", lambda ns: np.linalg.cond(ns.M), "plain")],
    "linalg.matrix_rank": [("2d", lambda ns: np.linalg.matrix_rank(ns.M), "plain")],
    "ndim": [("1d", lambda ns: np.ndim(ns.m), "plain")],
    "size": [("1d", lambda ns: np.size(ns.m), "plain")],
    "shape": [("1d", lambda ns: np.shape(ns.m), "plain")],
    "can_cast": [("1d", lambda ns: np.can_cast(ns.m, np.float64), "plain")],
    "common_type": [("1d", lambda ns: np.common_type(ns.m), "plain")],
    "min_scalar_type": [("1d", lambda ns: np.min_scalar_type(ns.m), "plain")],
    "result_type": [("1d", lambda ns: np.result_type(ns.m), "plain")],
    "einsum_path": [("dot", lambda ns: np.einsum_path("i,i", ns.m, ns.s), "plain")],
    "array2string": [("1d", lambda ns: np.array2string(ns.m), "plain")],
    "array_repr": [("1d", lambda ns: np.array_repr(ns.m), "plain")],
    "array_str": [("1d", lambda ns: np.array_str(ns.m), "plain")],
    "save": [("1d", lambda ns: np.save(ns.tmp / "a.npy", ns.m), "none")],
    "savetxt": [("1d", lambda ns: np.savetxt(ns.tmp / "a.txt", ns.m), "none")],
    "savez": [("1d", lambda ns: np.savez(ns.tmp / "a.npz", ns.m), "none")],
    "savez_compressed": [("1d", lambda ns: np.savez_compressed(ns.tmp / "b.npz", ns.m), "none")],
    "arange": [("like", lambda ns: np.arange(3, like=ns.m), "plain")],
    "array": [("like", lambda ns: np.array([1.0], like=ns.m), "plain")],
    "asarray": [("like", lambda ns: np.asarray([1.0], like=ns.m), "plain")],
    "asanyarray": [("like", lambda ns: np.asanyarray([1.0], like=ns.m), "plain")],
    "ascontiguousarray": [("like", lambda ns: np.ascontiguousarray([1.0], like=ns.m), "plain")],
    "asfortranarray": [("like", lambda ns: np.asfortranarray([1.0], like=ns.m), "plain")],
    "empty": [("like", lambda ns: np.empty(3, like=ns.m).shape, "plain")],
    "zeros": [("like", lambda ns: np.zeros(2, like=ns.m), "plain")],
    "ones": [("like", lambda ns: np.ones(2, like=ns.m), "plain")],
    "full": [("like", lambda ns: np.full(2, 1.0, like=ns.m), "plain")],
    "eye": [("like", lambda ns: np.eye(2, like=ns.m), "plain")],
    "identity": [("like", lambda ns: np.identity(2, like=ns.m), "plain")],
    "tri": [("like", lambda ns: np.tri(2, like=ns.m), "plain")],
    "require": [("like", lambda ns: np.require([1.0], like=ns.m), "plain")],
    "frombuffer": [("like", lambda ns: np.frombuffer(b"\0" * 8, like=ns.m), "plain")],
    "fromfile": [
        ("like", lambda ns: np.fromfile(__file__, dtype=np.uint8, count=1, like=ns.m), "plain")
    ],
    "fromfunction": [("like", lambda ns: np.fromfunction(lambda a: a, (2,), like=ns.m), "plain")],
    "fromiter": [("like", lambda ns: np.fromiter([1.0], float, like=ns.m), "plain")],
    "fromstring": [("like", lambda ns: np.fromstring("1 2", sep=" ", like=ns.m), "plain")],
    "loadtxt": [("like", lambda ns: np.loadtxt(["1 2"], like=ns.m), "plain")],
    "genfromtxt": [("like", lambda ns: np.genfromtxt(["1 2"], like=ns.m), "plain")],
    # ---- UNSUPPORTED ----------------------------------------------------------------
    "busday_count": [("1d", lambda ns: np.busday_count(ns.m, ns.m), "TypeError")],
    "busday_offset": [("1d", lambda ns: np.busday_offset(ns.m, ns.m), "TypeError")],
    "is_busday": [("1d", lambda ns: np.is_busday(ns.m), "TypeError")],
    "datetime_as_string": [("1d", lambda ns: np.datetime_as_string(ns.m), "TypeError")],
    "packbits": [("1d", lambda ns: np.packbits(ns.m), "TypeError")],
    "unpackbits": [("1d", lambda ns: np.unpackbits(ns.m), "TypeError")],
    "poly": [("1d", lambda ns: np.poly(ns.m), "TypeError")],
    "polyadd": [("1d", lambda ns: np.polyadd(ns.m, ns.m), "TypeError")],
    "polyder": [("1d", lambda ns: np.polyder(ns.m), "TypeError")],
    "polydiv": [("1d", lambda ns: np.polydiv(ns.m, ns.m), "TypeError")],
    "polyfit": [("1d", lambda ns: np.polyfit(ns.d, ns.m, 1), "TypeError")],
    "polyint": [("1d", lambda ns: np.polyint(ns.m), "TypeError")],
    "polymul": [("1d", lambda ns: np.polymul(ns.m, ns.d), "TypeError")],
    "polysub": [("1d", lambda ns: np.polysub(ns.m, ns.m), "TypeError")],
    "polyval": [("1d", lambda ns: np.polyval(ns.m, 2), "TypeError")],
    "roots": [("1d", lambda ns: np.roots(ns.m), "TypeError")],
    "vander": [("1d", lambda ns: np.vander(ns.m, 3), "TypeError")],
    "apply_over_axes": [("1d", lambda ns: np.apply_over_axes(np.sum, ns.m2, [0]), "TypeError")],
    "logspace": [("1d", lambda ns: np.logspace(0 * ns.unit, 1 * ns.unit, 3), "TypeError")],
    "linalg.lstsq": [("2d", lambda ns: np.linalg.lstsq(ns.M, ns.m[:2]), "TypeError")],
    "linalg.tensorinv": [("2d", lambda ns: np.linalg.tensorinv(ns.M, 1), "TypeError")],
    "linalg.tensorsolve": [("2d", lambda ns: np.linalg.tensorsolve(ns.M, ns.m[:2]), "TypeError")],
    "linalg.slogdet": [("2d", lambda ns: np.linalg.slogdet(ns.M), "TypeError")],
}


def numpy_function(name):
    module = np
    for part in name.split(".")[:-1]:
        module = getattr(module, part)
    return getattr(module, name.split(".")[-1])


def all_cases():
    for name, cases in CASES.items():
        for label, call, expected in cases:
            yield pytest.param(call, expected, id=f"{name}-{label}")


def check(result, plain, expected):
    if isinstance(expected, tuple):
        assert isinstance(result, (tuple, list)) and len(result) == len(expected)
        for item, plain_item, expected_item in zip(result, plain, expected, strict=True):
            check(item, plain_item, expected_item)
    elif expected == "none":
        assert result is None
    elif expected == "plain":
        assert not isinstance(result, Quantity)
        if isinstance(result, np.ndarray) and result.dtype.kind in "biufc":
            np.testing.assert_allclose(result, plain)
    else:
        assert isinstance(result, Quantity), f"expected a Quantity, got {type(result)}"
        assert result.dim is DIMS[expected], f"dimensions {result.dim}, expected {expected}"
        np.testing.assert_allclose(np.asarray(result), np.asarray(plain))


@pytest.mark.parametrize(("call", "expected"), list(all_cases()))
def test_numpy_function(call, expected, tmp_path):
    quantities = namespace(metre, second, tmp_path)
    if expected in EXCEPTIONS:
        with pytest.raises(EXCEPTIONS[expected]):
            call(quantities)
        return
    result = call(quantities)
    plain = call(namespace(1.0, 1.0, pathlib.Path(tmp_path)))
    check(result, plain, expected)


def test_every_classified_function_has_a_case():
    classified = (
        _array_functions.SUBCLASS_SAFE
        | _array_functions.UNIT_FREE
        | set(_array_functions.UNSUPPORTED)
        | set(_array_functions.HANDLED)
    )
    covered = {numpy_function(name) for name in CASES}
    assert not {f.__name__ for f in classified - covered}


def test_buckets_do_not_overlap():
    buckets = [
        _array_functions.SUBCLASS_SAFE,
        _array_functions.UNIT_FREE,
        set(_array_functions.UNSUPPORTED),
        set(_array_functions.HANDLED),
    ]
    for i, first in enumerate(buckets):
        for second_bucket in buckets[i + 1 :]:
            assert not first & second_bucket
