"""How NumPy functions treat quantities (the ``__array_function__`` protocol, NEP 18).

NumPy functions such as ``np.concatenate`` or ``np.linalg.inv`` do not go through
ufuncs. Before NumPy calls them on a Quantity it asks ``Quantity.__array_function__``,
which looks the function up here. Every function is in one bucket:

``SUBCLASS_SAFE``
    NumPy's own implementation already gives the right result, because it works
    through ufuncs and methods that Quantity handles (``np.sum``, ``np.reshape``).
``HANDLED``
    QuantSI's implementation (registered with `implements`): it checks the
    dimensions, computes on plain arrays and attaches the result's dimensions.
``UNIT_FREE``
    The result has no units (indices, booleans, shapes, dtypes, strings, files),
    so the function runs on the plain values, in base SI units.
``UNSUPPORTED``
    Refused with a TypeError saying why.

A function only goes into a bucket once a case in test_array_functions.py shows
that it belongs there.
"""

import numpy as np

from ._dimension import DIMENSIONLESS, fail_for_dimension_mismatch, get_dimensions
from ._errors import DimensionMismatchError

SUBCLASS_SAFE = set()
UNIT_FREE = set()
UNSUPPORTED = {}  # function -> why it is refused
HANDLED = {}  # function -> implementation(wrap, *args, **kwargs)


def implements(*functions):
    """Register the decorated function as QuantSI's implementation of ``functions``.

    The implementation is called as ``implementation(function, as_quantity, *args,
    **kwargs)``: the NumPy function that was called, a callable
    ``as_quantity(values, dim)`` that attaches dimensions to a plain result, and the
    arguments of the NumPy call. (Declare the first two positional-only, ``/``, so
    that they cannot clash with NumPy's own parameter names.)
    """

    def register(implementation):
        for function in functions:
            HANDLED[function] = implementation
        return implementation

    return register


def strip_units(obj):
    """The plain values (base SI units) of every Quantity in ``obj``, recursively."""
    if hasattr(obj, "dim") and isinstance(obj, np.ndarray):
        return np.asarray(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(strip_units(item) for item in obj)
    if isinstance(obj, dict):
        return {key: strip_units(value) for key, value in obj.items()}
    return obj


def unsupported_message(function):
    return (
        f"numpy.{function.__name__} is not supported for quantities: "
        f"{UNSUPPORTED[function]}. Apply it to np.asarray(x) (the values in base SI "
        "units) if dropping the units is intended."
    )


# ------------------------------------------------------------------------------
# SUBCLASS_SAFE: NumPy's implementation is correct for quantities.
# ------------------------------------------------------------------------------
SUBCLASS_SAFE.update(
    {
        # reductions and statistics (through ufunc reductions and methods)
        np.all, np.any, np.amax, np.amin, np.max, np.min, np.sum, np.prod, np.mean,
        np.std, np.var, np.median, np.percentile, np.quantile, np.average, np.ptp,
        np.nanmax, np.nanmin, np.nansum, np.nanprod, np.nanmean, np.nanstd, np.nanvar,
        np.nanmedian, np.nanpercentile, np.nanquantile, np.trace, np.linalg.trace,
        np.cumsum, np.cumulative_sum, np.nancumsum, np.cumulative_prod, np.diff,
        np.ediff1d, np.trapezoid,
        # rounding and element-wise helpers
        np.around, np.round, np.fix, np.clip, np.nan_to_num, np.real, np.imag,
        np.real_if_close,
        # shape, order and selection (views of the same data)
        np.reshape, np.ravel, np.transpose, np.matrix_transpose,
        np.linalg.matrix_transpose, np.swapaxes, np.moveaxis, np.rollaxis, np.squeeze,
        np.expand_dims, np.atleast_1d, np.atleast_2d, np.atleast_3d, np.flip, np.fliplr,
        np.flipud, np.rot90, np.roll, np.repeat, np.tile, np.diag, np.diagflat,
        np.diagonal, np.linalg.diagonal, np.take, np.take_along_axis, np.compress,
        np.extract, np.delete, np.trim_zeros, np.sort, np.partition, np.searchsorted,
        np.put, np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit, np.unstack,
        np.astype, np.piecewise, np.apply_along_axis, np.meshgrid, np.linspace,
        np.unique, np.unique_values, np.unique_counts, np.unique_inverse,
        # new arrays like an existing quantity (same dimensions)
        np.empty_like, np.zeros_like, np.ones_like, np.full_like,
        # products done with ufuncs
        np.kron, np.linalg.matmul, np.linalg.matrix_power, np.linalg.vecdot,
    }
)  # fmt: skip

# ------------------------------------------------------------------------------
# UNIT_FREE: the result has no units.
# ------------------------------------------------------------------------------
UNIT_FREE.update(
    {
        # indices and positions
        np.argmax, np.argmin, np.argsort, np.argpartition, np.argwhere, np.flatnonzero, np.nonzero, np.lexsort, np.nanargmax, np.nanargmin,
        np.ix_, np.diag_indices_from, np.tril_indices_from, np.triu_indices_from,
        np.unravel_index, np.ravel_multi_index,
        # counts, booleans and dimensionless results
        np.count_nonzero, np.isneginf, np.isposinf, np.iscomplex, np.isreal,
        np.iscomplexobj, np.isrealobj, np.may_share_memory, np.shares_memory,
        np.corrcoef, np.angle, np.linalg.cond, np.linalg.matrix_rank,
        # array properties
        np.ndim, np.size, np.shape, np.can_cast, np.common_type, np.min_scalar_type,
        np.result_type, np.einsum_path,
        # text and files: values in base SI units
        np.array2string, np.array_repr, np.array_str, np.save, np.savetxt, np.savez,
        np.savez_compressed,
        # creating arrays with like=quantity: the new array has no dimensions
        np.arange, np.array, np.asarray, np.asanyarray, np.ascontiguousarray,
        np.asfortranarray, np.empty, np.zeros, np.ones, np.full, np.eye, np.identity,
        np.tri, np.require, np.frombuffer, np.fromfile, np.fromfunction, np.fromiter,
        np.fromstring, np.loadtxt, np.genfromtxt,
    }
)  # fmt: skip

# ------------------------------------------------------------------------------
# UNSUPPORTED: refused, with the reason.
# ------------------------------------------------------------------------------
_DATES = "it works on dates and times"
_INTEGERS = "it works on integers or bits"
_POLYNOMIALS = "polynomial coefficients would need different dimensions"
UNSUPPORTED.update(
    {
        np.busday_count: _DATES,
        np.busday_offset: _DATES,
        np.is_busday: _DATES,
        np.datetime_as_string: _DATES,
        np.packbits: _INTEGERS,
        np.unpackbits: _INTEGERS,
        np.poly: _POLYNOMIALS,
        np.polyadd: _POLYNOMIALS,
        np.polyder: _POLYNOMIALS,
        np.polydiv: _POLYNOMIALS,
        np.polyfit: _POLYNOMIALS,
        np.polyint: _POLYNOMIALS,
        np.polymul: _POLYNOMIALS,
        np.polysub: _POLYNOMIALS,
        np.polyval: _POLYNOMIALS,
        np.roots: _POLYNOMIALS,
        np.vander: "its columns would have different dimensions",
        np.apply_over_axes: "use the reduction's axis argument instead",
        np.logspace: "its start and stop are exponents and must be dimensionless",
        np.linalg.lstsq: "its results have different dimensions; use np.linalg.solve",
        np.linalg.tensorinv: "it is not implemented for quantities",
        np.linalg.tensorsolve: "it is not implemented for quantities",
        np.linalg.slogdet: "the logarithm of a determinant with dimensions is undefined",
    }
)


# ------------------------------------------------------------------------------
# Helpers for the implementations
# ------------------------------------------------------------------------------


def _dim(obj):
    """get_dimensions(obj), without the cost of an exception for plain numbers."""
    dim = getattr(obj, "dim", None)
    return get_dimensions(obj) if dim is None else dim


def _shared_dimensions(values, function):
    """The dimensions of ``values``, which must agree (a plain 0 matches anything)."""
    values = list(values)
    for value in values[1:]:
        fail_for_dimension_mismatch(
            values[0], value, f"numpy.{function.__name__} needs values with the same dimensions"
        )
    for value in values:
        dim = _dim(value)
        if dim is not DIMENSIONLESS:
            return dim
    return DIMENSIONLESS


def _leaves(nested):
    """The arrays inside the nested lists that np.block accepts."""
    if isinstance(nested, list):
        for item in nested:
            yield from _leaves(item)
    else:
        yield nested


def _with_output(as_quantity, result, dim, out):
    """Attach ``dim`` to a result; with ``out=``, relabel and return the output."""
    if out is not None and hasattr(out, "dim"):
        out.dim = dim
        return out
    return as_quantity(result, dim)


# ------------------------------------------------------------------------------
# HANDLED: joining, stacking and reshaping
# ------------------------------------------------------------------------------


@implements(np.concatenate, np.stack)
def _concatenate(function, as_quantity, /, arrays, *args, out=None, **kwargs):
    arrays = list(arrays)
    dim = _shared_dimensions(arrays, function)
    raw_out = None if out is None else np.asarray(out)
    result = function(strip_units(arrays), *args, out=raw_out, **kwargs)
    return _with_output(as_quantity, result, dim, out)


@implements(np.hstack, np.vstack, np.dstack, np.column_stack)
def _stack(function, as_quantity, /, arrays, *args, **kwargs):
    arrays = list(arrays)
    dim = _shared_dimensions(arrays, function)
    return as_quantity(function(strip_units(arrays), *args, **kwargs), dim)


@implements(np.block)
def _block(function, as_quantity, /, arrays):
    dim = _shared_dimensions(_leaves(arrays), function)
    return as_quantity(np.block(strip_units(arrays)), dim)


@implements(np.append)
def _append(function, as_quantity, /, arr, values, axis=None):
    dim = _shared_dimensions([arr, values], function)
    return as_quantity(np.append(np.asarray(arr), np.asarray(values), axis=axis), dim)


@implements(np.insert)
def _insert(function, as_quantity, /, arr, obj, values, axis=None):
    dim = _shared_dimensions([arr, values], function)
    return as_quantity(np.insert(np.asarray(arr), obj, np.asarray(values), axis=axis), dim)


@implements(np.pad)
def _pad(function, as_quantity, /, array, pad_width, mode="constant", **kwargs):
    for name in ("constant_values", "end_values"):
        if name in kwargs:
            _shared_dimensions([array, kwargs[name]], function)
    result = np.pad(np.asarray(array), pad_width, mode, **strip_units(kwargs))
    return as_quantity(result, _dim(array))


@implements(np.broadcast_arrays)
def _broadcast_arrays(function, as_quantity, /, *args, subok=False):
    results = np.broadcast_arrays(*strip_units(args))
    return tuple(as_quantity(r, _dim(a)) for r, a in zip(results, args, strict=True))


@implements(
    np.broadcast_to,
    np.copy,
    np.resize,
    np.tril,
    np.triu,
    np.sort_complex,
    np.lib.stride_tricks.sliding_window_view,
)
def _same_dimensions(function, as_quantity, /, array, *args, **kwargs):
    # NumPy's implementation is right except that it returns a plain array
    # (np.copy and np.broadcast_to default to subok=False, for example).
    kwargs.pop("subok", None)
    return as_quantity(function(np.asarray(array), *args, **kwargs), _dim(array))


@implements(np.unique_all)
def _unique_all(function, as_quantity, /, x):
    result = np.unique_all(np.asarray(x))
    values = as_quantity(result.values, _dim(x))
    return type(result)(values, result.indices, result.inverse_indices, result.counts)


# ------------------------------------------------------------------------------
# HANDLED: selecting values, writing values, set operations
# ------------------------------------------------------------------------------


@implements(np.where)
def _where(function, as_quantity, /, condition, *values):
    if not values:  # np.where(condition) is np.nonzero(condition): indices
        return np.where(np.asarray(condition))
    dim = _shared_dimensions(values, function)
    return as_quantity(np.where(np.asarray(condition), *strip_units(values)), dim)


@implements(np.select)
def _select(function, as_quantity, /, condlist, choicelist, default=0):
    choicelist = list(choicelist)
    dim = _shared_dimensions([*choicelist, default], function)
    result = np.select(strip_units(list(condlist)), strip_units(choicelist), np.asarray(default))
    return as_quantity(result, dim)


@implements(np.choose)
def _choose(function, as_quantity, /, a, choices, out=None, mode="raise"):
    choices = list(choices)
    dim = _shared_dimensions(choices, function)
    raw_out = None if out is None else np.asarray(out)
    result = np.choose(np.asarray(a), strip_units(choices), out=raw_out, mode=mode)
    return _with_output(as_quantity, result, dim, out)


@implements(np.copyto)
def _copyto(function, as_quantity, /, dst, src, casting="same_kind", where=True):
    # A source with dimensions must match the destination. Plain numbers are taken
    # as values in base SI units: NumPy's own functions (np.full_like, the
    # nan-functions, ...) fill arrays this way.
    if _dim(src) is not DIMENSIONLESS:
        _shared_dimensions([dst, src], function)
    np.copyto(np.asarray(dst), np.asarray(src), casting=casting, where=strip_units(where))


def _check_values_fit(array, values, function):
    """Values written into ``array`` need its dimensions, as for ``array[i] = values``."""
    fail_for_dimension_mismatch(
        array,
        values,
        f"numpy.{function.__name__}: the values need the dimensions of the array",
    )


@implements(np.place)
def _place(function, as_quantity, /, arr, mask, vals):
    _check_values_fit(arr, vals, function)
    np.place(np.asarray(arr), np.asarray(mask), np.asarray(vals))


@implements(np.putmask)
def _putmask(function, as_quantity, /, a, mask, values):
    _check_values_fit(a, values, function)
    np.putmask(np.asarray(a), np.asarray(mask), np.asarray(values))


@implements(np.put_along_axis)
def _put_along_axis(function, as_quantity, /, arr, indices, values, axis):
    _check_values_fit(arr, values, function)
    np.put_along_axis(np.asarray(arr), np.asarray(indices), np.asarray(values), axis)


@implements(np.fill_diagonal)
def _fill_diagonal(function, as_quantity, /, a, val, wrap=False):
    _check_values_fit(a, val, function)
    np.fill_diagonal(np.asarray(a), np.asarray(val), wrap=wrap)


@implements(np.isin)
def _isin(function, as_quantity, /, element, test_elements, *args, **kwargs):
    _shared_dimensions([element, test_elements], function)
    return np.isin(np.asarray(element), np.asarray(test_elements), *args, **kwargs)


@implements(np.digitize)
def _digitize(function, as_quantity, /, x, bins, right=False):
    _shared_dimensions([x, bins], function)
    return np.digitize(np.asarray(x), np.asarray(bins), right=right)


@implements(np.union1d, np.setxor1d, np.setdiff1d)
def _set_operation(function, as_quantity, /, ar1, ar2, *args, **kwargs):
    dim = _shared_dimensions([ar1, ar2], function)
    return as_quantity(function(np.asarray(ar1), np.asarray(ar2), *args, **kwargs), dim)


@implements(np.intersect1d)
def _intersect1d(function, as_quantity, /, ar1, ar2, assume_unique=False, return_indices=False):
    dim = _shared_dimensions([ar1, ar2], function)
    result = np.intersect1d(
        np.asarray(ar1), np.asarray(ar2), assume_unique=assume_unique, return_indices=return_indices
    )
    if return_indices:
        values, *indices = result
        return (as_quantity(values, dim), *indices)
    return as_quantity(result, dim)


# ------------------------------------------------------------------------------
# HANDLED: comparing and summarising
# ------------------------------------------------------------------------------


@implements(np.isclose, np.allclose)
def _isclose(function, as_quantity, /, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    _shared_dimensions([a, b], function)
    if _dim(atol) is not DIMENSIONLESS:
        _shared_dimensions([a, atol], function)
    # A plain atol is a value in base SI units, like every plain value QuantSI sees.
    return function(
        np.asarray(a), np.asarray(b), rtol=rtol, atol=np.asarray(atol), equal_nan=equal_nan
    )


@implements(np.array_equal, np.array_equiv)
def _array_equal(function, as_quantity, /, a1, a2, *args, **kwargs):
    try:
        _shared_dimensions([a1, a2], function)
    except DimensionMismatchError:
        return False  # quantities of different kinds are never equal
    return function(np.asarray(a1), np.asarray(a2), *args, **kwargs)


@implements(np.cov)
def _cov(function, as_quantity, /, m, y=None, *args, **kwargs):
    dim = _shared_dimensions([m] if y is None else [m, y], function)
    y = None if y is None else np.asarray(y)
    return as_quantity(np.cov(np.asarray(m), y, *args, **strip_units(kwargs)), dim**2)


@implements(np.bincount)
def _bincount(function, as_quantity, /, x, weights=None, minlength=0):
    fail_for_dimension_mismatch(x, error_message="numpy.bincount counts integers, not quantities")
    result = np.bincount(np.asarray(x), weights=strip_units(weights), minlength=minlength)
    return result if weights is None else as_quantity(result, _dim(weights))


def _check_bins(a, bins, range, function):
    if not isinstance(bins, (int, np.integer, str)):
        _shared_dimensions([a, bins], function)
    if range is not None:
        _shared_dimensions([a, *range], function)


def _histogram_values_dim(sample_dims, density, weights):
    """Counts are plain, sums of weights have the weights' dimensions, and
    densities (normalised to integrate to one) the inverse of the samples'."""
    if density:
        dim = DIMENSIONLESS
        for sample_dim in sample_dims:
            dim = dim / sample_dim
        return dim
    return DIMENSIONLESS if weights is None else _dim(weights)


@implements(np.histogram)
def _histogram(function, as_quantity, /, a, bins=10, range=None, density=None, weights=None):
    _check_bins(a, bins, range, function)
    hist, edges = np.histogram(
        np.asarray(a),
        bins=strip_units(bins),
        range=strip_units(range),
        density=density,
        weights=strip_units(weights),
    )
    dim = _dim(a)
    return as_quantity(hist, _histogram_values_dim([dim], density, weights)), as_quantity(
        edges, dim
    )


@implements(np.histogram_bin_edges)
def _histogram_bin_edges(function, as_quantity, /, a, bins=10, range=None, weights=None):
    _check_bins(a, bins, range, function)
    edges = np.histogram_bin_edges(
        np.asarray(a),
        bins=strip_units(bins),
        range=strip_units(range),
        weights=strip_units(weights),
    )
    return as_quantity(edges, _dim(a))


@implements(np.histogram2d)
def _histogram2d(function, as_quantity, /, x, y, bins=10, range=None, density=None, weights=None):
    if isinstance(bins, (list, tuple)) and len(bins) == 2:
        for axis_bins, values in zip(bins, (x, y), strict=True):
            _check_bins(values, axis_bins, None, function)
    else:
        _check_bins(x, bins, None, function)
        _check_bins(y, bins, None, function)
    if range is not None:
        for axis_range, values in zip(range, (x, y), strict=True):
            if axis_range is not None:
                _shared_dimensions([values, *axis_range], function)
    hist, x_edges, y_edges = np.histogram2d(
        np.asarray(x),
        np.asarray(y),
        bins=strip_units(bins),
        range=strip_units(range),
        density=density,
        weights=strip_units(weights),
    )
    x_dim, y_dim = _dim(x), _dim(y)
    return (
        as_quantity(hist, _histogram_values_dim([x_dim, y_dim], density, weights)),
        as_quantity(x_edges, x_dim),
        as_quantity(y_edges, y_dim),
    )


@implements(np.histogramdd)
def _histogramdd(function, as_quantity, /, sample, bins=10, range=None, density=None, weights=None):
    if isinstance(sample, (list, tuple)):
        dims = [_dim(axis_sample) for axis_sample in sample]
    else:  # an (N, D) array: one dimension for all D axes
        dims = [_dim(sample)] * (np.shape(sample)[-1] if np.ndim(sample) > 1 else 1)
    hist, edges = np.histogramdd(
        strip_units(sample),
        bins=strip_units(bins),
        range=strip_units(range),
        density=density,
        weights=strip_units(weights),
    )
    edges = tuple(as_quantity(e, d) for e, d in zip(edges, dims, strict=True))
    return as_quantity(hist, _histogram_values_dim(dims, density, weights)), edges


@implements(np.interp)
def _interp(function, as_quantity, /, x, xp, fp, left=None, right=None, period=None):
    _shared_dimensions([x, xp] + ([] if period is None else [period]), function)
    dim = _shared_dimensions([fp] + [v for v in (left, right) if v is not None], function)
    result = np.interp(
        np.asarray(x),
        np.asarray(xp),
        np.asarray(fp),
        left=strip_units(left),
        right=strip_units(right),
        period=strip_units(period),
    )
    return as_quantity(result, dim)


@implements(np.geomspace)
def _geomspace(function, as_quantity, /, start, stop, *args, **kwargs):
    dim = _shared_dimensions([start, stop], function)
    return as_quantity(np.geomspace(np.asarray(start), np.asarray(stop), *args, **kwargs), dim)


@implements(np.gradient)
def _gradient(function, as_quantity, /, f, *varargs, axis=None, edge_order=1):
    result = np.gradient(np.asarray(f), *strip_units(varargs), axis=axis, edge_order=edge_order)
    # The spacing: none (unit steps), one for all axes, or one per axis.
    n_results = len(result) if isinstance(result, (tuple, list)) else 1
    if not varargs:
        spacing_dims = [DIMENSIONLESS] * n_results
    elif len(varargs) == 1:
        spacing_dims = [_dim(varargs[0])] * n_results
    else:
        spacing_dims = [_dim(spacing) for spacing in varargs]
    f_dim = _dim(f)
    if isinstance(result, (tuple, list)):
        return type(result)(
            as_quantity(r, f_dim / d) for r, d in zip(result, spacing_dims, strict=True)
        )
    return as_quantity(result, f_dim / spacing_dims[0])


@implements(np.cumprod, np.nancumprod)
def _cumprod(function, as_quantity, /, a, *args, **kwargs):
    # Quantity.cumprod refuses quantities with dimensions, but NumPy calls methods
    # through a wrapper that catches TypeError and retries on the plain values,
    # which would silently give a wrong result. So refuse here.
    if _dim(a) is not DIMENSIONLESS:
        raise TypeError(
            f"numpy.{function.__name__} is not supported for quantities with dimensions: "
            "the partial products would have different dimensions."
        )
    return function(np.asarray(a), *args, **kwargs)
