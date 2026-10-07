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

SUBCLASS_SAFE = set()
UNIT_FREE = set()
UNSUPPORTED = {}  # function -> why it is refused
HANDLED = {}  # function -> implementation(wrap, *args, **kwargs)


def implements(*functions):
    """Register the decorated function as QuantSI's implementation of ``functions``.

    The implementation receives ``wrap(values, dim)``, which attaches dimensions
    to a plain result, followed by the arguments of the NumPy call.
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
