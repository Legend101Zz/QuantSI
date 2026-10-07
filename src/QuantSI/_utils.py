"""Small helpers shared by QuantSI's internal modules."""

import numpy as np


def set_module(module):
    """Decorator that makes a function report ``module`` as its ``__module__``.

    pickle saves a function as its ``__module__`` and ``__qualname__``, and imports
    it from there again when loading. Two functions that QuantSI's pickles refer to
    used to live in ``QuantSI.fundamentalunits``. With this decorator they keep that
    path after moving to a private module, so older releases and Brian2 can still
    read new pickles. NumPy does the same (``numpy._utils.set_module``).
    """

    def decorator(func):
        func.__module__ = module
        return func

    return decorator


def _flatten(iterable):
    """
    Flatten a given list `iterable`.
    """
    for e in iterable:
        if isinstance(e, list):
            yield from _flatten(e)
        else:
            yield e


def _short_str(arr):
    """
    Return a short string representation of an array, suitable for use in
    error messages.
    """
    arr = np.asanyarray(arr)
    old_printoptions = np.get_printoptions()
    np.set_printoptions(edgeitems=2, threshold=5)
    arr_string = str(arr)
    np.set_printoptions(**old_printoptions)
    return arr_string
