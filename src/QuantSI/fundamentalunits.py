"""
Defines physical units and quantities.

This module only re-exports: the code lives in QuantSI's private modules (see
the "How the code is organised" developer page). It keeps the old names because
old pickles and Brian2's ``brian2.units.fundamentalunits`` depend on them, so
please don't add new code here.

=====================  ========  ======
Quantity               Unit      Symbol
---------------------  --------  ------
Length                 metre     m
Mass                   kilogram  kg
Time                   second    s
Electric current       ampere    A
Temperature            kelvin    K
Quantity of substance  mole      mol
Luminosity             candle    cd
=====================  ========  ======
"""

from ._decorators import check_units as check_units
from ._dimension import (
    DIMENSIONLESS as DIMENSIONLESS,
    Dimension as Dimension,
    _di as _di,
    _dimensions as _dimensions,
    _iclass_label as _iclass_label,
    _ilabel as _ilabel,
    _siprefixes as _siprefixes,
    fail_for_dimension_mismatch as fail_for_dimension_mismatch,
    get_dimensions as get_dimensions,
    get_or_create_dimension as get_or_create_dimension,
    have_same_dimensions as have_same_dimensions,
    is_dimensionless as is_dimensionless,
    is_scalar_type as is_scalar_type,
)
from ._errors import DimensionMismatchError as DimensionMismatchError
from ._formatting import in_best_unit as in_best_unit, in_unit as in_unit
from ._quantity import (
    Quantity as Quantity,
    quantity_with_dimensions as quantity_with_dimensions,
    wrap_function_keep_dimensions as wrap_function_keep_dimensions,
)
from ._registry import (
    UnitRegistry as UnitRegistry,
    additional_unit_register as additional_unit_register,
    get_unit as get_unit,
    get_unit_for_display as get_unit_for_display,
    register_new_unit as register_new_unit,
    standard_unit_register as standard_unit_register,
    user_unit_register as user_unit_register,
)
from ._ufuncs import (
    UFUNCS_CHANGE_DIMENSIONS as UFUNCS_CHANGE_DIMENSIONS,
    UFUNCS_COMPARISONS as UFUNCS_COMPARISONS,
    UFUNCS_DIMENSIONLESS as UFUNCS_DIMENSIONLESS,
    UFUNCS_DIMENSIONLESS_TWOARGS as UFUNCS_DIMENSIONLESS_TWOARGS,
    UFUNCS_INTEGERS as UFUNCS_INTEGERS,
    UFUNCS_LOGICAL as UFUNCS_LOGICAL,
    UFUNCS_MATCHING_DIMENSIONS as UFUNCS_MATCHING_DIMENSIONS,
    UFUNCS_PRESERVE_DIMENSIONS as UFUNCS_PRESERVE_DIMENSIONS,
)
from ._unit import Unit as Unit
from ._utils import _flatten as _flatten, _short_str as _short_str

__all__ = [
    "DimensionMismatchError",
    "get_or_create_dimension",
    "get_dimensions",
    "is_dimensionless",
    "have_same_dimensions",
    "in_unit",
    "in_best_unit",
    "Quantity",
    "Unit",
    "register_new_unit",
    "check_units",
    "is_scalar_type",
    "get_unit",
]
