"""The ufunc rule table, and the old name lists we keep for compatibility."""

import numpy as np
import pytest

from QuantSI import _ufuncs
from QuantSI._ufuncs import RULES, Rule

#: Which rules each old list stands for.
LEGACY_LISTS = {
    "UFUNCS_PRESERVE_DIMENSIONS": {Rule.PRESERVE},
    "UFUNCS_CHANGE_DIMENSIONS": {
        Rule.MULTIPLY,
        Rule.DIVIDE,
        Rule.SQRT,
        Rule.SQUARE,
        Rule.RECIPROCAL,
    },
    "UFUNCS_MATCHING_DIMENSIONS": {Rule.MATCH},
    "UFUNCS_COMPARISONS": {Rule.COMPARE},
    "UFUNCS_LOGICAL": {Rule.UNITLESS_RESULT},
    "UFUNCS_DIMENSIONLESS": {Rule.DIMENSIONLESS},
    "UFUNCS_DIMENSIONLESS_TWOARGS": {Rule.DIMENSIONLESS_BOTH},
    "UFUNCS_INTEGERS": {Rule.INTEGER_ONLY},
}


def test_rules_are_keyed_by_ufunc_objects():
    assert all(isinstance(ufunc, np.ufunc) for ufunc in RULES)
    assert all(isinstance(rule, Rule) for rule in RULES.values())


@pytest.mark.parametrize("list_name", sorted(LEGACY_LISTS))
def test_historical_lists_agree_with_the_rules(list_name):
    for name in getattr(_ufuncs, list_name):
        ufunc = getattr(np, name, None)
        if isinstance(ufunc, np.ufunc):  # some old names are not ufuncs
            assert RULES[ufunc] in LEGACY_LISTS[list_name], name


def test_aliases_share_one_rule():
    assert np.true_divide is np.divide and RULES[np.true_divide] is Rule.DIVIDE
    assert np.mod is np.remainder and RULES[np.mod] is Rule.MATCH
