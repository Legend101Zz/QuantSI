import contextlib

import pytest


def pytest_collection_modifyitems(config, items):
    # List of function names whose doctests should be skipped
    functions_to_skip = {
        "Quantity.fill",
        "Quantity.trace",
    }

    for item in items:
        function_name = item.location[2]  # The third element often contains the name in doctests
        # Skip specific functions' doctests
        if any(fn in function_name for fn in functions_to_skip):
            item.add_marker(
                pytest.mark.skip(
                    reason="Skipping doctest for specific function due to known documentation issues"
                )
            )


@contextlib.contextmanager
def preserved_unit_registries():
    """Undo, on exit, every unit registered inside the ``with`` block.

    Units are registered globally, either explicitly with register_new_unit or
    implicitly whenever units are combined (``siemens / cm**2``), and that changes
    how quantities are displayed from then on.
    """
    from QuantSI.fundamentalunits import (
        additional_unit_register,
        standard_unit_register,
        user_unit_register,
    )

    registries = (standard_unit_register, user_unit_register, additional_unit_register)
    saved = [
        (dict(r.units), {dim: dict(units) for dim, units in r.units_for_dimensions.items()})
        for r in registries
    ]
    try:
        yield
    finally:
        for registry, (units, by_dimension) in zip(registries, saved, strict=True):
            registry.units.clear()
            registry.units.update(units)
            registry.units_for_dimensions.clear()
            registry.units_for_dimensions.update(by_dimension)
            registry._tables.clear()


@pytest.fixture(autouse=True)
def _isolated_unit_registries():
    """No test (or doctest) can change how quantities display in another one."""
    with preserved_unit_registries():
        yield
