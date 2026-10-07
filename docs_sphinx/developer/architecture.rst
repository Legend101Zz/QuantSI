How the code is organised
=========================

QuantSI is a NumPy array subclass (``Quantity``) that carries a physical
dimension, plus the code that keeps that dimension right through calculations.
The code is split into small private modules with one job each, and a few public
modules re-export what users need, under names that don't change.

Public modules
--------------

Users import from these, and names never disappear from them
(``tests/test_public_api.py`` checks them against ``tests/data/public_api.json``):

``QuantSI``
    The curated top-level namespace: the classes, the functions and the
    commonly used units.
``QuantSI.fundamentalunits``
    The historical home of the core classes. It now only re-exports them, but it
    stays forever: pickles store the paths ``QuantSI.fundamentalunits.quantity_with_dimensions``
    and ``QuantSI.fundamentalunits.get_or_create_dimension``, and Brian2's
    ``brian2.units.fundamentalunits`` is a facade over it.
``QuantSI.allunits``, ``QuantSI.stdunits``, ``QuantSI.constants``
    The unit and constant catalogue.

Private modules
---------------

They are listed from the bottom up: at the top of a file, a module may only
import the modules listed *above* it (``tests/test_architecture.py`` checks
this). When a module needs one listed below it, for example to build an error
message, it imports it inside the function that needs it, so the import only
happens on that rare path.

``_utils``
    Small helpers (``set_module``, list flattening, short array strings).
``_errors``
    The exception classes.
``_dimension``
    ``Dimension``, the intern cache and ``DIMENSIONLESS``, plus the functions
    that get and compare the dimensions of arbitrary objects.
``_registry``
    The three unit registries used to pick a display unit.
``_ufuncs``
    The rule for every NumPy ufunc (``RULES``) and one handler per rule.
``_array_functions``
    How every other NumPy function treats quantities (``__array_function__``):
    the four buckets and QuantSI's implementations.
``_quantity``
    The ``Quantity`` class.
``_unit``
    The ``Unit`` class.
``_formatting``
    Turning quantities into text: ``format_quantity`` (``str``, ``repr``,
    ``format``, ``in_unit``) and ``format_quantity_latex``.
``_parsing``
    Reading quantities from text (``parse_quantity``) without ``eval``.
``_decorators``
    ``check_units``.
``_wrappers``
    The ``wrap_function_*`` helpers that ``unitsafefunctions`` is built from.

``QuantSI.unitsafefunctions`` (public, from Brian2) holds unit-aware versions of
``sin``, ``exp``, ``arange``, ... with the metadata Brian2's code generation
reads. ``tools/`` holds the code generator for the unit modules
(``generate_units.py``), the scripts that write the test fixtures, and the
Brian2 integration used in CI (``tools/brian2``).

Rules that keep the package correct
-----------------------------------

* **One shared object.** There is one ``Dimension`` object per set of exponents,
  and one copy of each registry, for the whole process. Don't copy them, and don't
  call ``Dimension(...)`` directly: use ``get_or_create_dimension``. A second copy
  would break every ``dim1 is dim2`` check.
* **Re-export explicitly.** Compatibility namespaces use ``from ._x import name as name``
  so that linters and type checkers know the import is on purpose.
* **Pickle paths are public API.** Functions that pickles refer to keep their
  old ``__module__`` through ``_utils.set_module``. ``tests/test_pickle_compat.py``
  checks this with pickles written before the code moved.
* **Metadata names are an interface.** ``check_units`` sets ``_arg_units``,
  ``_return_unit`` and a few more attributes on the functions it wraps, and
  Brian2's code generation reads them.
* **Types.** QuantSI ships ``py.typed``. Its functions and classes are annotated
  and checked with mypy in CI, together with ``tests/typing_examples.py``, which
  states the types users see. The results of arithmetic are typed as NumPy
  arrays: NumPy's type stubs type every operator that way, and an
  ``np.ndarray`` subclass cannot change that without overriding each operator.
