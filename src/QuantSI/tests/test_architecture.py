"""The private modules are layered: top-level imports may only point down.

At the top of a file, a module may only import the modules listed *before* it in
``LAYERS``. Importing a module from a higher layer is fine inside a function (on
a rare path, for example to build an error message), and keeps the imports free
of cycles. Private modules never import the public modules that re-export them.
See docs_sphinx/developer/architecture.rst.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).parent.parent

#: Bottom-up. Add new private modules here, at the lowest layer that works.
LAYERS = [
    "_utils",
    "_errors",
    "_dimension",
    "_registry",
    "_ufuncs",
    "_array_functions",
    "_quantity",
    "_unit",
    "_formatting",
    "_decorators",
]


def module_level_imports(path):
    """QuantSI modules imported by top-level statements of ``path``."""
    imported = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            # "from ._x import y" names _x; "from . import _x" names y
            imported.update([node.module] if node.module else [a.name for a in node.names])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("QuantSI"):
            imported.add(node.module.split(".")[1] if "." in node.module else node.module)
    return imported


def test_every_private_module_has_a_layer():
    private = {p.stem for p in PACKAGE.glob("_*.py")} - {"__init__", "_version"}
    assert private == set(LAYERS)


@pytest.mark.parametrize("module", LAYERS)
def test_imports_only_point_down(module):
    allowed = set(LAYERS[: LAYERS.index(module)])
    imported = module_level_imports(PACKAGE / f"{module}.py")
    assert imported <= allowed, f"{module} imports {sorted(imported - allowed)} at module level"
