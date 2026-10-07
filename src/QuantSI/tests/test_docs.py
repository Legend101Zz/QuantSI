"""The examples in the user documentation must keep working."""

import doctest
import pathlib

import pytest

USER_DOCS = pathlib.Path(__file__).parents[3] / "docs_sphinx" / "user"


@pytest.mark.skipif(
    not USER_DOCS.exists(), reason="documentation not available (installed package)"
)
@pytest.mark.parametrize("page", ["text.rst"])
def test_documentation_examples(page):
    result = doctest.testfile(
        str(USER_DOCS / page), module_relative=False, optionflags=doctest.ELLIPSIS
    )
    assert result.failed == 0
