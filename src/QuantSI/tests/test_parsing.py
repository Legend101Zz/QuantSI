"""parse_quantity: a small, safe language for quantities in text."""

import pytest

from QuantSI._errors import QuantityParseError
from QuantSI._parsing import MAX_LENGTH, parse_quantity
from QuantSI.allunits import amp, metre, msecond, mvolt, second, siemens, volt
from QuantSI.stdunits import cm, mV


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.5 * mV", 3.5 * mV),
        ("3.5*mvolt", 3.5 * mvolt),
        ("  -70 * mV ", -70 * mV),
        ("+3 * mV", 3 * mV),
        ("1e-9 * amp", 1e-9 * amp),
        ("0.5 * siemens / cm ** 2", 0.5 * siemens / cm**2),
        ("metre ** -2", metre**-2),
        ("(volt / second) * 2", (volt / second) * 2),
        ("2 * (3 * msecond)", 6 * msecond),
        ("metre ** (1 / 2)", metre**0.5),
    ],
)
def test_parse(text, expected):
    result = parse_quantity(text)
    assert result == expected
    assert result.dim is expected.dim  # the interned dimension


def test_dimensionless_results_are_plain_numbers():
    assert parse_quantity("3.5") == 3.5
    result = parse_quantity("3 * volt / volt")
    assert result == 3.0 and isinstance(result, float)
    # (volt / volt alone is a Unit: units combine into units, as in Python code)
    assert parse_quantity("volt / volt") == volt / volt
    assert parse_quantity("2 ** 10") == 1024.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("3.5 mV", "Not a valid expression"),  # no implicit multiplication
        ("", "Not a valid expression"),
        ("1; 2", "Not a valid expression"),
        ("sin(volt)", "Call is not allowed"),
        ("volt.dim", "Attribute is not allowed"),
        ("mvolt[0]", "Subscript is not allowed"),
        ("'3 mV'", "Constant is not allowed"),
        ("True * volt", "Constant is not allowed"),
        ("lambda: 1", "Lambda is not allowed"),
        ("mv * 3", "'mv' is not a known unit"),
        ("metre ** metre", "exponent must be a plain number"),
        ("metre ** 65", "larger than 64"),
        ("((10 ** 64) ** 64) ** 64", "too large"),
    ],
)
def test_refused(text, message):
    with pytest.raises(QuantityParseError, match=message):
        parse_quantity(text)


def test_input_limits():
    with pytest.raises(QuantityParseError, match="longer than"):
        parse_quantity("1" * (MAX_LENGTH + 1))
    with pytest.raises(QuantityParseError, match="more than"):
        parse_quantity("-" * 199 + "1")  # short, but 400 nodes


def test_parse_errors_are_value_errors():
    with pytest.raises(ValueError):
        parse_quantity("3 mV")


def test_custom_namespace():
    assert parse_quantity("2 * stride", namespace={"stride": metre}) == 2 * metre
    with pytest.raises(QuantityParseError, match="not a known unit"):
        parse_quantity("mV", namespace={})
