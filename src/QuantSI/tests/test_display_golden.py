"""Check that display output still matches data/display_golden.txt (see display_cases.py)."""

import pathlib
import subprocess
import sys

GOLDEN = pathlib.Path(__file__).parent / "data" / "display_golden.txt"


def test_display_output_matches_golden_file():
    # Use a fresh Python process, so units registered by other tests can't change the output.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from QuantSI.tests.display_cases import render; print(render(), end='')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = GOLDEN.read_text().splitlines()
    actual = result.stdout.splitlines()
    changed = [(e, a) for e, a in zip(expected, actual, strict=False) if e != a]
    assert not changed, "display output changed:\n" + "\n".join(
        f"- {e}\n+ {a}" for e, a in changed[:20]
    )
    assert len(actual) == len(expected)
