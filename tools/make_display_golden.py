"""Rewrite tests/data/display_golden.txt from the current display code.

Only run this when you change the output on purpose, and check the diff.
"""

import pathlib

from QuantSI.tests.display_cases import render

OUTPUT = pathlib.Path(__file__).parent.parent / "src/QuantSI/tests/data/display_golden.txt"

if __name__ == "__main__":
    OUTPUT.write_text(render())
    print(f"wrote {OUTPUT}")
