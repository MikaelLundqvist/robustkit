"""
Run the full quantiles_suite: io, trend, and reconstruct validation,
covering both synthetic edge cases (always run) and real SCB data
checks (run if data files are present, skipped with a clear message
otherwise).

To run against real data, place your own SCB JSON-stat exports at:
    validation/quantiles_suite/data/quartiles.json
    validation/quantiles_suite/data/age.json

Run with:

    python -m validation.quantiles_suite.main

Or run one file's checks in isolation during development, e.g.:

    python -m validation.quantiles_suite.run_reconstruct
"""

from . import run_io, run_trend, run_reconstruct
from ..openml_suite.common import section


def main():
    section("QUANTILES SUITE -- io")
    run_io.main()

    section("QUANTILES SUITE -- trend")
    run_trend.main()

    section("QUANTILES SUITE -- reconstruct")
    run_reconstruct.main()

    section("QUANTILES SUITE COMPLETE")


if __name__ == "__main__":
    main()
