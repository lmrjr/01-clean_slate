"""Run the scientific 9amHealth analysis and publish aggregate artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"

# Permit direct repository execution before an editable package installation;
# installed environments already resolve the same package without duplication.
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from nineam_health_analysis.nineam_analysis_pipeline import (  # noqa: E402
    AnalysisConfig,
    run_analysis,
    write_analysis_outputs,
)


def _nonnegative_integer(value: str) -> int:
    """Parse one command-line seed as a nonnegative integer.

    Args:
        value: Raw argument text supplied by ``argparse``.

    Returns:
        A validated nonnegative integer.

    Side effects:
        None.

    Statistical intent:
        Requires an explicit valid seed domain for deterministic resampling.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_integer(value: str) -> int:
    """Parse one command-line resampling count as a positive integer.

    Args:
        value: Raw argument text supplied by ``argparse``.

    Returns:
        A validated positive integer.

    Side effects:
        None.

    Statistical intent:
        Prevents empty repeat and stability-selection configurations.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface and repository defaults.

    Args:
        None.

    Returns:
        An argument parser for source, output, and resampling controls.

    Side effects:
        None.

    Statistical intent:
        Exposes every runtime sampling count and seed used by orchestration,
        avoiding hidden state while leaving model specifications fixed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible aggregate-only 9amHealth scientific analysis."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT.parent / "data",
        help=(
            "Directory containing the four supplied source extracts "
            "(default: <project>/data)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs",
        help=(
            "Directory receiving tables/, figures/, and deterministic metadata "
            "(default: <project>/python analysis/outputs)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=20260901,
        help="Nonnegative random seed (default: 20260901).",
    )
    parser.add_argument(
        "--cv-folds",
        type=_positive_integer,
        default=5,
        help="Grouped cross-validation folds (default: 5; minimum: 2).",
    )
    parser.add_argument(
        "--cv-repeats",
        type=_positive_integer,
        default=2,
        help="Grouped cross-validation repeats (default: 2).",
    )
    parser.add_argument(
        "--stability-resamples",
        type=_positive_integer,
        default=100,
        help="Member-level stability subsamples (default: 100).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run analysis from parsed arguments and write aggregate artifacts.

    Args:
        argv: Optional argument sequence; omission reads process arguments.

    Returns:
        Zero after all requested artifacts are written successfully.

    Side effects:
        Reads source extracts, creates the output directory, writes fifteen
        CSV tables, eighteen image files, one metadata file, and prints a
        concise completion message.

    Statistical intent:
        Maps visible CLI controls directly to deterministic grouped validation
        and exploratory stability-selection configuration.
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    config = AnalysisConfig(
        seed=arguments.seed,
        cv_folds=arguments.cv_folds,
        cv_repeats=arguments.cv_repeats,
        stability_resamples=arguments.stability_resamples,
    )
    result = run_analysis(arguments.data_dir, config=config)
    written = write_analysis_outputs(result, arguments.output_dir)
    print(
        f"Wrote {len(written)} scientific analysis files "
        "(15 tables, 18 figures, 1 metadata) to "
        f"{arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
