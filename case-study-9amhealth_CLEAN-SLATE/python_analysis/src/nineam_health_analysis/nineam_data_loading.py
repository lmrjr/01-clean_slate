"""Load and canonicalize the four 9amHealth case-study source tables.

The supplied files use UTF-16 encoding and tab separators despite their
``.csv`` extensions.  This module owns that source-specific knowledge so the
rest of the analysis can operate on predictable, snake-case schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class CaseStudyData:
    """Canonical source tables required by the case-study analysis."""

    module_completions: pd.DataFrame
    demographics: pd.DataFrame
    engagement: pd.DataFrame
    body_weights: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    """Describe one source file and its canonical schema."""

    filename: str
    column_names: dict[str, str]
    date_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()


_MODULE_SPEC = _SourceSpec(
    filename=(
        "Data Analyst Case Study Doc 1 - "
        "12 Weeks Weight Loss Modules Completion.csv"
    ),
    column_names={
        "Readable Id": "member_id",
        "Questionnaire Title (All Questionnaire Records)": (
            "questionnaire_title"
        ),
        "Day of Answered At (All Questionnaire Records)": "answered_at",
    },
    date_columns=("answered_at",),
)

_DEMOGRAPHICS_SPEC = _SourceSpec(
    filename="Data Analyst Case Study Doc 2 - Demographics.csv",
    column_names={
        "Readable Id": "member_id",
        "Day of Start Date": "start_date",
        "Status (Subscriptions)": "subscription_status",
        "cancellation_date": "cancellation_date",
        "Sex": "sex",
        "ethnicity": "ethnicity",
    },
    date_columns=("start_date", "cancellation_date"),
)

_ENGAGEMENT_SPEC = _SourceSpec(
    filename="Data Analyst Case Study Doc 3 - Engagement Data.csv",
    column_names={
        "Readable Id": "member_id",
        "Day of Activity Timestamp": "activity_at",
        "Type": "event_type",
    },
    date_columns=("activity_at",),
)

_BODY_WEIGHT_SPEC = _SourceSpec(
    filename="Data Analyst Case Study Doc 4 - BW_Detail.csv",
    column_names={
        "User Id": "member_id",
        "Weight-loss Member Type": "member_type",
        "Day of BW First Measurement Effective": "first_weight_at",
        "Day of BW Last Measurement Effective": "last_weight_at",
        "First": "first_weight",
        "Last": "last_weight",
        "Diff": "weight_difference",
        "BW Days between First/Last": "weight_days",
        "days in weight-loss program": "tenure_days",
    },
    date_columns=("first_weight_at", "last_weight_at"),
    numeric_columns=(
        "first_weight",
        "last_weight",
        "weight_difference",
        "weight_days",
        "tenure_days",
    ),
)


def _read_source_table(data_dir: Path, spec: _SourceSpec) -> pd.DataFrame:
    """Read one source table and return only its canonical columns.

    Args:
        data_dir: Directory containing the supplied case-study files.
        spec: File-specific source and canonical column definitions.

    Returns:
        A dataframe with ordered, snake-case columns and parsed values.

    Side effects:
        Reads one source file; does not write files or mutate arguments.

    Statistical intent:
        Prevents encoding, schema, or type drift from changing model inputs.

    Raises:
        FileNotFoundError: If the expected source file is absent.
        ValueError: If a required column is absent or a value cannot be parsed.
    """
    # Fail at the data boundary so a missing extract cannot masquerade as an
    # empty analytical population later in the pipeline.
    source_path = data_dir / spec.filename
    if not source_path.is_file():
        raise FileNotFoundError(f"Required source file not found: {source_path}")

    # Read the source's actual TSV format and remove only unnamed trailing
    # fields; named all-null fields such as cancellation_date remain meaningful.
    table = pd.read_csv(source_path, encoding="utf-16", sep="\t")
    while (
        len(table.columns) > 0
        and str(table.columns[-1]).startswith("Unnamed:")
        and table.iloc[:, -1].isna().all()
    ):
        table = table.iloc[:, :-1]
    table.columns = [str(column).strip() for column in table.columns]

    # Reject schema drift before renaming so downstream models cannot silently
    # bind a semantically different field.
    missing_columns = [
        column for column in spec.column_names if column not in table.columns
    ]
    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"{spec.filename} is missing required columns: {missing_display}"
        )

    # Limit each table to its analytical contract and expose consistent names
    # to every subsequent cohort and feature operation.
    table = table.loc[:, list(spec.column_names)].rename(
        columns=spec.column_names
    )

    # Parsed dates establish a common time scale for last-weight censoring and
    # longitudinal ordering rather than leaving locale-dependent strings.
    for column in spec.date_columns:
        table[column] = pd.to_datetime(
            table[column],
            format="mixed",
            errors="raise",
        )

    # Numeric conversion fails loudly on unexpected source values because
    # silent coercion would change exclusion counts and model inputs.
    for column in spec.numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="raise")

    # Trim identifiers and categorical labels so formatting whitespace cannot
    # create false members or factor levels.
    text_columns = table.columns.difference(
        (*spec.date_columns, *spec.numeric_columns),
        sort=False,
    )
    for column in text_columns:
        table[column] = table[column].astype("string").str.strip()

    return table.reset_index(drop=True)


def load_case_study_data(data_dir: str | Path) -> CaseStudyData:
    """Load all supplied case-study tables from ``data_dir``.

    Args:
        data_dir: Directory containing the four original case-study files.

    Returns:
        Canonically named, typed source tables grouped in ``CaseStudyData``.

    Side effects:
        Reads four source files; does not write files or mutate arguments.

    Statistical intent:
        Supplies consistent variable types and meanings to all model paths.

    Raises:
        FileNotFoundError: If ``data_dir`` or an expected file does not exist.
        ValueError: If a source schema or value is invalid.
    """
    # Validate the shared source directory once before loading the four tables
    # as a single, internally consistent analysis input.
    source_directory = Path(data_dir)
    print(data_dir)
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"Case-study data directory not found: {source_directory}"
        )

    # Keep source roles explicit rather than relying on file discovery order,
    # which could pair the wrong extract with a statistical meaning.
    return CaseStudyData(
        module_completions=_read_source_table(source_directory, _MODULE_SPEC),
        demographics=_read_source_table(source_directory, _DEMOGRAPHICS_SPEC),
        engagement=_read_source_table(source_directory, _ENGAGEMENT_SPEC),
        body_weights=_read_source_table(source_directory, _BODY_WEIGHT_SPEC),
    )