"""Build aggregate scientific tables for the 9amHealth leadership analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .nineam_cohort_selection import (
    PRIMARY_MEMBER_TYPES,
    REFERENCE_MEMBER_TYPE,
    CohortResult,
)
from .nineam_data_loading import CaseStudyData
from .nineam_feature_engineering import FeatureResult
from .nineam_final_model import (
    COEFFICIENT_COLUMNS,
    DIAGNOSTIC_SUMMARY_COLUMNS,
    FIT_STATISTICS_COLUMNS,
    LockedModelResult,
)

REPORTING_SCHEMAS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "nineam_cohort_flow.csv": (
            "flow_order", "row_type", "stage_id", "stage_label", "starting_n",
            "excluded_n", "retained_n", "percent_of_enrolled",
            "exclusion_definition", "notes",
        ),
        "nineam_sample_characteristics.csv": (
            "scope_order", "scope", "characteristic_order", "characteristic",
            "characteristic_label", "level_order", "level", "summary_type",
            "unit", "n_total", "n_observed", "n_missing", "count",
            "percentage", "mean", "standard_deviation", "median", "q1", "q3",
            "minimum", "maximum", "notes",
        ),
        "nineam_outcomes_by_member_type.csv": (
            "scope_order", "scope", "outcome_order", "outcome", "outcome_label",
            "outcome_type", "unit", "n_total", "n_observed", "n_missing",
            "mean", "standard_deviation", "median", "q1", "q3", "minimum",
            "maximum", "numerator", "denominator", "percentage", "ci_lower",
            "ci_upper", "ci_level", "ci_method",
        ),
        "nineam_engagement_by_member_type.csv": (
            "scope_order", "scope", "metric_order", "metric", "metric_label",
            "unit", "n_total", "n_observed", "n_missing", "zero_n", "reached_n",
            "mean", "standard_deviation", "median", "q1", "q3", "minimum",
            "maximum", "exposure_window_definition", "denominator_definition",
        ),
        "nineam_engagement_activity_summary.csv": (
            "event_order", "event_type", "actionability_class", "is_repeatable",
            "n_analysis", "n_reached", "n_zero", "total_events",
            "events_per_reached_member", "median_events_among_reached",
            "q1_events_among_reached", "q3_events_among_reached", "predictor",
            "predictor_transform", "effect_unit", "estimate", "standard_error",
            "test_statistic", "reference_distribution", "degrees_of_freedom",
            "p_value", "ci_lower", "ci_upper", "ci_level", "fdr_method",
            "fdr_family", "fdr_test_count", "fdr_adjusted_p_value",
            "covariance_estimator", "adjustment_terms", "reference_member_type",
            "test_status", "interpretation",
        ),
        "nineam_modules_by_member_type.csv": (
            "scope_order", "scope", "module_order", "module_variable",
            "module_label", "module_group", "available_title_denominator",
            "availability_status", "support_scope", "n_members",
            "n_members_with_completion", "n_members_without_completion",
            "member_completion_percentage", "total_unique_completions",
            "mean_completion_count", "standard_deviation_count",
            "median_completion_count", "q1_completion_count", "q3_completion_count",
            "minimum_completion_count", "maximum_completion_count",
            "mean_completion_proportion", "standard_deviation_proportion",
            "zero_completion_group_flag", "completion_definition",
            "cross_group_comparability", "interpretation",
        ),
        "nineam_base_model_comparison_summary.csv": (
            "model_order", "model_id", "model_family", "modeled_outcome",
            "prediction_target", "metric_order", "metric", "metric_unit",
            "n_members", "cv_repeats", "cv_folds", "n_fold_scores",
            "n_test_predictions", "mean_score", "standard_deviation",
            "minimum_score", "maximum_score", "fold_plan_id", "score_aggregation",
            "rank", "is_winner", "winner_rule", "tie_tolerance",
            "validation_scope",
        ),
        "nineam_lasso_mean_selection.csv": (
            "specification_order", "module_spec", "is_winning_specification",
            "candidate_order", "candidate", "candidate_label", "candidate_role",
            "support_scope", "reference_level", "penalty_status",
            "selected_lambda_ratio", "full_sample_lambda", "full_sample_lambda_max",
            "cv_selection_rule", "cv_mean_mse", "cv_standard_error",
            "fold_plan_id", "n_resamples", "subsample_fraction",
            "selection_threshold", "full_sample_coefficient",
            "full_sample_standardized_coefficient", "selection_count",
            "selection_frequency", "selected_at_threshold",
            "eligible_for_locked_model", "locked_model_status", "exclusion_reason",
            "coefficient_unit", "interpretation",
        ),
        "nineam_lasso_domain_selection.csv": (
            "specification_order", "module_spec", "is_winning_specification",
            "candidate_order", "candidate", "candidate_label", "candidate_role",
            "support_scope", "reference_level", "penalty_status",
            "selected_lambda_ratio", "full_sample_lambda", "full_sample_lambda_max",
            "cv_selection_rule", "cv_mean_mse", "cv_standard_error",
            "fold_plan_id", "n_resamples", "subsample_fraction",
            "selection_threshold", "full_sample_coefficient",
            "full_sample_standardized_coefficient", "selection_count",
            "selection_frequency", "selected_at_threshold",
            "eligible_for_locked_model", "locked_model_status", "exclusion_reason",
            "coefficient_unit", "interpretation",
        ),
        "nineam_locked_model_coefficients_hc3.csv": tuple(COEFFICIENT_COLUMNS),
        "nineam_locked_model_fit_statistics.csv": tuple(FIT_STATISTICS_COLUMNS),
        "nineam_model_diagnostics.csv": tuple(DIAGNOSTIC_SUMMARY_COLUMNS),
        "nineam_hypothesis_evidence.csv": (
            "hypothesis_order", "hypothesis_id", "question",
            "prespecified_expectation", "estimand", "population",
            "exposure_or_predictor", "outcome", "adjustment_set", "evidence_table",
            "evidence_filter", "result_status", "effect_summary",
            "uncertainty_summary", "multiplicity_control", "inference_status",
            "causal_status", "noncausal_interpretation", "leadership_action",
            "randomized_pilot_population", "randomized_pilot_comparator",
            "randomized_pilot_kpi", "randomized_pilot_time_horizon",
        ),
        "nineam_findings_and_implications.csv": (
            "finding_order", "finding_id", "finding", "quantitative_evidence",
            "evidence_table", "evidence_filter", "interpretation", "certainty",
            "causal_status", "leadership_implication", "recommended_test",
            "pilot_population", "pilot_intervention", "pilot_comparator",
            "pilot_primary_kpi", "pilot_time_horizon",
        ),
        "nineam_limitations.csv": (
            "limitation_order", "limitation_id", "category", "limitation",
            "empirical_evidence", "affected_estimand", "potential_impact",
            "direction_of_bias", "affected_outputs", "mitigation_in_current_analysis",
            "recommended_future_design", "severity",
        ),
    }
)

_SCOPE_LABELS = ("Overall", *PRIMARY_MEMBER_TYPES)
_MODEL_ORDER = (
    "log_compound_symmetry_gls",
    "percentage_loss_ols",
)
_LOCKED_MODEL_ORDER = (
    "locked_percentage_loss_primary",
    "locked_percentage_loss_duration_sensitivity",
)
_ACTIVITY_MINIMUM_REACH = 30
_TIE_TOLERANCE = 0.0
_CANONICAL_CANDIDATE_ORDER = {
    "engagement_volume_repeatable": 1,
    "engagement_volume_repeatable_rate": 2,
    "engagement_breadth": 3,
    "tenure_days": 4,
    "module_mean": 5,
    "module_core": 6,
    "module_mindset": 7,
    "module_nutrition": 8,
    "module_physical_activity": 9,
    "sex[MALE]": 10,
}
_SPECIFICATION_CANDIDATES = {
    "mean": frozenset(
        {
            "engagement_volume_repeatable",
            "engagement_volume_repeatable_rate",
            "engagement_breadth",
            "tenure_days",
            "module_mean",
            "sex[MALE]",
        }
    ),
    "domains": frozenset(
        {
            "engagement_volume_repeatable",
            "engagement_volume_repeatable_rate",
            "engagement_breadth",
            "tenure_days",
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
            "sex[MALE]",
        }
    ),
}

_ACTIONABILITY_BY_EVENT = {
    "CHART_REVIEW": "care_delivery",
    "COMPLETED_CONSULTATION": "care_delivery",
    "COMPLETED_LAB_TEST": "clinical_or_administrative",
    "CONSUMED_DIGITAL_CONTENT": "member_behavior",
    "MEAL_PLAN_GENERATED": "care_delivery",
    "MEDICAL_QUESTIONNAIRE_ANSWERED": "clinical_or_administrative",
    "MEDICATION_CHANGE": "care_delivery",
    "QUESTIONNAIRE_ANSWERED": "member_behavior",
    "RECORD_BLOOD_GLUCOSE": "member_behavior",
    "RECORD_BLOOD_GLUCOSE_WITH_REVIEW": "care_delivery",
    "RECORD_BLOOD_PRESSURE": "member_behavior",
    "RECORD_BLOOD_PRESSURE_WITH_REVIEW": "care_delivery",
    "RECORD_BODY_WEIGHT": "member_behavior",
    "RECORD_STEPS": "member_behavior",
    "REGISTRATION": "clinical_or_administrative",
    "SUBSCRIPTION_STARTED": "clinical_or_administrative",
    "TEXT_MESSAGE_CARE_ONLY": "care_delivery",
    "VIDEO_CALL_COMPLETED": "care_delivery",
    "VOICE_MESSAGE_CARE_ONLY": "care_delivery",
}
ACTIONABILITY_CLASSES = frozenset(_ACTIONABILITY_BY_EVENT.values())

_MODULES = (
    ("module_mean", "Overall four-domain mean", "overall", None),
    ("module_core", "Core curriculum", "core", 9),
    ("module_mindset", "Mindset extension", "extension", 4),
    ("module_nutrition", "Nutrition extension", "extension", 4),
    (
        "module_physical_activity",
        "Physical-activity extension",
        "extension",
        4,
    ),
)

LIMITATION_IDS = (
    "attrition_complete_pair",
    "observational_residual_confounding",
    "two_condition_restriction",
    "same_window_reverse_causation",
    "variable_follow_up",
    "age_absent",
    "post_selection_inference",
    "activity_multiple_testing",
    "extension_module_availability",
    "external_validation_absent",
)

_LASSO_INPUT_COLUMNS = (
    "module_spec",
    "candidate_order",
    "candidate",
    "lambda_ratio",
    "full_sample_lambda",
    "full_sample_lambda_max",
    "cv_selection_rule",
    "cv_mean_mse",
    "cv_standard_error",
    "fold_plan_id",
    "n_resamples",
    "subsample_fraction",
    "selection_threshold",
    "full_sample_coefficient",
    "full_sample_standardized_coefficient",
    "selection_count",
    "selection_frequency",
    "selected_at_threshold",
    "excluded_from_full_fit",
)


class _DefensiveTableMapping(Mapping[str, pd.DataFrame]):
    """Store private deep copies and return a fresh copy on every access."""

    __slots__ = ("__tables",)

    def __init__(self, tables: Mapping[str, pd.DataFrame]) -> None:
        """Copy every dataframe before retaining it."""
        self.__tables = {
            filename: table.copy(deep=True) for filename, table in tables.items()
        }

    def __getitem__(self, filename: str) -> pd.DataFrame:
        """Return an isolated copy of the requested publication table."""
        return self.__tables[filename].copy(deep=True)

    def __iter__(self):
        """Iterate filenames in publication order."""
        return iter(self.__tables)

    def __len__(self) -> int:
        """Return the number of publication tables."""
        return len(self.__tables)


@dataclass(frozen=True, slots=True)
class ReportingResult:
    """Hold filename-keyed aggregate tables in their publication order.

    Args:
        tables: Closed-schema aggregate dataframes keyed by approved filenames.

    Returns:
        An immutable result wrapper whose dataframe values are copied on access.

    Side effects:
        None.

    Statistical intent:
        Keeps the scientific artifact boundary aggregate-only and explicit.
    """

    tables: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        """Freeze the filename mapping after validating its publication order."""
        if tuple(self.tables) != tuple(REPORTING_SCHEMAS):
            raise ValueError("Reporting tables must follow the approved file order")
        object.__setattr__(self, "tables", _DefensiveTableMapping(self.tables))


def _require_columns(
    table: pd.DataFrame,
    required: tuple[str, ...] | set[str],
    label: str,
) -> None:
    """Validate dataframe columns without mutating analytical inputs.

    Args:
        table: Input dataframe to inspect.
        required: Columns needed by the calling builder.
        label: Input label used in an actionable error.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Prevents missing provenance from silently changing an estimand.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def wilson_confidence_interval(
    numerator: int,
    denominator: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a two-sided Wilson binomial interval on the 0--1 scale.

    Args:
        numerator: Count satisfying the binary outcome.
        denominator: Observed binary-outcome count.
        confidence: Two-sided confidence level strictly between zero and one.

    Returns:
        Lower and upper Wilson bounds as proportions.

    Side effects:
        None.

    Statistical intent:
        Gives stable finite-sample uncertainty for the 5% response rate.
    """
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise TypeError("Wilson counts must be integers")
    if int(numerator) != numerator or int(denominator) != denominator:
        raise TypeError("Wilson counts must be integers")
    successes = int(numerator)
    total = int(denominator)
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= numerator <= denominator")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator_term = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator_term
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator_term
    )
    return float(center - half_width), float(center + half_width)


def _continuous_statistics(values: pd.Series) -> dict[str, float | int]:
    """Summarize one continuous vector with approved sample conventions.

    Args:
        values: Numeric or numeric-convertible observations, including missing.

    Returns:
        Counts, sample SD, linear quartiles, and extrema.

    Side effects:
        None.

    Statistical intent:
        Standardizes every continuous descriptive table on the same estimands.
    """
    numeric = pd.to_numeric(values, errors="raise")
    observed = numeric.dropna().astype(float)
    result: dict[str, float | int] = {
        "n_total": int(len(numeric)),
        "n_observed": int(len(observed)),
        "n_missing": int(numeric.isna().sum()),
    }
    if observed.empty:
        result.update(
            {
                name: np.nan
                for name in (
                    "mean",
                    "standard_deviation",
                    "median",
                    "q1",
                    "q3",
                    "minimum",
                    "maximum",
                )
            }
        )
        return result
    result.update(
        {
            "mean": float(observed.mean()),
            "standard_deviation": (
                float(observed.std(ddof=1)) if len(observed) > 1 else np.nan
            ),
            "median": float(observed.median()),
            "q1": float(observed.quantile(0.25, interpolation="linear")),
            "q3": float(observed.quantile(0.75, interpolation="linear")),
            "minimum": float(observed.min()),
            "maximum": float(observed.max()),
        }
    )
    return result


def _scopes(members: pd.DataFrame) -> tuple[tuple[int, str, pd.DataFrame], ...]:
    """Return fixed overall and member-type analysis scopes.

    Args:
        members: Validated primary member-level dataframe.

    Returns:
        Ordered scope number, label, and stable member subset tuples.

    Side effects:
        None.

    Statistical intent:
        Makes every main descriptive table use identical denominators.
    """
    ordered = members.sort_values("member_id", kind="stable").reset_index(drop=True)
    records = [(1, "Overall", ordered)]
    records.extend(
        (
            order,
            member_type,
            ordered.loc[ordered["member_type"].eq(member_type)].reset_index(drop=True),
        )
        for order, member_type in enumerate(PRIMARY_MEMBER_TYPES, start=2)
    )
    return tuple(records)


def _build_cohort_flow(cohort: CohortResult) -> pd.DataFrame:
    """Convert sequential cohort audit counts to the closed flow schema.

    Args:
        cohort: Primary cohort with mutually exclusive audit counts.

    Returns:
        Ordered enrollment-through-comparability flow rows.

    Side effects:
        None.

    Statistical intent:
        Reconciles every exclusion to the demographic enrollment denominator.
    """
    stages = (
        ("enrolled", "Enrolled members", None, "All demographics rows"),
        (
            "body_weight_record",
            "Body-weight record available",
            "excluded_missing_body_weight_row",
            "No matching body-weight detail row",
        ),
        (
            "eligible_subscription",
            "Eligible subscription status",
            "excluded_ineligible_status",
            "Subscription status is not ACTIVE or FINISHED",
        ),
        (
            "paired_weights_observed",
            "First and last weights observed",
            "excluded_missing_weights",
            "First or last weight is missing",
        ),
        (
            "paired_weights_positive",
            "First and last weights positive",
            "excluded_nonpositive_weights",
            "First or last weight is nonpositive",
        ),
        (
            "weight_interval_observed",
            "Measurement interval observed",
            "excluded_missing_weight_days",
            "Weight-days interval is missing",
        ),
        (
            "weight_interval_positive",
            "Measurement interval positive",
            "excluded_nonpositive_weight_days",
            "Weight-days interval is zero or negative",
        ),
        (
            "primary_member_types",
            "Confirmed two-condition population",
            "excluded_nonprimary_member_type",
            (
                "Member type is Null, Active GLP-1 for Diabetes, or Active "
                "Generic Medication for Weight-loss (NOT on GLP-1 for weight-loss)"
            ),
        ),
    )
    required = {"source_demographic_rows", "included_members"}
    required.update(key for _, _, key, _ in stages if key is not None)
    missing = sorted(required.difference(cohort.audit_counts))
    if missing:
        raise ValueError("Cohort audit is missing counts: " + ", ".join(missing))
    enrolled = int(cohort.audit_counts["source_demographic_rows"])
    retained = enrolled
    rows = []
    for order, (stage_id, label, exclusion_key, definition) in enumerate(
        stages, start=1
    ):
        starting = retained
        excluded = 0 if exclusion_key is None else int(cohort.audit_counts[exclusion_key])
        retained = starting - excluded
        notes = ""
        if stage_id == "body_weight_record":
            notes = (
                f"orphan body-weight rows: {int(cohort.audit_counts.get('orphan_weight_rows', 0))}"
            )
        rows.append(
            {
                "flow_order": order,
                "row_type": "stage",
                "stage_id": stage_id,
                "stage_label": label,
                "starting_n": starting,
                "excluded_n": excluded,
                "retained_n": retained,
                "percent_of_enrolled": 100.0 * retained / enrolled,
                "exclusion_definition": definition,
                "notes": notes,
            }
        )
    if retained != int(cohort.audit_counts["included_members"]):
        raise ValueError("Cohort audit counts do not reconcile to included_members")
    return pd.DataFrame.from_records(rows, columns=REPORTING_SCHEMAS["nineam_cohort_flow.csv"])


def _categorical_rows(
    scope_order: int,
    scope: str,
    members: pd.DataFrame,
    characteristic_order: int,
    characteristic: str,
    label: str,
    levels: tuple[str, ...],
    notes: str,
) -> list[dict[str, object]]:
    """Create complete-level categorical rows for one sample characteristic.

    Args:
        scope_order: Fixed output scope number.
        scope: Human-readable scope label.
        members: Members in the current scope.
        characteristic_order: Fixed characteristic number.
        characteristic: Source column name.
        label: Publication label.
        levels: Globally fixed output levels.
        notes: Interpretation note carried on every row.

    Returns:
        Closed-schema categorical records, including observed zero cells.

    Side effects:
        None.

    Statistical intent:
        Keeps subgroup percentages on comparable observed denominators.
    """
    values = members[characteristic].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    observed_n = int((~missing).sum())
    records = []
    for level_order, level in enumerate(levels, start=1):
        if level == "Missing":
            count = int(missing.sum())
            denominator = len(values)
        else:
            count = int(values.eq(level).sum())
            denominator = observed_n
        records.append(
            {
                "scope_order": scope_order,
                "scope": scope,
                "characteristic_order": characteristic_order,
                "characteristic": characteristic,
                "characteristic_label": label,
                "level_order": level_order,
                "level": level,
                "summary_type": "categorical",
                "unit": "",
                "n_total": int(len(values)),
                "n_observed": observed_n,
                "n_missing": int(missing.sum()),
                "count": count,
                "percentage": 100.0 * count / denominator if denominator else np.nan,
                "mean": np.nan,
                "standard_deviation": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
                "notes": notes,
            }
        )
    return records


def _categorical_levels(values: pd.Series) -> tuple[str, ...]:
    """Return sorted observed labels plus one explicit blank/null level."""
    normalized = values.astype("string").str.strip()
    missing = normalized.isna() | normalized.eq("")
    levels = tuple(
        sorted(str(value) for value in normalized.loc[~missing].unique())
    )
    if missing.any() and "Missing" not in levels:
        levels = (*levels, "Missing")
    return levels


def _build_sample_characteristics(members: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive member-type, sex, and ethnicity rows.

    Args:
        members: One row per primary-cohort member.

    Returns:
        Closed Table 1-style sample-characteristic rows.

    Side effects:
        None.

    Statistical intent:
        Describes the retained sample without using ethnicity as a model term.
    """
    sex_levels = _categorical_levels(members["sex"])
    ethnicity_levels = _categorical_levels(members["ethnicity"])
    records: list[dict[str, object]] = []
    for scope_order, scope, frame in _scopes(members):
        if scope == "Overall":
            records.extend(
                _categorical_rows(
                    scope_order,
                    scope,
                    frame,
                    1,
                    "member_type",
                    "Member type",
                    PRIMARY_MEMBER_TYPES,
                    "Confirmed primary analysis conditions",
                )
            )
        records.extend(
            _categorical_rows(
                scope_order,
                scope,
                frame,
                2,
                "sex",
                "Sex",
                sex_levels,
                "Prespecified LASSO candidate",
            )
        )
        records.extend(
            _categorical_rows(
                scope_order,
                scope,
                frame,
                3,
                "ethnicity",
                "Ethnicity response combination",
                ethnicity_levels,
                "Descriptive only; source multi-select combinations are retained",
            )
        )
    return pd.DataFrame.from_records(
        records,
        columns=REPORTING_SCHEMAS["nineam_sample_characteristics.csv"],
    )


def _build_outcomes(members: pd.DataFrame) -> pd.DataFrame:
    """Summarize continuous and binary outcomes across fixed scopes.

    Args:
        members: Primary members with confirmed endpoint outcome fields.

    Returns:
        Overall and member-type outcome rows with Wilson response intervals.

    Side effects:
        None.

    Statistical intent:
        Reports the prespecified continuous primary and binary secondary outcome.
    """
    outcomes = (
        ("first_weight", "First weight", "continuous", "pounds"),
        ("last_weight", "Last weight", "continuous", "pounds"),
        ("absolute_weight_loss", "Absolute weight loss", "continuous", "pounds"),
        ("percentage_loss", "Percentage weight loss", "continuous", "percent"),
        ("weight_days", "Measurement interval", "continuous", "days"),
        (
            "weight_loss_success_5pct",
            "At least 5% weight loss",
            "binary",
            "percent",
        ),
    )
    rows = []
    for scope_order, scope, frame in _scopes(members):
        for outcome_order, (name, label, outcome_type, unit) in enumerate(
            outcomes, start=1
        ):
            if outcome_type == "continuous":
                summary = _continuous_statistics(frame[name])
                row = {
                    "scope_order": scope_order,
                    "scope": scope,
                    "outcome_order": outcome_order,
                    "outcome": name,
                    "outcome_label": label,
                    "outcome_type": outcome_type,
                    "unit": unit,
                    **summary,
                    "numerator": np.nan,
                    "denominator": np.nan,
                    "percentage": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_level": np.nan,
                    "ci_method": "",
                }
            else:
                values = frame[name].astype("boolean")
                observed = values.dropna()
                numerator = int(observed.sum())
                denominator = int(len(observed))
                lower, upper = wilson_confidence_interval(numerator, denominator)
                row = {
                    "scope_order": scope_order,
                    "scope": scope,
                    "outcome_order": outcome_order,
                    "outcome": name,
                    "outcome_label": label,
                    "outcome_type": outcome_type,
                    "unit": unit,
                    "n_total": int(len(values)),
                    "n_observed": denominator,
                    "n_missing": int(values.isna().sum()),
                    "mean": np.nan,
                    "standard_deviation": np.nan,
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "numerator": numerator,
                    "denominator": denominator,
                    "percentage": 100.0 * numerator / denominator,
                    "ci_lower": 100.0 * lower,
                    "ci_upper": 100.0 * upper,
                    "ci_level": 0.95,
                    "ci_method": "Wilson",
                }
            rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_outcomes_by_member_type.csv"],
    )


def _build_engagement(members: pd.DataFrame) -> pd.DataFrame:
    """Summarize confirmed engagement features and rate sensitivity.

    Args:
        members: Primary member features through each last-weight date.

    Returns:
        Fixed-scope engagement distribution rows.

    Side effects:
        None; interval-normalized sensitivity is computed on a copy.

    Statistical intent:
        Preserves the tenure rate while exposing its denominator sensitivity.
    """
    enriched = members.copy()
    enriched[
        "engagement_volume_repeatable_rate_weight_days_sensitivity"
    ] = enriched["engagement_volume_repeatable"] / enriched["weight_days"].clip(
        lower=7.0
    )
    metrics = (
        (
            "engagement_breadth",
            "Engagement breadth",
            "activity types",
            "",
        ),
        (
            "engagement_volume_repeatable",
            "Repeatable engagement volume",
            "events",
            "",
        ),
        (
            "engagement_volume_repeatable_rate",
            "Tenure-normalized repeatable engagement rate",
            "events per day",
            "max(tenure_days, 7)",
        ),
        ("tenure_days", "Program tenure", "days", ""),
        (
            "engagement_volume_repeatable_rate_weight_days_sensitivity",
            "Weight-interval-normalized repeatable engagement rate sensitivity",
            "events per day",
            "max(weight_days, 7)",
        ),
    )
    rows = []
    for scope_order, scope, frame in _scopes(enriched):
        for metric_order, (metric, label, unit, denominator) in enumerate(
            metrics, start=1
        ):
            summary = _continuous_statistics(frame[metric])
            observed = pd.to_numeric(frame[metric], errors="raise").dropna()
            rows.append(
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "metric_order": metric_order,
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    **summary,
                    "zero_n": int(observed.eq(0.0).sum()),
                    "reached_n": int(observed.gt(0.0).sum()),
                    "exposure_window_definition": (
                        "retained events through each member's last-weight date; "
                        "no lower date cutoff"
                        if metric != "tenure_days"
                        else "source-supplied program tenure at outcome observation"
                    ),
                    "denominator_definition": denominator,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_engagement_by_member_type.csv"],
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust one finite p-value family by Benjamini--Hochberg.

    Args:
        p_values: One-dimensional finite values between zero and one.

    Returns:
        Adjusted p-values aligned to the original order.

    Side effects:
        None.

    Statistical intent:
        Controls the activity-family false discovery rate in one operation.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must lie between zero and one")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * values.size / np.arange(1, values.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(monotone)
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def _fit_activity_association(
    members: pd.DataFrame,
    event_counts: pd.Series,
) -> dict[str, float | str]:
    """Fit one standardized activity association with HC3 covariance.

    Args:
        members: Aligned outcome, baseline, type, and duration rows.
        event_counts: Aligned zero-filled event counts.

    Returns:
        The activity coefficient and HC3 uncertainty fields.

    Side effects:
        None; statsmodels fits an in-memory OLS equation.

    Statistical intent:
        Implements the prespecified adjusted descriptive activity sensitivity.
    """
    required = {"percentage_loss", "first_weight", "member_type", "weight_days"}
    _require_columns(members, required, "Activity members")
    counts = pd.to_numeric(event_counts, errors="raise").to_numpy(dtype=float)
    if len(counts) != len(members) or not np.isfinite(counts).all():
        raise ValueError("Activity event counts must be finite and row-aligned")
    transformed = np.log1p(counts)
    scale = float(np.std(transformed, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Activity predictor has no estimable variation")
    standardized = (transformed - float(np.mean(transformed))) / scale
    days = pd.to_numeric(members["weight_days"], errors="raise").to_numpy(float)
    centered_days = days - float(np.mean(days))
    first_weight = pd.to_numeric(
        members["first_weight"], errors="raise"
    ).to_numpy(float)
    outcome = pd.to_numeric(
        members["percentage_loss"], errors="raise"
    ).to_numpy(float)
    active = members["member_type"].astype(str).eq(PRIMARY_MEMBER_TYPES[0]).to_numpy(float)
    design = np.column_stack(
        [
            np.ones(len(members)),
            first_weight,
            active,
            centered_days,
            centered_days**2,
            standardized,
        ]
    )
    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("Activity model values must be finite")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Activity model design is rank deficient")
    fitted = sm.OLS(outcome, design).fit(cov_type="HC3")
    index = design.shape[1] - 1
    confidence = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    return {
        "estimate": float(fitted.params[index]),
        "standard_error": float(fitted.bse[index]),
        "test_statistic": float(fitted.tvalues[index]),
        "reference_distribution": "normal",
        "degrees_of_freedom": np.nan,
        "p_value": float(fitted.pvalues[index]),
        "ci_lower": float(confidence[index, 0]),
        "ci_upper": float(confidence[index, 1]),
        "ci_level": 0.95,
        "covariance_estimator": "HC3",
    }


def _build_activity_summary(features: FeatureResult) -> pd.DataFrame:
    """Build all activity descriptions and eligible adjusted tests.

    Args:
        features: Primary member features, full-source type classification, and
            sparse cohort-window event counts.

    Returns:
        Nineteen alphabetic activity rows with one BH-adjusted tested family.

    Side effects:
        None.

    Statistical intent:
        Describes zero-inclusive reach and treats associations as noncausal.
    """
    summary = features.event_type_summary.copy()
    counts = features.member_event_counts.copy()
    _require_columns(summary, {"event_type", "is_repeatable"}, "Event summary")
    _require_columns(
        counts,
        {"member_id", "event_type", "event_count", "is_repeatable"},
        "Member event counts",
    )
    if summary["event_type"].duplicated().any():
        raise ValueError("Event summary must contain one row per event_type")
    event_types = tuple(sorted(summary["event_type"].astype(str)))
    unknown = sorted(set(event_types).difference(_ACTIONABILITY_BY_EVENT))
    missing = sorted(set(_ACTIONABILITY_BY_EVENT).difference(event_types))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        if missing:
            details.append("missing=" + ",".join(missing))
        raise ValueError("Activity actionability mapping mismatch: " + "; ".join(details))
    if counts.duplicated(["member_id", "event_type"]).any():
        raise ValueError("Member event counts contain duplicate member-event pairs")
    members = features.member_features.sort_values(
        "member_id", kind="stable"
    ).reset_index(drop=True)
    member_ids = set(members["member_id"].astype(str))
    if not set(counts["member_id"].astype(str)).issubset(member_ids):
        raise ValueError("Member event counts contain nonanalysis members")
    repeatability = summary.set_index("event_type")["is_repeatable"].astype(bool)
    rows: list[dict[str, object]] = []
    for event_order, event_type in enumerate(event_types, start=1):
        observed = counts.loc[counts["event_type"].astype(str).eq(event_type)].copy()
        expected_repeatable = bool(repeatability.loc[event_type])
        if not observed.empty and not observed["is_repeatable"].astype(bool).eq(
            expected_repeatable
        ).all():
            raise ValueError(f"Repeatability mismatch for activity {event_type}")
        observed_map = observed.set_index(observed["member_id"].astype(str))[
            "event_count"
        ]
        zero_filled = members["member_id"].astype(str).map(observed_map).fillna(0)
        zero_filled = pd.to_numeric(zero_filled, errors="raise").astype("int64")
        if zero_filled.lt(0).any():
            raise ValueError("Activity event counts cannot be negative")
        reached_values = zero_filled.loc[zero_filled.gt(0)]
        n_reached = int(len(reached_values))
        base_row: dict[str, object] = {
            "event_order": event_order,
            "event_type": event_type,
            "actionability_class": _ACTIONABILITY_BY_EVENT[event_type],
            "is_repeatable": expected_repeatable,
            "n_analysis": int(len(members)),
            "n_reached": n_reached,
            "n_zero": int(zero_filled.eq(0).sum()),
            "total_events": int(zero_filled.sum()),
            "events_per_reached_member": (
                float(zero_filled.sum() / n_reached) if n_reached else np.nan
            ),
            "median_events_among_reached": (
                float(reached_values.median()) if n_reached else np.nan
            ),
            "q1_events_among_reached": (
                float(reached_values.quantile(0.25, interpolation="linear"))
                if n_reached
                else np.nan
            ),
            "q3_events_among_reached": (
                float(reached_values.quantile(0.75, interpolation="linear"))
                if n_reached
                else np.nan
            ),
            "predictor": "event_count",
            "predictor_transform": "standardized_log1p",
            "effect_unit": "percentage_points_per_1_sd_log1p_events",
            "estimate": np.nan,
            "standard_error": np.nan,
            "test_statistic": np.nan,
            "reference_distribution": "",
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_level": np.nan,
            "fdr_method": "Benjamini-Hochberg",
            "fdr_family": "eligible_activity_types_reach_ge_30",
            "fdr_test_count": 0,
            "fdr_adjusted_p_value": np.nan,
            "covariance_estimator": "",
            "adjustment_terms": (
                "first_weight + member_type + centered_weight_days + "
                "centered_weight_days_squared"
            ),
            "reference_member_type": REFERENCE_MEMBER_TYPE,
            "test_status": "descriptive_low_reach",
            "interpretation": "descriptive_low_reach_no_association_test",
        }
        if n_reached >= _ACTIVITY_MINIMUM_REACH:
            try:
                inference = _fit_activity_association(members, zero_filled)
            except ValueError:
                base_row["test_status"] = "not_estimable"
                base_row["interpretation"] = "eligible_reach_but_model_not_estimable"
            else:
                base_row.update(inference)
                base_row["test_status"] = "tested"
                base_row["interpretation"] = "exploratory_noncausal_association"
        rows.append(base_row)
    tested_indices = [
        index for index, row in enumerate(rows) if row["test_status"] == "tested"
    ]
    adjusted = _benjamini_hochberg(
        np.array([rows[index]["p_value"] for index in tested_indices], dtype=float)
    )
    for row in rows:
        row["fdr_test_count"] = len(tested_indices)
    for index, adjusted_value in zip(tested_indices, adjusted, strict=True):
        rows[index]["fdr_adjusted_p_value"] = float(adjusted_value)
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_engagement_activity_summary.csv"],
    )


def _build_modules(members: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique-title module completions across fixed scopes.

    Args:
        members: Primary feature rows with domain counts and proportions.

    Returns:
        Overall, core, and extension completion rows with support warnings.

    Side effects:
        None.

    Statistical intent:
        Retains observed Coaching zeros without asserting extension availability.
    """
    count_columns = [f"{name}_count" for name, _, _, _ in _MODULES[1:]]
    coaching = members.loc[members["member_type"].eq(REFERENCE_MEMBER_TYPE)]
    extension_columns = [
        "module_mindset_count",
        "module_nutrition_count",
        "module_physical_activity_count",
    ]
    extension_glp_only = bool(coaching[extension_columns].sum().sum() == 0)
    enriched = members.copy()
    enriched["module_mean_count"] = enriched[count_columns].sum(axis="columns")
    rows = []
    for scope_order, scope, frame in _scopes(enriched):
        for module_order, (variable, label, group, denominator) in enumerate(
            _MODULES, start=1
        ):
            count_variable = "module_mean_count" if variable == "module_mean" else f"{variable}_count"
            count_summary = _continuous_statistics(frame[count_variable])
            proportion_summary = _continuous_statistics(frame[variable])
            counts = pd.to_numeric(frame[count_variable], errors="raise")
            completed = int(counts.gt(0).sum())
            if group == "extension":
                availability = "extension_availability_not_verifiable"
                support = (
                    "glp_1_weight_loss_only"
                    if extension_glp_only
                    else "both_member_types_observed"
                )
                comparability = (
                    "not_cross_group_comparable"
                    if extension_glp_only
                    else "descriptive_both_groups"
                )
            elif group == "core":
                availability = "fixed_core_title_set"
                support = "both_member_types_observed"
                comparability = "descriptive_both_groups"
            else:
                availability = "mixed_core_and_unverified_extensions"
                support = "both_member_types_observed_with_extension_limitation"
                comparability = "limited_by_extension_support"
            rows.append(
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "module_order": module_order,
                    "module_variable": variable,
                    "module_label": label,
                    "module_group": group,
                    "available_title_denominator": (
                        denominator if denominator is not None else np.nan
                    ),
                    "availability_status": availability,
                    "support_scope": support,
                    "n_members": int(len(frame)),
                    "n_members_with_completion": completed,
                    "n_members_without_completion": int(len(frame) - completed),
                    "member_completion_percentage": (
                        100.0 * completed / len(frame) if len(frame) else np.nan
                    ),
                    "total_unique_completions": int(counts.sum()),
                    "mean_completion_count": count_summary["mean"],
                    "standard_deviation_count": count_summary["standard_deviation"],
                    "median_completion_count": count_summary["median"],
                    "q1_completion_count": count_summary["q1"],
                    "q3_completion_count": count_summary["q3"],
                    "minimum_completion_count": count_summary["minimum"],
                    "maximum_completion_count": count_summary["maximum"],
                    "mean_completion_proportion": proportion_summary["mean"],
                    "standard_deviation_proportion": proportion_summary[
                        "standard_deviation"
                    ],
                    "zero_completion_group_flag": bool(counts.eq(0).all()),
                    "completion_definition": (
                        "one unique answered member-title record through the "
                        "member's last-weight date"
                    ),
                    "cross_group_comparability": comparability,
                    "interpretation": "descriptive_noncausal_completion_summary",
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_modules_by_member_type.csv"],
    )


def _base_model_scores(base_model_cv: pd.DataFrame) -> pd.DataFrame:
    """Validate paired folds and calculate model-level RMSE/MAE means.

    Args:
        base_model_cv: Fold-level common-target prediction scores.

    Returns:
        Model-indexed mean RMSE and MAE.

    Side effects:
        None.

    Statistical intent:
        Supports ranking without cross-family likelihood comparisons.
    """
    required = {"repeat", "fold", "model", "n_test", "rmse", "mae"}
    _require_columns(base_model_cv, required, "Base-model CV")
    table = base_model_cv.copy()
    if table.duplicated(["repeat", "fold", "model"]).any():
        raise ValueError("Base-model CV contains duplicate model-fold rows")
    if set(table["model"].astype(str)) != set(_MODEL_ORDER):
        raise ValueError("Base-model CV must contain both approved model families")
    plans = []
    for model in _MODEL_ORDER:
        plan = table.loc[table["model"].eq(model), ["repeat", "fold", "n_test"]]
        plans.append(plan.sort_values(["repeat", "fold"]).reset_index(drop=True))
    if not plans[0].equals(plans[1]):
        raise ValueError("Base-model CV families do not share the same fold plan")
    return table.groupby("model", sort=True)[["rmse", "mae"]].mean()


def _build_base_model_summary(
    base_model_cv: pd.DataFrame,
    locked_model: LockedModelResult,
) -> pd.DataFrame:
    """Aggregate common-target predictive performance into closed rows.

    Args:
        base_model_cv: Fold-level paired RMSE and MAE values.
        locked_model: Final result carrying the chosen base-family identifier.

    Returns:
        Two metrics per model with deterministic rank and winner labels.

    Side effects:
        None.

    Statistical intent:
        Ranks families only on held-out raw last-weight errors.
    """
    scores = _base_model_scores(base_model_cv)
    winner = locked_model.winning_base_model
    if winner not in _MODEL_ORDER:
        raise ValueError("Locked model contains an unsupported base-model winner")
    minimum_rmse = float(scores["rmse"].min())
    eligible = scores.index[scores["rmse"].eq(minimum_rmse)]
    minimum_mae = float(scores.loc[eligible, "mae"].min())
    finalists = set(
        name
        for name in eligible
        if scores.loc[name, "mae"] == minimum_mae
    )
    calculated = (
        "percentage_loss_ols"
        if "percentage_loss_ols" in finalists
        else sorted(finalists)[0]
    )
    if calculated != winner:
        raise ValueError("Locked winner does not follow base-model CV ranking")
    rows = []
    for model_order, model in enumerate(_MODEL_ORDER, start=1):
        model_rows = base_model_cv.loc[base_model_cv["model"].eq(model)]
        repeat_totals = model_rows.groupby("repeat", sort=True)["n_test"].sum()
        if repeat_totals.nunique() != 1:
            raise ValueError("Base-model CV repeats do not have equal member totals")
        for metric_order, metric in enumerate(("rmse", "mae"), start=1):
            values = pd.to_numeric(model_rows[metric], errors="raise")
            rows.append(
                {
                    "model_order": model_order,
                    "model_id": model,
                    "model_family": (
                        "longitudinal_marginal_gls"
                        if model == "log_compound_symmetry_gls"
                        else "ordinary_least_squares"
                    ),
                    "modeled_outcome": (
                        "log_weight_trajectory"
                        if model == "log_compound_symmetry_gls"
                        else "percentage_weight_loss"
                    ),
                    "prediction_target": "raw_last_weight",
                    "metric_order": metric_order,
                    "metric": metric,
                    "metric_unit": "pounds",
                    "n_members": int(repeat_totals.iloc[0]),
                    "cv_repeats": int(model_rows["repeat"].nunique()),
                    "cv_folds": int(model_rows["fold"].nunique()),
                    "n_fold_scores": int(len(model_rows)),
                    "n_test_predictions": int(model_rows["n_test"].sum()),
                    "mean_score": float(values.mean()),
                    "standard_deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else np.nan
                    ),
                    "minimum_score": float(values.min()),
                    "maximum_score": float(values.max()),
                    "fold_plan_id": "shared_grouped_stratified_member_folds",
                    "score_aggregation": "unweighted_mean_across_repeat_folds",
                    "rank": 1 if model == winner else 2,
                    "is_winner": model == winner,
                    "winner_rule": (
                        "lowest_mean_rmse_then_mean_mae_then_percentage_loss_ols"
                    ),
                    "tie_tolerance": _TIE_TOLERANCE,
                    "validation_scope": "paired_held_out_raw_last_weight",
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS[
            "nineam_base_model_comparison_summary.csv"
        ],
    )


def _candidate_metadata(candidate: str) -> tuple[str, str, str, str]:
    """Map one canonical LASSO candidate to reporting labels and units.

    Args:
        candidate: Exact penalized-design candidate name.

    Returns:
        Label, role, support scope, and coefficient unit.

    Side effects:
        None.

    Statistical intent:
        Keeps candidate interpretation stable across both module specifications.
    """
    labels = {
        "engagement_volume_repeatable": "Repeatable engagement volume",
        "engagement_volume_repeatable_rate": "Tenure-normalized engagement rate",
        "engagement_breadth": "Engagement breadth",
        "tenure_days": "Program tenure days",
        "module_mean": "Overall four-domain module mean",
        "module_core": "Core module completion proportion",
        "module_mindset": "Mindset module completion proportion",
        "module_nutrition": "Nutrition module completion proportion",
        "module_physical_activity": "Physical-activity module completion proportion",
    }
    if candidate.startswith("sex[") and candidate.endswith("]"):
        return (
            f"Sex contrast: {candidate[4:-1]}",
            "demographic",
            "primary_two_group",
            "percentage_points_contrast",
        )
    if candidate not in labels:
        raise ValueError(f"Unsupported LASSO candidate: {candidate}")
    role = "engagement" if candidate.startswith("engagement_") or candidate == "tenure_days" else "module"
    if candidate == "module_mean":
        support = "both_member_types_observed_with_extension_limitation"
    elif candidate in {
        "module_mindset",
        "module_nutrition",
        "module_physical_activity",
    }:
        support = "glp_1_weight_loss_only"
    else:
        support = "primary_two_group"
    unit = (
        "percentage_points_per_proportion"
        if candidate.startswith("module_")
        else "percentage_points_per_source_unit"
    )
    return labels[candidate], role, support, unit


def _build_lasso_selection(
    table: pd.DataFrame,
    expected_spec: str,
    locked_model: LockedModelResult,
) -> pd.DataFrame:
    """Expand one enriched LASSO input into its closed publication schema.

    Args:
        table: Candidate rows with tuning, full-fit, and resampling provenance.
        expected_spec: Required ``mean`` or ``domains`` specification.
        locked_model: Final model carrying the winning spec and locked candidates.

    Returns:
        Ordered closed-schema LASSO rows.

    Side effects:
        None.

    Statistical intent:
        Separates predictive tuning, stability, and post-selection locking.
    """
    _require_columns(table, _LASSO_INPUT_COLUMNS, f"{expected_spec} LASSO selection")
    prepared = table.loc[:, list(_LASSO_INPUT_COLUMNS)].copy()
    if set(prepared["module_spec"].astype(str)) != {expected_spec}:
        raise ValueError(f"LASSO input must contain only {expected_spec} rows")
    candidate_names = prepared["candidate"].astype(str)
    if candidate_names.duplicated().any():
        raise ValueError("LASSO candidates must be unique within a specification")
    observed_order = pd.to_numeric(prepared["candidate_order"], errors="raise")
    if (
        not np.isfinite(observed_order.to_numpy(dtype=float)).all()
        or not observed_order.eq(np.floor(observed_order)).all()
        or not observed_order.gt(0).all()
    ):
        raise ValueError("LASSO candidate_order must contain positive integers")
    prepared["candidate_order"] = observed_order.astype("int64")
    if prepared["candidate_order"].duplicated().any():
        raise ValueError("LASSO candidate_order must be unique")
    approved = _SPECIFICATION_CANDIDATES[expected_spec]
    incompatible = sorted(
        set(candidate_names).intersection(_CANONICAL_CANDIDATE_ORDER).difference(
            approved
        )
    )
    if incompatible:
        raise ValueError(
            f"LASSO {expected_spec} specification contains incompatible candidates: "
            + ", ".join(incompatible)
        )
    for candidate, order in zip(
        candidate_names,
        prepared["candidate_order"],
        strict=True,
    ):
        canonical_order = _CANONICAL_CANDIDATE_ORDER.get(candidate)
        if canonical_order is not None and int(order) != canonical_order:
            raise ValueError("LASSO candidates must use canonical candidate_order values")
    extra_candidates = sorted(
        set(candidate_names).difference(_CANONICAL_CANDIDATE_ORDER)
    )
    if any(
        not candidate.startswith("sex[") or not candidate.endswith("]")
        for candidate in extra_candidates
    ):
        raise ValueError("Unsupported noncanonical LASSO candidate")
    expected_extra_order = {
        candidate: len(_CANONICAL_CANDIDATE_ORDER) + offset
        for offset, candidate in enumerate(extra_candidates, start=1)
    }
    if any(
        int(order) != expected_extra_order[candidate]
        for candidate, order in zip(
            candidate_names,
            prepared["candidate_order"],
            strict=True,
        )
        if candidate in expected_extra_order
    ):
        raise ValueError(
            "Noncanonical sex contrasts must use deterministic appended candidate_order values"
        )

    numeric_columns = (
        "n_resamples",
        "selection_count",
        "selection_frequency",
        "selection_threshold",
    )
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    numeric_values = prepared.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("LASSO resampling provenance must be finite")
    for column in ("n_resamples", "selection_count"):
        if not prepared[column].eq(np.floor(prepared[column])).all():
            raise ValueError(f"LASSO {column} must contain integers")
    if not prepared["n_resamples"].gt(0).all():
        raise ValueError("LASSO n_resamples must be positive")
    if (
        prepared["selection_count"].lt(0).any()
        or prepared["selection_count"].gt(prepared["n_resamples"]).any()
    ):
        raise ValueError("LASSO selection_count must lie between zero and n_resamples")
    expected_frequency = prepared["selection_count"] / prepared["n_resamples"]
    if not np.array_equal(
        expected_frequency.to_numpy(dtype=float),
        prepared["selection_frequency"].to_numpy(dtype=float),
    ):
        raise ValueError(
            "LASSO selection_count / n_resamples must equal selection_frequency"
        )
    if not prepared["selection_threshold"].between(
        0.0,
        1.0,
        inclusive="right",
    ).all():
        raise ValueError("LASSO selection_threshold must lie in (0, 1]")
    for column in ("selected_at_threshold", "excluded_from_full_fit"):
        if not prepared[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"LASSO {column} must contain booleans")
    excluded_flags = prepared["excluded_from_full_fit"].astype(bool)
    selected_flags = prepared["selected_at_threshold"].astype(bool)
    expected_selected = prepared["selection_frequency"].ge(
        prepared["selection_threshold"]
    ) & ~excluded_flags
    if not np.array_equal(
        selected_flags.to_numpy(dtype=bool),
        expected_selected.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "LASSO selected_at_threshold must equal frequency >= threshold "
            "and require a nonexcluded full fit"
        )
    prepared = prepared.sort_values("candidate_order", kind="stable").reset_index(drop=True)
    constant_columns = (
        "lambda_ratio",
        "full_sample_lambda",
        "full_sample_lambda_max",
        "cv_selection_rule",
        "cv_mean_mse",
        "cv_standard_error",
        "fold_plan_id",
        "n_resamples",
        "subsample_fraction",
        "selection_threshold",
    )
    for column in constant_columns:
        if prepared[column].nunique(dropna=False) != 1:
            raise ValueError(f"LASSO {column} must be consistent within specification")
    winning = expected_spec == locked_model.winning_module_spec
    locked_candidates = set(locked_model.selected_candidates)
    stable_candidates = set(
        prepared.loc[
            prepared["selected_at_threshold"].astype(bool)
            & ~prepared["excluded_from_full_fit"].astype(bool),
            "candidate",
        ].astype(str)
    ).intersection(approved)
    unsupported_locked = locked_candidates.difference(approved)
    if winning and unsupported_locked:
        raise ValueError(
            "Locked model contains candidates outside its approved specification"
        )
    if winning and stable_candidates != locked_candidates:
        raise ValueError("Winning LASSO stable candidates do not match locked model")
    rows = []
    for row in prepared.itertuples(index=False):
        candidate = str(row.candidate)
        label, role, support, unit = _candidate_metadata(candidate)
        excluded = bool(row.excluded_from_full_fit)
        selected = bool(row.selected_at_threshold)
        canonical = candidate in approved
        locked = winning and canonical and candidate in locked_candidates
        if not canonical:
            status = "ineligible_noncanonical_candidate"
            reason = (
                "only the canonical sex contrast sex[MALE] is eligible for the "
                "locked model"
            )
        elif locked:
            status = "locked"
            reason = ""
        elif not winning:
            status = "nonwinning_specification"
            reason = "module specification had higher paired-CV MSE"
        elif excluded:
            status = "inestimable_full_fit"
            reason = "candidate excluded from the full-sample fit"
        else:
            status = "below_stability_threshold"
            reason = "selection frequency below prespecified threshold"
        reference = ""
        if candidate.startswith("sex["):
            observed_level = candidate[4:-1]
            reference = "FEMALE" if observed_level in {"MALE", "M"} else "reference sex level"
        rows.append(
            {
                "specification_order": 1 if expected_spec == "mean" else 2,
                "module_spec": expected_spec,
                "is_winning_specification": winning,
                "candidate_order": int(row.candidate_order),
                "candidate": candidate,
                "candidate_label": label,
                "candidate_role": role,
                "support_scope": support,
                "reference_level": reference,
                "penalty_status": "penalized",
                "selected_lambda_ratio": float(row.lambda_ratio),
                "full_sample_lambda": float(row.full_sample_lambda),
                "full_sample_lambda_max": float(row.full_sample_lambda_max),
                "cv_selection_rule": str(row.cv_selection_rule),
                "cv_mean_mse": float(row.cv_mean_mse),
                "cv_standard_error": float(row.cv_standard_error),
                "fold_plan_id": str(row.fold_plan_id),
                "n_resamples": int(row.n_resamples),
                "subsample_fraction": float(row.subsample_fraction),
                "selection_threshold": float(row.selection_threshold),
                "full_sample_coefficient": float(row.full_sample_coefficient),
                "full_sample_standardized_coefficient": float(
                    row.full_sample_standardized_coefficient
                ),
                "selection_count": int(row.selection_count),
                "selection_frequency": float(row.selection_frequency),
                "selected_at_threshold": selected,
                "eligible_for_locked_model": locked,
                "locked_model_status": status,
                "exclusion_reason": reason,
                "coefficient_unit": unit,
                "interpretation": "exploratory_stability_not_significance",
            }
        )
    filename = (
        "nineam_lasso_mean_selection.csv"
        if expected_spec == "mean"
        else "nineam_lasso_domain_selection.csv"
    )
    return pd.DataFrame.from_records(rows, columns=REPORTING_SCHEMAS[filename])


def _ordered_locked_table(
    table: pd.DataFrame,
    schema: tuple[str, ...],
    label: str,
    secondary_order: str | None,
) -> pd.DataFrame:
    """Validate and deterministically order one locked-model aggregate table.

    Args:
        table: Task 2 aggregate dataframe.
        schema: Exact approved columns.
        label: Error label.
        secondary_order: Optional within-model ordering field.

    Returns:
        A defensive, schema-ordered copy.

    Side effects:
        None.

    Statistical intent:
        Publishes Task 2 estimates without recomputation or row-level leakage.
    """
    if tuple(table.columns) != schema:
        raise ValueError(f"{label} does not match its closed schema")
    copied = table.copy()
    if not set(copied["model_id"]).issubset(_LOCKED_MODEL_ORDER):
        raise ValueError(f"{label} contains an unknown model_id")
    copied["_model_order"] = copied["model_id"].map(
        {name: order for order, name in enumerate(_LOCKED_MODEL_ORDER, start=1)}
    )
    order_columns = ["_model_order"]
    if secondary_order is not None:
        order_columns.append(secondary_order)
    copied = copied.sort_values(order_columns, kind="stable").drop(
        columns="_model_order"
    )
    return copied.reset_index(drop=True)


def _equal_count_chunks(values: np.ndarray) -> list[np.ndarray]:
    """Partition stable-sorted row indices into bins of at least two.

    Args:
        values: Numeric series defining the stable rank order.

    Returns:
        Ordered index arrays with near-equal sizes.

    Side effects:
        None.

    Statistical intent:
        Prevents raw diagnostic points or singleton bins from publication.
    """
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Diagnostic series require at least two points")
    n_bins = min(10, len(values) // 2)
    ordered = np.argsort(values, kind="stable")
    chunks = [chunk for chunk in np.array_split(ordered, n_bins) if len(chunk)]
    if any(len(chunk) < 2 for chunk in chunks):
        raise ValueError("Diagnostic aggregation produced a singleton bin")
    return chunks


def _binned_rows(
    model_id: str,
    diagnostic_type: str,
    series: str,
    order: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize one two-dimensional diagnostic series in equal-count bins.

    Args:
        model_id: Locked model identifier.
        diagnostic_type: Controlled diagnostic family name.
        series: Controlled plotted-series name.
        order: Series-level diagnostic order.
        x_values: Values defining stable bin ranks and plotted x coordinates.
        y_values: Values summarized within each x bin.

    Returns:
        Closed diagnostic binned-series records.

    Side effects:
        None.

    Statistical intent:
        Retains visual diagnostic shape without publishing member-level points.
    """
    rows = []
    for bin_order, indices in enumerate(_equal_count_chunks(x_values), start=1):
        x_bin = x_values[indices]
        y_bin = y_values[indices]
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_order": order,
                "diagnostic_type": diagnostic_type,
                "row_type": "plot_summary",
                "series": series,
                "bin_order": bin_order,
                "bin_method": "equal_count_stable_rank",
                "bin_lower": float(np.min(x_bin)),
                "bin_upper": float(np.max(x_bin)),
                "bin_count": int(len(indices)),
                "x_value": float(np.mean(x_bin)),
                "y_value": float(np.mean(y_bin)),
                "y_lower": float(np.quantile(y_bin, 0.25, method="linear")),
                "y_upper": float(np.quantile(y_bin, 0.75, method="linear")),
                "metric": "",
                "value": np.nan,
                "threshold": np.nan,
                "flag": pd.NA,
                "status": "descriptive",
                "interpretation": "aggregate diagnostic bin; no member-level points",
            }
        )
    return rows


def _aggregate_diagnostic_points(points: pd.DataFrame) -> pd.DataFrame:
    """Aggregate internal Task 2 point diagnostics into allowed plot series.

    Args:
        points: No-ID fitted, residual, leverage, and Cook's values by model.

    Returns:
        Binned residual, normal-Q-Q, leverage, and Cook's rows.

    Side effects:
        None.

    Statistical intent:
        Supports model checking while preserving the aggregate output boundary.
    """
    required = {
        "model_id",
        "fitted_value",
        "residual",
        "standardized_residual",
        "leverage",
        "cooks_distance",
    }
    _require_columns(points, required, "Diagnostic points")
    if "member_id" in points.columns:
        raise ValueError("Diagnostic points cannot contain member_id")
    rows: list[dict[str, object]] = []
    for model_id in _LOCKED_MODEL_ORDER:
        frame = points.loc[points["model_id"].eq(model_id)].reset_index(drop=True)
        if frame.empty:
            continue
        fitted = frame["fitted_value"].to_numpy(float)
        residual = frame["residual"].to_numpy(float)
        standardized = frame["standardized_residual"].to_numpy(float)
        leverage = frame["leverage"].to_numpy(float)
        cooks = frame["cooks_distance"].to_numpy(float)
        rows.extend(
            _binned_rows(
                model_id,
                "residuals_vs_fitted",
                "binned_mean",
                10,
                fitted,
                residual,
            )
        )
        observed_order = np.argsort(standardized, kind="stable")
        observed = standardized[observed_order]
        probabilities = (np.arange(len(observed), dtype=float) + 0.5) / len(observed)
        theoretical = np.array(
            [NormalDist().inv_cdf(float(value)) for value in probabilities],
            dtype=float,
        )
        rows.extend(
            _binned_rows(
                model_id,
                "normal_qq",
                "quantile_pair",
                20,
                theoretical,
                observed,
            )
        )
        ranks = (np.arange(len(frame), dtype=float) + 0.5) / len(frame)
        leverage_order = np.argsort(leverage, kind="stable")
        rows.extend(
            _binned_rows(
                model_id,
                "leverage",
                "binned_mean",
                30,
                ranks,
                leverage[leverage_order],
            )
        )
        cooks_order = np.argsort(cooks, kind="stable")
        rows.extend(
            _binned_rows(
                model_id,
                "cooks_distance",
                "binned_mean",
                40,
                ranks,
                cooks[cooks_order],
            )
        )
    return pd.DataFrame.from_records(rows, columns=DIAGNOSTIC_SUMMARY_COLUMNS)


def _build_diagnostics(locked_model: LockedModelResult) -> pd.DataFrame:
    """Combine Task 2 diagnostic thresholds with aggregate binned series.

    Args:
        locked_model: Final result with summary and internal point diagnostics.

    Returns:
        Closed diagnostic table containing aggregate rows only.

    Side effects:
        None.

    Statistical intent:
        Preserves fitted-model checks without exporting per-member diagnostics.
    """
    summary = locked_model.diagnostic_summary
    if tuple(summary.columns) != tuple(DIAGNOSTIC_SUMMARY_COLUMNS):
        raise ValueError("Locked diagnostic summary does not match closed schema")
    binned = _aggregate_diagnostic_points(locked_model.diagnostic_points)
    existing_series = set(
        zip(
            summary["model_id"],
            summary["diagnostic_type"],
            summary["series"],
            strict=True,
        )
    )
    binned = binned.loc[
        [
            (row.model_id, row.diagnostic_type, row.series) not in existing_series
            for row in binned.itertuples(index=False)
        ]
    ]
    combined = pd.concat([summary.copy(), binned], ignore_index=True)
    # Task 2 already supplies aggregate residual/Q-Q plot rows; reporting adds
    # only missing leverage and Cook's plot series from the internal points.
    allowed_rows = {"summary", "plot_summary"}
    if not set(combined["row_type"]).issubset(allowed_rows):
        raise ValueError("Locked diagnostics contain an unsupported row_type")
    allowed_types = {
        "residuals_vs_fitted",
        "normal_qq",
        "leverage",
        "cooks_distance",
        "global_metric",
    }
    if not set(combined["diagnostic_type"]).issubset(allowed_types):
        raise ValueError("Locked diagnostics contain an unsupported diagnostic_type")
    allowed_series = {"binned_mean", "quantile_pair", "summary_metric"}
    if not set(combined["series"]).issubset(allowed_series):
        raise ValueError("Locked diagnostics contain an unsupported series")
    combined["_model_order"] = combined["model_id"].map(
        {name: order for order, name in enumerate(_LOCKED_MODEL_ORDER, start=1)}
    )
    combined = combined.sort_values(
        ["_model_order", "diagnostic_order", "bin_order"],
        kind="stable",
        na_position="first",
    ).drop(columns="_model_order")
    return combined.reset_index(drop=True).loc[:, list(DIAGNOSTIC_SUMMARY_COLUMNS)]


def _format_number(value: object, digits: int = 2) -> str:
    """Format one finite evidence value deterministically.

    Args:
        value: Numeric scalar or missing value.
        digits: Decimal places for finite values.

    Returns:
        Fixed-decimal text or ``not estimable``.

    Side effects:
        None.

    Statistical intent:
        Prevents editorial tables from changing with platform display defaults.
    """
    if pd.isna(value):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def _build_hypotheses(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Populate a prespecified evidence register from validated aggregate tables.

    Args:
        tables: Earlier scientific reporting tables keyed by filename.

    Returns:
        Deterministic hypothesis questions, statuses, and randomized-pilot fields.

    Side effects:
        None.

    Statistical intent:
        Separates observed associations from future causal tests.
    """
    outcomes = tables["nineam_outcomes_by_member_type.csv"]
    coefficients = tables["nineam_locked_model_coefficients_hc3.csv"]
    activity = tables["nineam_engagement_activity_summary.csv"]
    base = tables["nineam_base_model_comparison_summary.csv"]
    lasso_filenames = (
        "nineam_lasso_mean_selection.csv",
        "nineam_lasso_domain_selection.csv",
    )
    winning_lasso_filenames = tuple(
        filename
        for filename in lasso_filenames
        if tables[filename]["is_winning_specification"].astype(bool).all()
    )
    if len(winning_lasso_filenames) != 1:
        raise ValueError("Exactly one LASSO specification must be the winner")
    winning_lasso_filename = winning_lasso_filenames[0]
    lasso = tables[winning_lasso_filename]
    winning_module_spec = str(lasso["module_spec"].iloc[0])
    primary = coefficients.loc[
        coefficients["model_id"].eq("locked_percentage_loss_primary")
    ]
    type_rows = primary.loc[primary["term"].astype(str).str.startswith("member_type[")]
    type_row = type_rows.iloc[0]
    type_signal = bool(type_row["ci_lower"] > 0 or type_row["ci_upper"] < 0)
    engagement_locked = lasso.loc[
        lasso["eligible_for_locked_model"].astype(bool)
        & lasso["candidate_role"].eq("engagement")
    ]
    module_locked = lasso.loc[
        lasso["eligible_for_locked_model"].astype(bool)
        & lasso["candidate_role"].eq("module")
    ]
    activity_signals = activity.loc[
        activity["fdr_adjusted_p_value"].notna()
        & activity["fdr_adjusted_p_value"].le(0.05)
    ]
    winner = base.loc[base["is_winner"].astype(bool), "model_id"].iloc[0]
    responses = outcomes.loc[
        outcomes["outcome"].eq("weight_loss_success_5pct")
    ].set_index("scope")
    common = {
        "population": "primary two-group cohort",
        "causal_status": "noncausal_observational",
        "randomized_pilot_time_horizon": "12 weeks",
    }
    rows = [
        {
            "hypothesis_order": 1,
            "hypothesis_id": "member_type_percentage_loss",
            "question": "Is member type associated with percentage weight loss after baseline adjustment?",
            "prespecified_expectation": "Active GLP-1 for Weight-loss has greater percentage loss than Coaching Only",
            "estimand": "adjusted percentage-point member-type contrast",
            **common,
            "exposure_or_predictor": "member_type",
            "outcome": "percentage_loss",
            "adjustment_set": "first_weight plus locked stable candidates",
            "evidence_table": "nineam_locked_model_coefficients_hc3.csv",
            "evidence_filter": (
                "model_id=locked_percentage_loss_primary; term_role=base; "
                "contrast=Active GLP-1 for Weight-loss versus Coaching Only"
            ),
            "result_status": "association_detected" if type_signal else "no_clear_association",
            "effect_summary": f"estimate {_format_number(type_row['estimate'])} percentage points",
            "uncertainty_summary": f"95% HC3 CI [{_format_number(type_row['ci_lower'])}, {_format_number(type_row['ci_upper'])}]",
            "multiplicity_control": "not applicable to prespecified member-type term",
            "inference_status": "conditional_exploratory",
            "noncausal_interpretation": "Adjusted association does not establish a medication or coaching effect",
            "leadership_action": "Test the integrated treatment pathway prospectively",
            "randomized_pilot_population": "new eligible members for whom both pathways are clinically appropriate",
            "randomized_pilot_comparator": "usual coaching pathway",
            "randomized_pilot_kpi": "mean percentage weight loss",
        },
        {
            "hypothesis_order": 2,
            "hypothesis_id": "five_percent_response",
            "question": "What proportion reaches at least 5% weight loss in each retained condition?",
            "prespecified_expectation": "Response is descriptively higher in the GLP-1 weight-loss condition",
            "estimand": "condition-specific 5% response proportion",
            **common,
            "exposure_or_predictor": "member_type",
            "outcome": "weight_loss_success_5pct",
            "adjustment_set": "none; descriptive",
            "evidence_table": "nineam_outcomes_by_member_type.csv",
            "evidence_filter": "outcome=weight_loss_success_5pct",
            "result_status": "descriptive_estimate_available",
            "effect_summary": "; ".join(
                f"{scope}: {_format_number(responses.loc[scope, 'percentage'])}%"
                for scope in PRIMARY_MEMBER_TYPES
            ),
            "uncertainty_summary": "Wilson 95% intervals reported by condition",
            "multiplicity_control": "not applicable to descriptive response summaries",
            "inference_status": "descriptive",
            "noncausal_interpretation": "Unadjusted condition differences can reflect selection and confounding",
            "leadership_action": "Use response rate as a prospective operational KPI",
            "randomized_pilot_population": "new primary-condition-eligible members",
            "randomized_pilot_comparator": "usual coaching pathway",
            "randomized_pilot_kpi": "proportion with at least 5% weight loss",
        },
        {
            "hypothesis_order": 3,
            "hypothesis_id": "engagement_stability",
            "question": "Are engagement measures stable exploratory predictors after base adjustment?",
            "prespecified_expectation": "Greater repeatable engagement is associated with greater percentage loss",
            "estimand": "post-LASSO adjusted engagement coefficient",
            **common,
            "exposure_or_predictor": "engagement candidates",
            "outcome": "percentage_loss",
            "adjustment_set": "first_weight and member_type",
            "evidence_table": winning_lasso_filename,
            "evidence_filter": (
                f"module_spec={winning_module_spec}; candidate_role=engagement; "
                "eligible_for_locked_model=true"
            ),
            "result_status": "stable_candidate_selected" if len(engagement_locked) else "no_stable_candidate",
            "effect_summary": ", ".join(engagement_locked["candidate"].astype(str)) or "none selected",
            "uncertainty_summary": "selection frequency threshold 0.75; HC3 only after locking",
            "multiplicity_control": "stability threshold, not a p-value selection rule",
            "inference_status": "conditional_exploratory",
            "noncausal_interpretation": "Same-window engagement can reflect response or care intensity",
            "leadership_action": "Randomize an engagement-support intervention",
            "randomized_pilot_population": "members with low early repeatable engagement",
            "randomized_pilot_comparator": "usual engagement support",
            "randomized_pilot_kpi": "12-week percentage weight loss",
        },
        {
            "hypothesis_order": 4,
            "hypothesis_id": "module_stability",
            "question": "Are module-completion measures stable exploratory predictors?",
            "prespecified_expectation": "Greater completion is associated with greater percentage loss",
            "estimand": "post-LASSO adjusted module coefficient",
            **common,
            "exposure_or_predictor": "module mean or domain candidates",
            "outcome": "percentage_loss",
            "adjustment_set": "first_weight and member_type",
            "evidence_table": winning_lasso_filename,
            "evidence_filter": (
                f"module_spec={winning_module_spec}; candidate_role=module; "
                "eligible_for_locked_model=true"
            ),
            "result_status": "stable_candidate_selected" if len(module_locked) else "no_stable_candidate",
            "effect_summary": ", ".join(module_locked["candidate"].astype(str)) or "none selected",
            "uncertainty_summary": "Extension availability is not verifiable and Coaching support is zero",
            "multiplicity_control": "separate mean/domain specifications and stability threshold",
            "inference_status": "conditional_exploratory",
            "noncausal_interpretation": "Completion can proxy motivation, access, or treatment assignment",
            "leadership_action": "Randomize supported module encouragement",
            "randomized_pilot_population": "members with verified module access",
            "randomized_pilot_comparator": "usual module presentation",
            "randomized_pilot_kpi": "module completion and percentage weight loss",
        },
        {
            "hypothesis_order": 5,
            "hypothesis_id": "activity_family",
            "question": "Which reached activity intensities have adjusted exploratory associations?",
            "prespecified_expectation": "Actionable repeatable activities are positively associated with percentage loss",
            "estimand": "percentage points per SD log1p event count",
            **common,
            "exposure_or_predictor": "activity-specific standardized log1p count",
            "outcome": "percentage_loss",
            "adjustment_set": "first_weight, member_type, linear and quadratic weight_days",
            "evidence_table": "nineam_engagement_activity_summary.csv",
            "evidence_filter": "test_status=tested",
            "result_status": "fdr_signal_detected" if len(activity_signals) else "no_fdr_signal",
            "effect_summary": ", ".join(activity_signals["event_type"].astype(str)) or "no q<=0.05 activity",
            "uncertainty_summary": f"{int(activity['fdr_test_count'].max())} eligible activity tests",
            "multiplicity_control": "one Benjamini-Hochberg eligible-activity family",
            "inference_status": "exploratory",
            "noncausal_interpretation": "Activity timing overlaps follow-up and can reflect reverse causation",
            "leadership_action": "Pilot the most actionable signal rather than operationalizing association",
            "randomized_pilot_population": "members eligible for the selected activity",
            "randomized_pilot_comparator": "usual activity access",
            "randomized_pilot_kpi": "activity uptake and percentage weight loss",
        },
        {
            "hypothesis_order": 6,
            "hypothesis_id": "base_model_prediction",
            "question": "Which base family predicts held-out raw last weight more accurately?",
            "prespecified_expectation": "Lowest mean RMSE wins with MAE and primary-outcome tie breaks",
            "estimand": "paired held-out raw last-weight RMSE and MAE",
            **common,
            "exposure_or_predictor": "base model family",
            "outcome": "raw_last_weight",
            "adjustment_set": "shared member folds",
            "evidence_table": "nineam_base_model_comparison_summary.csv",
            "evidence_filter": "is_winner=true",
            "result_status": "predictive_winner_identified",
            "effect_summary": f"winner: {winner}",
            "uncertainty_summary": "fold-score distributions reported for RMSE and MAE",
            "multiplicity_control": "not applicable; prespecified ranking rule",
            "inference_status": "predictive_validation",
            "noncausal_interpretation": "Prediction ranking is not a treatment-effect estimate",
            "leadership_action": "Use the winning family for locked descriptive adjustment",
            "randomized_pilot_population": "not applicable to model-family comparison",
            "randomized_pilot_comparator": "not applicable",
            "randomized_pilot_kpi": "prospective raw last-weight RMSE",
        },
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_hypothesis_evidence.csv"],
    )


def _build_findings(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Translate evidence-register rows into deterministic leadership findings.

    Args:
        tables: Earlier reporting tables including the hypothesis register.

    Returns:
        Evidence-filtered, explicitly noncausal findings and pilot implications.

    Side effects:
        None.

    Statistical intent:
        Couples every leadership statement to a quantitative source artifact.
    """
    hypotheses = tables["nineam_hypothesis_evidence.csv"]
    rows = []
    certainty_by_status = {
        "descriptive": "descriptive",
        "predictive_validation": "predictive_validation",
        "conditional_exploratory": "conditional_exploratory",
        "exploratory": "exploratory",
    }
    for finding_order, hypothesis in enumerate(
        hypotheses.itertuples(index=False), start=1
    ):
        rows.append(
            {
                "finding_order": finding_order,
                "finding_id": f"finding_{hypothesis.hypothesis_id}",
                "finding": hypothesis.question,
                "quantitative_evidence": (
                    f"{hypothesis.effect_summary}; {hypothesis.uncertainty_summary}"
                ),
                "evidence_table": hypothesis.evidence_table,
                "evidence_filter": hypothesis.evidence_filter,
                "interpretation": hypothesis.noncausal_interpretation,
                "certainty": certainty_by_status.get(
                    hypothesis.inference_status, "exploratory"
                ),
                "causal_status": "noncausal_observational",
                "leadership_implication": hypothesis.leadership_action,
                "recommended_test": "prospective randomized pilot",
                "pilot_population": hypothesis.randomized_pilot_population,
                "pilot_intervention": hypothesis.leadership_action,
                "pilot_comparator": hypothesis.randomized_pilot_comparator,
                "pilot_primary_kpi": hypothesis.randomized_pilot_kpi,
                "pilot_time_horizon": hypothesis.randomized_pilot_time_horizon,
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_findings_and_implications.csv"],
    )


def _build_limitations(
    cohort: CohortResult,
    members: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Populate the complete prespecified limitation register.

    Args:
        cohort: Primary cohort audit counts.
        members: Primary member features.
        tables: Earlier reporting artifacts supplying empirical evidence.

    Returns:
        Ten ordered limitations with current mitigation and future design needs.

    Side effects:
        None.

    Statistical intent:
        Makes uncertainty sources inseparable from leadership interpretation.
    """
    enrolled = int(cohort.audit_counts["source_demographic_rows"])
    included = int(cohort.audit_counts["included_members"])
    pre_restriction = int(
        cohort.audit_counts["pre_member_type_restriction_members"]
    )
    nonprimary = int(cohort.audit_counts["excluded_nonprimary_member_type"])
    if pre_restriction != included + nonprimary:
        raise ValueError(
            "pre_member_type_restriction_members must equal included plus nonprimary"
        )
    days = pd.to_numeric(members["weight_days"], errors="raise")
    activity = tables["nineam_engagement_activity_summary.csv"]
    activity_tests = int(activity["fdr_test_count"].max())
    limitation_content = (
        (
            "attrition_complete_pair",
            "selection",
            "Outcome eligibility excludes members without usable paired follow-up.",
            (
                f"{enrolled - pre_restriction} of {enrolled} enrolled members "
                "lack a usable complete outcome pair."
            ),
            "primary outcomes and all adjusted associations",
            "Selection can make the retained cohort differ from all enrolled members.",
            "Sequential cohort flow preserves every exclusion count.",
            "Prospectively collect endpoint weights for all enrolled members.",
            "high",
        ),
        (
            "observational_residual_confounding",
            "causal inference",
            "Treatment condition and engagement were not randomized.",
            "The extract is observational and adjustment is limited to supplied covariates.",
            "member-type, engagement, module, and activity associations",
            "Unmeasured clinical need or motivation can explain observed contrasts.",
            "Association language and baseline adjustment are used throughout.",
            "Randomize the intervention or use a prespecified causal design.",
            "high",
        ),
        (
            "two_condition_restriction",
            "population",
            (
                "Primary analysis excludes Null, Active GLP-1 for Diabetes, and "
                "Active Generic Medication for Weight-loss conditions."
            ),
            f"{nonprimary} complete-pair members were removed by the member-type restriction.",
            "generalizability of all primary outputs",
            "Results do not represent the excluded medication or Null conditions.",
            "Excluded conditions remain visible in cohort accounting.",
            "Recruit adequate prespecified comparison groups prospectively.",
            "medium",
        ),
        (
            "same_window_reverse_causation",
            "temporality",
            "Engagement and module exposure accumulate during the outcome interval.",
            "Events and completions are censored at, not before, the last-weight date.",
            "engagement, module, and activity associations",
            "Weight-loss response can change subsequent engagement or care intensity.",
            "Post-outcome events are excluded and causal language is prohibited.",
            "Randomize an early exposure and measure a later outcome window.",
            "high",
        ),
        (
            "variable_follow_up",
            "measurement",
            "Members have unequal first-to-last weight intervals.",
            f"Observed weight_days range from {_format_number(days.min(), 0)} to {_format_number(days.max(), 0)} days.",
            "percentage-loss comparisons and engagement opportunity",
            "Longer observation can permit more loss and more recorded activity.",
            "A common-support linear/quadratic duration sensitivity is reported.",
            "Use a fixed follow-up visit window in a prospective study.",
            "high",
        ),
        (
            "age_absent",
            "missing covariate",
            "Age is absent from the supplied demographics extract.",
            f"age column present: {'yes' if 'age' in members.columns else 'no'}.",
            "adjusted member-type and exploratory associations",
            "Age-related outcome and treatment differences cannot be adjusted.",
            "The missing covariate is disclosed rather than imputed.",
            "Collect age prospectively and prespecify its functional form.",
            "high",
        ),
        (
            "post_selection_inference",
            "statistical inference",
            "HC3 intervals follow data-driven LASSO specification and stability selection.",
            "Locked coefficient inference is labeled conditional_exploratory.",
            "locked-model coefficient intervals and p-values",
            "Nominal intervals do not account for the selection process.",
            "Selection provenance and the 0.75 stability threshold are reported.",
            "Validate the locked equation in an untouched external sample.",
            "high",
        ),
        (
            "activity_multiple_testing",
            "multiplicity",
            "Multiple activity-specific associations are examined.",
            f"{activity_tests} reach-eligible activity types entered one tested family.",
            "activity association p-values",
            "Some nominal signals can occur by chance.",
            "One Benjamini-Hochberg correction covers all eligible types.",
            "Prespecify one activity intervention and primary KPI in a pilot.",
            "medium",
        ),
        (
            "extension_module_availability",
            "measurement support",
            "Extension-module availability cannot be verified from the extract.",
            "Coaching has zero observed mindset, nutrition, and physical-activity completions.",
            "module-domain descriptions and coefficients",
            "Zero completion can reflect no access rather than no uptake.",
            "Extension support is labeled GLP-1-only and noncomparable.",
            "Record module assignment, availability, and completion separately.",
            "high",
        ),
        (
            "external_validation_absent",
            "validation",
            "No independent cohort validates prediction or selected coefficients.",
            "All fits and stability resamples use the supplied case-study cohort.",
            "predictive ranking and locked-model generalization",
            "Performance and selected terms may not reproduce elsewhere.",
            "Grouped held-out comparison is limited to internal prediction.",
            "Freeze the analysis and evaluate it in a new cohort.",
            "high",
        ),
    )
    rows = []
    for order, content in enumerate(limitation_content, start=1):
        (
            limitation_id,
            category,
            limitation,
            evidence,
            affected,
            impact,
            mitigation,
            future,
            severity,
        ) = content
        rows.append(
            {
                "limitation_order": order,
                "limitation_id": limitation_id,
                "category": category,
                "limitation": limitation,
                "empirical_evidence": evidence,
                "affected_estimand": affected,
                "potential_impact": impact,
                "direction_of_bias": "indeterminate",
                "affected_outputs": affected,
                "mitigation_in_current_analysis": mitigation,
                "recommended_future_design": future,
                "severity": severity,
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=REPORTING_SCHEMAS["nineam_limitations.csv"],
    )


def _validate_primary_inputs(
    data: CaseStudyData,
    cohort: CohortResult,
    features: FeatureResult,
) -> pd.DataFrame:
    """Validate the common aggregate-reporting population and return its rows.

    Args:
        data: Canonical source wrapper retained for interface provenance.
        cohort: Primary two-condition cohort result.
        features: Features built from exactly that cohort.

    Returns:
        Stable member feature rows.

    Side effects:
        None.

    Statistical intent:
        Prevents descriptive and inferential tables from using different members.
    """
    if not isinstance(data, CaseStudyData):
        raise TypeError("data must be a CaseStudyData")
    if not isinstance(cohort, CohortResult):
        raise TypeError("cohort must be a CohortResult")
    if not isinstance(features, FeatureResult):
        raise TypeError("features must be a FeatureResult")
    members = features.member_features.copy()
    required = {
        "member_id",
        "member_type",
        "sex",
        "ethnicity",
        "first_weight",
        "last_weight",
        "absolute_weight_loss",
        "percentage_loss",
        "weight_loss_success_5pct",
        "weight_days",
        "tenure_days",
        "engagement_breadth",
        "engagement_volume_repeatable",
        "engagement_volume_repeatable_rate",
        "module_core_count",
        "module_mindset_count",
        "module_nutrition_count",
        "module_physical_activity_count",
        "module_core",
        "module_mindset",
        "module_nutrition",
        "module_physical_activity",
        "module_mean",
    }
    _require_columns(members, required, "Member features")
    _require_columns(cohort.members, {"member_id", "member_type"}, "Cohort members")
    if members["member_id"].duplicated().any() or cohort.members["member_id"].duplicated().any():
        raise ValueError("Reporting inputs must contain one row per member_id")
    if set(members["member_type"].astype(str)) != set(PRIMARY_MEMBER_TYPES):
        raise ValueError("Reporting requires exactly the two primary member types")
    if set(cohort.members["member_type"].astype(str)) != set(PRIMARY_MEMBER_TYPES):
        raise ValueError("Cohort contains nonprimary member types")
    if set(members["member_id"].astype(str)) != set(cohort.members["member_id"].astype(str)):
        raise ValueError("Cohort and feature member IDs do not match")
    feature_member_types = pd.Series(
        members["member_type"].astype(str).to_numpy(),
        index=members["member_id"].astype(str),
    ).sort_index()
    cohort_member_types = pd.Series(
        cohort.members["member_type"].astype(str).to_numpy(),
        index=cohort.members["member_id"].astype(str),
    ).sort_index()
    if not feature_member_types.equals(cohort_member_types):
        raise ValueError(
            "Cohort and feature member_id-to-member_type assignments do not match"
        )
    return members.sort_values("member_id", kind="stable").reset_index(drop=True)


def _validate_reporting_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    """Enforce every closed schema and the aggregate privacy boundary.

    Args:
        tables: Completed filename-keyed reporting table mapping.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Stops schema drift or row-level identifiers before serialization.
    """
    if tuple(tables) != tuple(REPORTING_SCHEMAS):
        raise ValueError("Reporting filenames do not match the approved contract")
    for filename, schema in REPORTING_SCHEMAS.items():
        table = tables[filename]
        if tuple(table.columns) != schema:
            raise ValueError(f"{filename} does not match its closed schema")
        if any("member_id" in str(column).casefold() for column in table.columns):
            raise ValueError(f"{filename} exposes a member identifier column")
        text = table.select_dtypes(include=["object", "string"])
        if not text.empty and text.astype("string").eq("<SUPPRESSED_RARE_LEVELS>").any(axis=None):
            raise ValueError(f"{filename} contains an unapproved suppression label")


def build_reporting_tables(
    data: CaseStudyData,
    cohort: CohortResult,
    features: FeatureResult,
    base_model_cv: pd.DataFrame,
    lasso_mean_selection: pd.DataFrame,
    lasso_domain_selection: pd.DataFrame,
    locked_model: LockedModelResult,
) -> ReportingResult:
    """Create all aggregate scientific tables used by leadership visuals.

    Args:
        data: Canonical case-study source wrapper.
        cohort: Confirmed primary two-condition cohort.
        features: Member features and sparse activity aggregates for that cohort.
        base_model_cv: Paired fold-level base-family prediction scores.
        lasso_mean_selection: Enriched mean-specification LASSO provenance rows.
        lasso_domain_selection: Enriched domain-specification LASSO rows.
        locked_model: Task 2 HC3 coefficients, fit statistics, and diagnostics.

    Returns:
        ``ReportingResult`` containing all fifteen exact closed-schema tables.

    Side effects:
        None; no files are written and input objects are not mutated.

    Statistical intent:
        Centralizes descriptive, exploratory, sensitivity, and interpretation
        artifacts while preserving their distinct inferential meanings.
    """
    if not isinstance(locked_model, LockedModelResult):
        raise TypeError("locked_model must be a LockedModelResult")
    members = _validate_primary_inputs(data, cohort, features)
    if int(cohort.audit_counts.get("included_members", -1)) != len(members):
        raise ValueError("Cohort included_members does not match member features")
    mean_fold_ids = set(lasso_mean_selection.get("fold_plan_id", pd.Series(dtype=str)).astype(str))
    domain_fold_ids = set(lasso_domain_selection.get("fold_plan_id", pd.Series(dtype=str)).astype(str))
    if mean_fold_ids and domain_fold_ids and mean_fold_ids != domain_fold_ids:
        raise ValueError("Mean and domain LASSO specifications require one paired fold plan")

    tables: dict[str, pd.DataFrame] = {}
    tables["nineam_cohort_flow.csv"] = _build_cohort_flow(cohort)
    tables["nineam_sample_characteristics.csv"] = _build_sample_characteristics(members)
    tables["nineam_outcomes_by_member_type.csv"] = _build_outcomes(members)
    tables["nineam_engagement_by_member_type.csv"] = _build_engagement(members)
    tables["nineam_engagement_activity_summary.csv"] = _build_activity_summary(features)
    tables["nineam_modules_by_member_type.csv"] = _build_modules(members)
    tables["nineam_base_model_comparison_summary.csv"] = _build_base_model_summary(
        base_model_cv, locked_model
    )
    tables["nineam_lasso_mean_selection.csv"] = _build_lasso_selection(
        lasso_mean_selection, "mean", locked_model
    )
    tables["nineam_lasso_domain_selection.csv"] = _build_lasso_selection(
        lasso_domain_selection, "domains", locked_model
    )
    tables["nineam_locked_model_coefficients_hc3.csv"] = _ordered_locked_table(
        locked_model.coefficient_table,
        REPORTING_SCHEMAS["nineam_locked_model_coefficients_hc3.csv"],
        "Locked coefficient table",
        "term_order",
    )
    tables["nineam_locked_model_fit_statistics.csv"] = _ordered_locked_table(
        locked_model.fit_statistics,
        REPORTING_SCHEMAS["nineam_locked_model_fit_statistics.csv"],
        "Locked fit-statistics table",
        None,
    )
    tables["nineam_model_diagnostics.csv"] = _build_diagnostics(locked_model)
    tables["nineam_hypothesis_evidence.csv"] = _build_hypotheses(tables)
    tables["nineam_findings_and_implications.csv"] = _build_findings(tables)
    tables["nineam_limitations.csv"] = _build_limitations(cohort, members, tables)
    _validate_reporting_tables(tables)
    return ReportingResult(tables=tables)


__all__ = [
    "ACTIONABILITY_CLASSES",
    "LIMITATION_IDS",
    "REPORTING_SCHEMAS",
    "ReportingResult",
    "build_reporting_tables",
    "wilson_confidence_interval",
]
