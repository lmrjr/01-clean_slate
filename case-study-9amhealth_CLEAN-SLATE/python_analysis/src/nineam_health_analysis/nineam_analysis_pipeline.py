"""Orchestrate the reproducible aggregate-only 9amHealth analysis.

The pipeline reads the four supplied extracts without modifying them, builds
the confirmed cohort and features, fits the requested base and sensitivity
models, performs grouped predictive comparison, and runs separate module-mean
and module-domain percentage-LASSO analyses. Longitudinal penalization is kept
as retrospective two-stage exploration with fixed prewhitening; it is never
presented as unbiased nested-CV performance or statistical significance.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from nineam_health_analysis.nineam_cohort_selection import (
    REFERENCE_MEMBER_TYPE,
    CohortResult,
    build_analysis_cohort,
    restrict_to_primary_member_types,
)
from nineam_health_analysis.nineam_data_loading import (
    CaseStudyData,
    load_case_study_data,
)
from nineam_health_analysis.nineam_feature_engineering import (
    FeatureResult,
    build_member_features,
)
from nineam_health_analysis.nineam_final_model import (
    LockedModelResult,
    choose_base_model_winner,
    fit_locked_percentage_model,
    select_locked_candidates,
)
from nineam_health_analysis.nineam_penalized_models import (
    diagnose_collinearity,
    fit_partially_penalized_lasso,
)
from nineam_health_analysis.nineam_reporting import (
    REPORTING_SCHEMAS,
    build_reporting_tables,
)
from nineam_health_analysis.nineam_resampling import (
    PenalizedDesign,
    build_longitudinal_penalized_design,
    build_percentage_penalized_design,
    compare_base_models,
    select_lambda_by_grouped_cv,
    stability_select_lasso,
)
from nineam_health_analysis.nineam_statistical_models import (
    LongitudinalGLSResult,
    PercentageLossOLSResult,
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
)
from nineam_health_analysis.nineam_visualizations import (
    FIGURE_STEMS,
    write_analysis_figures,
)

_DEFAULT_LAMBDA_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01)
_LONGITUDINAL_EXPLORATORY_LAMBDA_RATIO = 0.2
_SELECTION_INTERPRETATION = "exploratory_stability_not_significance"
_LONGITUDINAL_INTERPRETATION = (
    "retrospective_two_stage_fixed_prewhitening_not_performance"
)
_MINIMUM_REPORTING_CELL_SIZE = 10
_SUPPRESSED_RARE_LEVELS_LABEL = "<SUPPRESSED_RARE_LEVELS>"
_TABLE_RELATIVE_PATHS = tuple(
    Path("tables") / filename for filename in REPORTING_SCHEMAS
)
_FIGURE_RELATIVE_PATHS = tuple(
    Path("figures") / f"{stem}.{extension}"
    for stem in FIGURE_STEMS
    for extension in ("png", "svg")
)
_METADATA_RELATIVE_PATH = Path("nineam_analysis_metadata.json")
_OUTPUT_RELATIVE_PATHS = (
    *_TABLE_RELATIVE_PATHS,
    *_FIGURE_RELATIVE_PATHS,
    _METADATA_RELATIVE_PATH,
)
_DESCRIPTIVE_FEATURES = (
    "first_weight",
    "last_weight",
    "percentage_loss",
    "engagement_volume_repeatable",
    "engagement_volume_repeatable_rate",
    "engagement_breadth",
    "tenure_days",
    "module_mean",
    "module_core",
    "module_mindset",
    "module_nutrition",
    "module_physical_activity",
)
_LASSO_CANDIDATE_ORDER = {
    candidate: order
    for order, candidate in enumerate(
        (
            "engagement_volume_repeatable",
            "engagement_volume_repeatable_rate",
            "engagement_breadth",
            "tenure_days",
            "module_mean",
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
            "sex[MALE]",
        ),
        start=1,
    )
}
_CSV_OUTPUT_SCHEMAS = REPORTING_SCHEMAS
_METADATA_KEYS = {
    "analysis_schema_version",
    "seed",
    "cv_folds",
    "cv_repeats",
    "stability_resamples",
    "lambda_ratios",
    "stability_subsample_fraction",
    "stability_selection_threshold",
    "complete_pair_eligible_members",
    "excluded_nonprimary_members",
    "included_members",
    "reference_member_type",
    "responder_threshold_percentage",
    "source_row_counts",
    "base_model_comparison_target",
    "base_model_comparison_metrics",
    "base_model_winner_rule",
    "winning_base_model",
    "cross_family_likelihood_comparison",
    "locked_covariance_estimator",
    "locked_inference_status",
    "selection_interpretation",
    "longitudinal_lasso_interpretation",
    "longitudinal_exploratory_lambda_ratio",
    "lasso_specification_tie_tolerance",
    "lasso_fold_plan_id",
    "lasso_fold_plan",
    "output_files",
}


def _positive_integer(value: object, label: str, *, minimum: int = 1) -> int:
    """Validate one integer control against a positive lower boundary.

    Args:
        value: Candidate resampling or iteration count.
        label: Human-readable name used in validation messages.
        minimum: Smallest permitted integer value.

    Returns:
        The validated Python integer.

    Side effects:
        None.

    Statistical intent:
        Prevents degenerate fold, repeat, and stability-selection designs.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    validated = int(value)
    if validated < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return validated


def _probability(value: object, label: str, *, include_one: bool) -> float:
    """Validate a finite probability-like pipeline control.

    Args:
        value: Candidate subsampling fraction or selection threshold.
        label: Human-readable name used in validation messages.
        include_one: Whether the upper boundary of one is permitted.

    Returns:
        A validated floating-point value.

    Side effects:
        None.

    Statistical intent:
        Keeps resample fractions and empirical selection thresholds meaningful.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    validated = float(value)
    upper_valid = validated <= 1.0 if include_one else validated < 1.0
    if not np.isfinite(validated) or validated <= 0.0 or not upper_valid:
        boundary = "(0, 1]" if include_one else "(0, 1)"
        raise ValueError(f"{label} must lie in {boundary}")
    return validated


def _lambda_grid(values: object) -> tuple[float, ...]:
    """Validate and freeze unique lambda-max fractions for grouped tuning.

    Args:
        values: Sequence of numeric penalty fractions.

    Returns:
        A tuple of unique finite ratios in the interval ``(0, 1]``.

    Side effects:
        None.

    Statistical intent:
        Makes every tuning column identifiable and bounded by fold-local
        lambda-max.
    """
    if not isinstance(values, (tuple, list)):
        raise TypeError("lambda_ratios must be a tuple or list")
    try:
        ratios = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError("lambda_ratios must contain numeric values") from error
    if not ratios:
        raise ValueError("lambda_ratios cannot be empty")
    if any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0
        for value in ratios
    ):
        raise ValueError("lambda_ratios must lie in (0, 1]")
    if len(set(ratios)) != len(ratios):
        raise ValueError("lambda_ratios must be unique")
    return ratios


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Store deterministic modeling and member-resampling controls."""

    seed: int = 20260901
    cv_folds: int = 5
    cv_repeats: int = 2
    stability_resamples: int = 100
    lambda_ratios: tuple[float, ...] = _DEFAULT_LAMBDA_RATIOS
    stability_subsample_fraction: float = 0.7
    stability_selection_threshold: float = 0.75

    def __post_init__(self) -> None:
        """Validate every control before source loading or model fitting.

        Args:
            self: Newly constructed configuration.

        Returns:
            None.

        Side effects:
            Replaces fields on the frozen instance with normalized scalar values.

        Statistical intent:
            Fixes all random and tuning choices required for reproducible grouped
            validation and exploratory stability selection.
        """
        seed = _positive_integer(self.seed, "seed", minimum=0)
        folds = _positive_integer(self.cv_folds, "cv_folds", minimum=2)
        repeats = _positive_integer(self.cv_repeats, "cv_repeats")
        resamples = _positive_integer(
            self.stability_resamples,
            "stability_resamples",
        )
        ratios = _lambda_grid(self.lambda_ratios)
        fraction = _probability(
            self.stability_subsample_fraction,
            "stability_subsample_fraction",
            include_one=False,
        )
        threshold = _probability(
            self.stability_selection_threshold,
            "stability_selection_threshold",
            include_one=True,
        )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "cv_folds", folds)
        object.__setattr__(self, "cv_repeats", repeats)
        object.__setattr__(self, "stability_resamples", resamples)
        object.__setattr__(self, "lambda_ratios", ratios)
        object.__setattr__(self, "stability_subsample_fraction", fraction)
        object.__setattr__(self, "stability_selection_threshold", threshold)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Store aggregate analysis tables and deterministic run metadata."""

    cohort_summary: pd.DataFrame
    feature_summary: pd.DataFrame
    model_summary: pd.DataFrame
    base_model_cv: pd.DataFrame
    lasso_mean_selection: pd.DataFrame
    lasso_domain_selection: pd.DataFrame
    diagnostics: pd.DataFrame
    locked_model: LockedModelResult
    reporting_tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, object]
    source_data_directory: Path


def _cohort_summary(cohort: CohortResult) -> pd.DataFrame:
    """Convert mutually exclusive cohort audit counts to a stable table.

    Args:
        cohort: Completed cohort result with aggregate attrition counts.

    Returns:
        A two-column dataframe sorted by audit metric name.

    Side effects:
        None.

    Statistical intent:
        Exposes the analysis denominator and every eligibility exclusion without
        releasing member-level rows.
    """
    records = [
        {"metric": metric, "count": int(count)}
        for metric, count in sorted(cohort.audit_counts.items())
    ]
    return pd.DataFrame.from_records(records, columns=["metric", "count"])


def _feature_summary(features: FeatureResult) -> pd.DataFrame:
    """Summarize feature audits and univariate distributions in long form.

    Args:
        features: Engineered member features and aggregate feature audit counts.

    Returns:
        Aggregate audit and descriptive rows with no member identifiers.

    Side effects:
        None.

    Statistical intent:
        Makes feature construction and scale visible before modeling while
        avoiding row-level disclosure or inferential claims.
    """
    records: list[dict[str, str | float]] = []
    for variable, count in sorted(features.audit_counts.items()):
        records.append(
            {
                "category": "feature_audit",
                "variable": variable,
                "statistic": "count",
                "value": float(count),
            }
        )

    for variable in _DESCRIPTIVE_FEATURES:
        values = pd.to_numeric(
            features.member_features[variable],
            errors="raise",
        )
        observed = values.dropna().to_numpy(dtype=float)
        statistics = (
            ("count", float(observed.size)),
            ("missing_count", float(values.isna().sum())),
            ("mean", float(np.mean(observed))),
            ("standard_deviation", float(np.std(observed, ddof=1))),
            ("minimum", float(np.min(observed))),
            ("median", float(np.median(observed))),
            ("maximum", float(np.max(observed))),
        )
        for statistic, value in statistics:
            records.append(
                {
                    "category": "descriptive",
                    "variable": variable,
                    "statistic": statistic,
                    "value": value,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["category", "variable", "statistic", "value"],
    )


def _coefficient_json(
    names: tuple[str, ...],
    values: np.ndarray,
) -> str:
    """Serialize named aggregate coefficient values deterministically.

    Args:
        names: Coefficient names in fitted order.
        values: Numeric estimates or standard errors aligned with ``names``.

    Returns:
        Compact JSON preserving fitted coefficient order.

    Side effects:
        None.

    Statistical intent:
        Retains model estimates in one aggregate summary row without exposing
        individual fitted values or residuals.
    """
    mapping = {
        name: float(value) for name, value in zip(names, values, strict=True)
    }
    return json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))


def _model_summary(
    raw_un: LongitudinalGLSResult,
    log_cs: LongitudinalGLSResult,
    log_cs_interaction: LongitudinalGLSResult,
    percentage: PercentageLossOLSResult,
) -> pd.DataFrame:
    """Create one aggregate fit-summary row for each requested model.

    Args:
        raw_un: Raw-weight additive unstructured-covariance GLS fit.
        log_cs: Log-weight additive compound-symmetry GLS fit.
        log_cs_interaction: Log-CS time-by-member-type sensitivity fit.
        percentage: Percentage-loss OLS fit.

    Returns:
        Stable model rows containing coefficients and within-family fit indices.

    Side effects:
        None.

    Statistical intent:
        Reports each model without using AIC or negative-two-log-likelihood to
        declare a winner across different outcome families.
    """
    records: list[dict[str, object]] = []
    longitudinal_models = (
        ("raw_un_additive_gls", "base", raw_un),
        ("log_cs_additive_gls", "base", log_cs),
        (
            "log_cs_time_by_member_type_sensitivity",
            "sensitivity",
            log_cs_interaction,
        ),
    )
    for model_name, role, result in longitudinal_models:
        records.append(
            {
                "model": model_name,
                "model_family": "longitudinal_marginal_gls",
                "model_role": role,
                "outcome_scale": result.outcome_scale,
                "covariance_structure": result.covariance_structure,
                "include_time_by_member_type": (
                    result.include_time_by_member_type
                ),
                "n_members": result.n_members,
                "n_observations": result.n_observations,
                "parameter_count": result.parameter_count,
                "negative_two_log_likelihood": (
                    result.negative_two_log_likelihood
                ),
                "aic": result.aic,
                "bic": result.bic,
                "r_squared": np.nan,
                "adjusted_r_squared": np.nan,
                "training_rmse": np.nan,
                "coefficients": _coefficient_json(
                    result.coefficient_names,
                    result.coefficients,
                ),
                "standard_errors": _coefficient_json(
                    result.coefficient_names,
                    result.standard_errors,
                ),
                "likelihood_scope": "longitudinal_weight_family_only",
            }
        )
    records.append(
        {
            "model": "percentage_loss_ols",
            "model_family": "ordinary_least_squares",
            "model_role": "base",
            "outcome_scale": "percentage_loss",
            "covariance_structure": "independent",
            "include_time_by_member_type": False,
            "n_members": percentage.n_members,
            "n_observations": percentage.n_members,
            "parameter_count": percentage.parameter_count,
            "negative_two_log_likelihood": (
                percentage.negative_two_log_likelihood
            ),
            "aic": percentage.aic,
            "bic": percentage.bic,
            "r_squared": percentage.r_squared,
            "adjusted_r_squared": percentage.adjusted_r_squared,
            "training_rmse": percentage.rmse,
            "coefficients": _coefficient_json(
                percentage.coefficient_names,
                percentage.coefficients,
            ),
            "standard_errors": _coefficient_json(
                percentage.coefficient_names,
                percentage.standard_errors,
            ),
            "likelihood_scope": "percentage_loss_family_only",
        }
    )
    return pd.DataFrame.from_records(records)


def _percentage_lasso_summary(
    design: PenalizedDesign,
    module_spec: str,
    config: AnalysisConfig,
    *,
    stability_seed_offset: int,
) -> pd.DataFrame:
    """Tune and summarize one percentage-loss LASSO candidate specification.

    Args:
        design: Percentage-loss penalized design for mean or domain modules.
        module_spec: Literal module specification label for output rows.
        config: Grouped-CV and stability-selection controls.
        stability_seed_offset: Deterministic offset separating the stability-
            sampling stream from the shared paired-CV fold plan.

    Returns:
        One aggregate row per candidate with tuning, full-fit, and stability
        information.

    Side effects:
        None; all fits occur in memory.

    Statistical intent:
        Uses grouped inner CV only to choose shrinkage, then reports member-
        subsample selection consistency as exploration rather than significance.
    """
    tuning = select_lambda_by_grouped_cv(
        design,
        config.lambda_ratios,
        n_splits=config.cv_folds,
        n_repeats=config.cv_repeats,
        # Both module specifications use the same member-fold seed so their CV
        # MSEs are paired; only stability subsampling receives a spec-specific
        # stream below.
        random_state=config.seed + 1_000,
        selection_rule="one_standard_error",
    )
    stability = stability_select_lasso(
        design,
        lambda_ratio=tuning.selected_lambda_ratio,
        n_resamples=config.stability_resamples,
        subsample_fraction=config.stability_subsample_fraction,
        selection_threshold=config.stability_selection_threshold,
        random_state=config.seed + stability_seed_offset,
    )

    # The full-sample equation is descriptive. Its coefficients do not inherit
    # uncertainty claims from LASSO tuning or stability frequencies.
    fitted = fit_partially_penalized_lasso(
        design.outcome,
        design.base_design,
        design.candidate_design,
        design.base_names,
        design.candidate_names,
        tuning.selected_lambda_ratio,
    )
    selected_score = float(tuning.mean_scores[tuning.selected_index])
    selected_score_se = float(tuning.standard_errors[tuning.selected_index])
    excluded = set(fitted.excluded_candidate_names)
    additional_candidates = sorted(
        set(design.candidate_names).difference(_LASSO_CANDIDATE_ORDER)
    )
    candidate_order = dict(_LASSO_CANDIDATE_ORDER)
    candidate_order.update(
        {
            candidate: len(_LASSO_CANDIDATE_ORDER) + offset
            for offset, candidate in enumerate(additional_candidates, start=1)
        }
    )
    fold_plan_id = (
        f"grouped_member_cv_seed_{config.seed + 1_000}_"
        f"r{config.cv_repeats}_k{config.cv_folds}"
    )
    records = []
    for index, candidate in enumerate(design.candidate_names):
        records.append(
            {
                "module_spec": module_spec,
                "candidate_order": candidate_order[candidate],
                "candidate": candidate,
                "lambda_ratio": tuning.selected_lambda_ratio,
                "full_sample_lambda": fitted.lambda_value,
                "full_sample_lambda_max": fitted.lambda_max,
                "cv_selection_rule": tuning.selection_rule,
                "cv_mean_mse": selected_score,
                "cv_standard_error": selected_score_se,
                "fold_plan_id": fold_plan_id,
                "n_resamples": config.stability_resamples,
                "subsample_fraction": config.stability_subsample_fraction,
                "full_sample_coefficient": (
                    fitted.candidate_coefficients[index]
                ),
                "full_sample_standardized_coefficient": (
                    fitted.standardized_candidate_coefficients[index]
                ),
                "selection_count": int(
                    np.count_nonzero(stability.selected_matrix[:, index])
                ),
                "selection_frequency": stability.selection_frequencies[index],
                "selected_at_threshold": (
                    candidate in stability.selected_candidate_names
                    and candidate not in excluded
                ),
                "excluded_from_full_fit": candidate in excluded,
                "selection_threshold": config.stability_selection_threshold,
                "interpretation": _SELECTION_INTERPRETATION,
            }
        )
    return pd.DataFrame.from_records(records)


def _missingness_records(
    data: CaseStudyData,
    member_features: pd.DataFrame,
) -> list[dict[str, object]]:
    """Calculate aggregate missing counts and rates for every analysis table.

    Args:
        data: Canonical source tables loaded read-only.
        member_features: Final engineered member-level analysis features.

    Returns:
        Long-form aggregate diagnostic records.

    Side effects:
        None.

    Statistical intent:
        Makes source and analysis-variable missingness visible without returning
        row identities or silently changing the eligible cohort.
    """
    tables = (
        ("source_modules", data.module_completions),
        ("source_demographics", data.demographics),
        ("source_engagement", data.engagement),
        ("source_body_weights", data.body_weights),
        ("analysis_features", member_features),
    )
    records: list[dict[str, object]] = []
    for scope, table in tables:
        denominator = len(table)
        for variable in table.columns:
            missing_count = int(table[variable].isna().sum())
            missing_rate = (
                float(missing_count / denominator) if denominator else np.nan
            )
            records.append(
                {
                    "diagnostic_type": "missingness",
                    "scope": scope,
                    "variable": variable,
                    "subgroup": "",
                    "count": missing_count,
                    "value": missing_rate,
                    "detail": "missing_rate",
                }
            )
    return records


def _subgroup_records(
    member_features: pd.DataFrame,
) -> list[dict[str, object]]:
    """Report subgroup sizes after combining cells smaller than ten members.

    Args:
        member_features: Final one-row-per-member feature table.

    Returns:
        Aggregate member-type, sex, and ethnicity count rows with rare levels
        combined under one non-identifying label per variable.

    Side effects:
        None.

    Statistical intent:
        Describes representation only. It never reports subgroup outcomes, and
        the conservative cell-size rule prevents disclosure of rare labels.
    """
    records: list[dict[str, object]] = []
    for factor in ("member_type", "sex", "ethnicity"):
        counts = member_features[factor].value_counts(dropna=False, sort=False)
        reportable_rows: list[tuple[str, int]] = []
        suppressed_count = 0
        for level, count in counts.items():
            level_label = "<MISSING>" if pd.isna(level) else str(level)
            if int(count) < _MINIMUM_REPORTING_CELL_SIZE:
                suppressed_count += int(count)
            else:
                reportable_rows.append((level_label, int(count)))

        # Sorting reportable labels makes output independent of source row order;
        # every rare label contributes only to one combined disclosure-safe cell.
        reportable_rows.sort(key=lambda item: item[0])
        if suppressed_count:
            reportable_rows.append(
                (_SUPPRESSED_RARE_LEVELS_LABEL, suppressed_count)
            )
        for level_label, count in reportable_rows:
            records.append(
                {
                    "diagnostic_type": "subgroup",
                    "scope": factor,
                    "variable": "member_count",
                    "subgroup": level_label,
                    "count": count,
                    "value": np.nan,
                    "detail": "member_count",
                }
            )
    return records


def _collinearity_records(
    mean_design: PenalizedDesign,
    domain_design: PenalizedDesign,
) -> list[dict[str, object]]:
    """Diagnose separate designs and the known combined module dependency.

    Args:
        mean_design: Percentage candidates containing module mean only.
        domain_design: Percentage candidates containing four module domains.

    Returns:
        Aggregate rank and null-space diagnostic records.

    Side effects:
        None.

    Statistical intent:
        Verifies each fitted specification separately and demonstrates why
        module mean and its four defining domains must not enter one LASSO run.
    """
    records: list[dict[str, object]] = []
    separate_designs = (
        ("module_mean_spec", mean_design),
        ("module_domain_spec", domain_design),
    )
    for scope, design in separate_designs:
        diagnostic = diagnose_collinearity(
            design.candidate_design,
            design.candidate_names,
        )
        for metric, value in (
            ("rank", diagnostic.rank),
            ("n_columns", diagnostic.n_columns),
            ("is_rank_deficient", int(diagnostic.is_rank_deficient)),
        ):
            records.append(
                {
                    "diagnostic_type": "collinearity",
                    "scope": scope,
                    "variable": "candidate_design",
                    "subgroup": "",
                    "count": np.nan,
                    "value": float(value),
                    "detail": metric,
                }
            )

    # Combine one copy of shared engagement/sex columns with module mean and all
    # four domains solely for diagnosis; this matrix is never supplied to LASSO.
    mean_names = mean_design.candidate_names
    domain_names = domain_design.candidate_names
    module_mean_index = mean_names.index("module_mean")
    domain_indices = [
        domain_names.index(name)
        for name in (
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
        )
    ]
    combined_names = (
        *mean_names[: module_mean_index + 1],
        *(domain_names[index] for index in domain_indices),
        *mean_names[module_mean_index + 1 :],
    )
    combined_design = np.column_stack(
        [
            mean_design.candidate_design[:, : module_mean_index + 1],
            domain_design.candidate_design[:, domain_indices],
            mean_design.candidate_design[:, module_mean_index + 1 :],
        ]
    )
    combined = diagnose_collinearity(combined_design, combined_names)
    for metric, value in (
        ("rank", combined.rank),
        ("n_columns", combined.n_columns),
        ("is_rank_deficient", int(combined.is_rank_deficient)),
    ):
        records.append(
            {
                "diagnostic_type": "collinearity",
                "scope": "combined_module_mean_and_domains_diagnostic_only",
                "variable": "candidate_design",
                "subgroup": "",
                "count": np.nan,
                "value": float(value),
                "detail": metric,
            }
        )
    for relation_index, relation in enumerate(combined.null_space_vectors, 1):
        canonical_relation = np.array(relation, dtype=float, copy=True)
        nonzero_indices = np.flatnonzero(np.abs(canonical_relation) > 1e-10)
        # Singular vectors are defined only up to sign. Orienting the first
        # material loading positively stabilizes diagnostics across backends.
        if nonzero_indices.size and canonical_relation[nonzero_indices[0]] < 0.0:
            canonical_relation *= -1.0
        for name, coefficient in zip(
            combined_names,
            canonical_relation,
            strict=True,
        ):
            if abs(coefficient) > 1e-10:
                records.append(
                    {
                        "diagnostic_type": "collinearity",
                        "scope": (
                            "combined_module_mean_and_domains_diagnostic_only"
                        ),
                        "variable": name,
                        "subgroup": "",
                        "count": np.nan,
                        "value": float(coefficient),
                        "detail": f"null_relation_{relation_index}",
                    }
                )
    return records


def _longitudinal_lasso_records(
    member_features: pd.DataFrame,
    log_cs: LongitudinalGLSResult,
) -> list[dict[str, object]]:
    """Run the fixed-penalty retrospective longitudinal LASSO code path.

    Args:
        member_features: Same members used for the supplied longitudinal fit.
        log_cs: Full-sample additive log-CS model supplying fixed prewhitening.

    Returns:
        Aggregate exploratory coefficient diagnostic rows.

    Side effects:
        None.

    Statistical intent:
        Exercises hierarchical time-by-feature penalization with fixed full-
        sample covariance. It performs no inner covariance refit and therefore
        makes no unbiased CV-performance or significance claim.
    """
    design = build_longitudinal_penalized_design(
        member_features,
        log_cs,
        module_spec="mean",
    )
    fixed_count = len(log_cs.coefficient_names)
    selected_base_indices = list(range(fixed_count))
    retained_candidate_indices: list[int] = []
    current_rank = int(
        np.linalg.matrix_rank(design.base_design[:, selected_base_indices])
    )

    # Candidate main effects are required for hierarchy, but a constant or
    # aliased feature can make the unpenalized base singular in small samples.
    # Retain a feature and its matched time interaction only when its main
    # effect increases rank; excluded terms are reported explicitly below.
    for candidate_index in range(len(design.candidate_names)):
        base_index = fixed_count + candidate_index
        proposed_indices = [*selected_base_indices, base_index]
        proposed_rank = int(
            np.linalg.matrix_rank(design.base_design[:, proposed_indices])
        )
        if proposed_rank > current_rank:
            selected_base_indices.append(base_index)
            retained_candidate_indices.append(candidate_index)
            current_rank = proposed_rank
    if not retained_candidate_indices:
        raise ValueError(
            "Longitudinal exploratory LASSO has no estimable candidates"
        )
    reduced_design = PenalizedDesign(
        outcome=design.outcome,
        base_design=design.base_design[:, selected_base_indices],
        candidate_design=design.candidate_design[:, retained_candidate_indices],
        base_names=tuple(design.base_names[index] for index in selected_base_indices),
        candidate_names=tuple(
            design.candidate_names[index]
            for index in retained_candidate_indices
        ),
        group_ids=design.group_ids,
        strata=design.strata,
    )
    fitted = fit_partially_penalized_lasso(
        reduced_design.outcome,
        reduced_design.base_design,
        reduced_design.candidate_design,
        reduced_design.base_names,
        reduced_design.candidate_names,
        _LONGITUDINAL_EXPLORATORY_LAMBDA_RATIO,
    )
    records = []
    for index, candidate in enumerate(design.candidate_names):
        if index in retained_candidate_indices:
            fitted_index = retained_candidate_indices.index(index)
            coefficient = float(fitted.candidate_coefficients[fitted_index])
            is_selected = int(
                abs(fitted.standardized_candidate_coefficients[fitted_index])
                > 1e-12
            )
        else:
            coefficient = 0.0
            is_selected = 0
        records.append(
            {
                "diagnostic_type": "longitudinal_lasso_exploratory",
                "scope": "log_cs_module_mean_fixed_prewhitening",
                "variable": candidate,
                "subgroup": "",
                "count": is_selected,
                "value": coefficient,
                "detail": _LONGITUDINAL_INTERPRETATION,
            }
        )
    return records


def _diagnostics(
    data: CaseStudyData,
    member_features: pd.DataFrame,
    mean_design: PenalizedDesign,
    domain_design: PenalizedDesign,
    log_cs: LongitudinalGLSResult,
) -> pd.DataFrame:
    """Combine missingness, subgroup, collinearity, and exploratory diagnostics.

    Args:
        data: Canonical read-only source tables.
        member_features: Final one-row-per-member feature table.
        mean_design: Percentage-LASSO module-mean design.
        domain_design: Percentage-LASSO module-domain design.
        log_cs: Additive log-CS fit for retrospective longitudinal whitening.

    Returns:
        One deterministic aggregate diagnostic dataframe.

    Side effects:
        None.

    Statistical intent:
        Centralizes analysis-quality checks without mixing descriptive,
        collinearity, selection, or predictive interpretations.
    """
    records = [
        *_missingness_records(data, member_features),
        *_subgroup_records(member_features),
        *_collinearity_records(mean_design, domain_design),
        *_longitudinal_lasso_records(member_features, log_cs),
    ]
    return pd.DataFrame.from_records(
        records,
        columns=[
            "diagnostic_type",
            "scope",
            "variable",
            "subgroup",
            "count",
            "value",
            "detail",
        ],
    )


def _metadata(
    data: CaseStudyData,
    complete_pair_cohort: CohortResult,
    cohort: CohortResult,
    config: AnalysisConfig,
    mean_selection: pd.DataFrame,
    domain_selection: pd.DataFrame,
    locked_model: LockedModelResult,
) -> dict[str, object]:
    """Build deterministic run metadata without paths, timestamps, or row data.

    Args:
        data: Canonical source tables used for aggregate row counts.
        complete_pair_cohort: Eligibility result before condition restriction.
        cohort: Final primary cohort used for included-member count.
        config: Complete deterministic analysis configuration.
        mean_selection: Mean-specification LASSO provenance rows.
        domain_selection: Domain-specification LASSO provenance rows.
        locked_model: Selected percentage-loss HC3 result.

    Returns:
        A JSON-serializable aggregate metadata dictionary.

    Side effects:
        None.

    Statistical intent:
        Records reproducibility controls and interpretation boundaries without
        embedding machine-specific paths or member information.
    """
    fold_ids = {
        *mean_selection["fold_plan_id"].astype(str).unique(),
        *domain_selection["fold_plan_id"].astype(str).unique(),
    }
    if len(fold_ids) != 1:
        raise ValueError(
            "Mean and domain LASSO summaries must share one fold_plan_id"
        )
    complete_pair_members = int(len(complete_pair_cohort.members))
    if complete_pair_members != int(
        cohort.audit_counts["pre_member_type_restriction_members"]
    ):
        raise ValueError("Complete-pair cohort does not reconcile to primary audit")
    return {
        "analysis_schema_version": "2.0.0",
        "seed": config.seed,
        "cv_folds": config.cv_folds,
        "cv_repeats": config.cv_repeats,
        "stability_resamples": config.stability_resamples,
        "lambda_ratios": list(config.lambda_ratios),
        "stability_subsample_fraction": (
            config.stability_subsample_fraction
        ),
        "stability_selection_threshold": (
            config.stability_selection_threshold
        ),
        "complete_pair_eligible_members": complete_pair_members,
        "excluded_nonprimary_members": int(
            cohort.audit_counts["excluded_nonprimary_member_type"]
        ),
        "included_members": int(len(cohort.members)),
        "reference_member_type": REFERENCE_MEMBER_TYPE,
        "responder_threshold_percentage": 5.0,
        "source_row_counts": {
            "body_weights": int(len(data.body_weights)),
            "demographics": int(len(data.demographics)),
            "engagement": int(len(data.engagement)),
            "module_completions": int(len(data.module_completions)),
        },
        "base_model_comparison_target": "raw_last_weight",
        "base_model_comparison_metrics": ["rmse", "mae"],
        "base_model_winner_rule": (
            "lowest_mean_rmse_then_lowest_mean_mae_then_"
            "percentage_loss_ols_on_exact_tie"
        ),
        "winning_base_model": locked_model.winning_base_model,
        "cross_family_likelihood_comparison": False,
        "locked_covariance_estimator": "HC3",
        "locked_inference_status": "conditional_exploratory",
        "selection_interpretation": _SELECTION_INTERPRETATION,
        "longitudinal_lasso_interpretation": _LONGITUDINAL_INTERPRETATION,
        "longitudinal_exploratory_lambda_ratio": (
            _LONGITUDINAL_EXPLORATORY_LAMBDA_RATIO
        ),
        "lasso_specification_tie_tolerance": 1e-12,
        "lasso_fold_plan_id": next(iter(fold_ids)),
        "lasso_fold_plan": (
            "paired_grouped_member_folds_shared_across_mean_and_domains"
        ),
        "output_files": sorted(
            path.as_posix() for path in _OUTPUT_RELATIVE_PATHS
        ),
    }


def run_analysis(
    data_dir: str | Path,
    *,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Run every confirmed model path and return aggregate results in memory.

    Args:
        data_dir: Directory containing the four original case-study extracts.
        config: Optional deterministic controls; omission uses documented defaults.

    Returns:
        Aggregate cohort, feature, model, CV, LASSO, diagnostic, and metadata
        artifacts plus the source directory boundary used by the writer.

    Side effects:
        Reads source files but writes nothing and does not modify input extracts.

    Statistical intent:
        Applies one common eligible cohort, uses common-target held-out metrics
        for cross-family prediction, and keeps exploratory selection separate
        from significance or unbiased longitudinal performance claims.
    """
    if config is None:
        analysis_config = AnalysisConfig()
    elif isinstance(config, AnalysisConfig):
        analysis_config = config
    else:
        raise TypeError("config must be an AnalysisConfig or None")
    source_directory = Path(data_dir).resolve()

    output_path = Path("/home/luis/projects/internal/01-clean_slate/case-study-9amhealth_CLEAN-SLATE/python_analysis/outputs/out.txt")

    # Source loading, cohort selection, and feature construction are separated
    # so their aggregate audits remain traceable through pipeline outputs.
    data = load_case_study_data(source_directory)
    complete_pair_cohort = build_analysis_cohort(data)
    # Preserve complete-pair eligibility in the audit, then restrict every
    # feature and model to the two conditions confirmed as comparable.
    cohort = restrict_to_primary_member_types(complete_pair_cohort)
    features = build_member_features(data, cohort)
    members = features.member_features

    # Fit the three longitudinal specifications and percentage OLS exactly as
    # requested. Likelihood metrics stay within their labeled outcome families.
    raw_un = fit_longitudinal_gls(
        members,
        outcome_scale="raw",
        covariance_structure="unstructured",
    )
    log_cs = fit_longitudinal_gls(
        members,
        outcome_scale="log",
        covariance_structure="compound_symmetry",
    )
    log_cs_interaction = fit_longitudinal_gls(
        members,
        outcome_scale="log",
        covariance_structure="compound_symmetry",
        include_time_by_member_type=True,
    )
    percentage = fit_percentage_loss_ols(members)

    # Both base families receive identical stratified member folds and are
    # evaluated only after conversion to the common raw last-weight target.
    base_cv = compare_base_models(
        members,
        n_splits=analysis_config.cv_folds,
        n_repeats=analysis_config.cv_repeats,
        random_state=analysis_config.seed,
    )

    with output_path.open("w") as output_file:
        print("\nBASE CV RESULTS \n", file=output_file)
        print(base_cv, file=output_file)
        
    # Module mean and its defining domains remain in separate LASSO designs so
    # their exact deterministic dependency cannot make allocations non-unique.
    mean_design = build_percentage_penalized_design(members, module_spec="mean")
    domain_design = build_percentage_penalized_design(
        members,
        module_spec="domains",
    )
    mean_selection = _percentage_lasso_summary(
        mean_design,
        "mean",
        analysis_config,
        stability_seed_offset=11_000,
    )
    domain_selection = _percentage_lasso_summary(
        domain_design,
        "domains",
        analysis_config,
        stability_seed_offset=12_000,
    )
    diagnostics = _diagnostics(
        data,
        members,
        mean_design,
        domain_design,
        log_cs,
    )

    print("\nDIAGNOSTICS \n")
    print(diagnostics)

    # The generic design intentionally retains any additional observed sex
    # contrasts for descriptive provenance. Only the approved candidate set is
    # eligible for the locked equation, so extra factor levels cannot enter the
    # post-selection HC3 refit by accident.
    mean_locking_rows = mean_selection.loc[
        mean_selection["candidate"].isin(_LASSO_CANDIDATE_ORDER)
    ].copy()
    domain_locking_rows = domain_selection.loc[
        domain_selection["candidate"].isin(_LASSO_CANDIDATE_ORDER)
    ].copy()
    for locking_rows in (mean_locking_rows, domain_locking_rows):
        inestimable = locking_rows["excluded_from_full_fit"].astype(bool)
        locking_rows.loc[inestimable, "selection_frequency"] = 0.0
        locking_rows.loc[inestimable, "selection_count"] = 0
        locking_rows.loc[inestimable, "selected_at_threshold"] = False
    winning_base_model = choose_base_model_winner(base_cv)
    locked_selection = select_locked_candidates(
        mean_locking_rows,
        domain_locking_rows,
        threshold=analysis_config.stability_selection_threshold,
    )
    locked_model = fit_locked_percentage_model(
        members,
        locked_selection,
        winning_base_model,
    )
    reporting = build_reporting_tables(
        data,
        cohort,
        features,
        base_cv,
        mean_selection,
        domain_selection,
        locked_model,
    )

    return AnalysisResult(
        cohort_summary=_cohort_summary(cohort),
        feature_summary=_feature_summary(features),
        model_summary=_model_summary(
            raw_un,
            log_cs,
            log_cs_interaction,
            percentage,
        ),
        base_model_cv=base_cv,
        lasso_mean_selection=mean_selection,
        lasso_domain_selection=domain_selection,
        diagnostics=diagnostics,
        locked_model=locked_model,
        reporting_tables=reporting.tables,
        metadata=_metadata(
            data,
            complete_pair_cohort,
            cohort,
            analysis_config,
            mean_selection,
            domain_selection,
            locked_model,
        ),
        source_data_directory=source_directory,
    )


def _result_tables(result: AnalysisResult) -> dict[str, pd.DataFrame]:
    """Copy each approved scientific CSV table from the in-memory result.

    Args:
        result: Completed aggregate analysis result.

    Returns:
        A filename-keyed mapping in deterministic output order.

    Side effects:
        None.

    Statistical intent:
        Centralizes the exact disclosure schema associated with every artifact.
    """
    if not isinstance(result.reporting_tables, Mapping):
        raise TypeError("reporting_tables must be a mapping")
    if tuple(result.reporting_tables) != tuple(REPORTING_SCHEMAS):
        raise ValueError("reporting_tables must match the approved file order")
    return {
        filename: result.reporting_tables[filename].copy(deep=True)
        for filename in REPORTING_SCHEMAS
    }


def _normalize_identifier(value: str) -> str:
    """Normalize text for conservative source-identifier substring matching.

    Args:
        value: Source or output string potentially containing a member ID.

    Returns:
        A whitespace-trimmed, case-folded comparison value.

    Side effects:
        None.

    Statistical intent:
        Detects member identifiers despite harmless casing or surrounding-space
        changes while retaining legitimate labels that contain no actual ID.
    """
    return value.strip().casefold()


def _source_privacy_context(
    source_data_directory: Path,
) -> tuple[frozenset[str], dict[str, int]]:
    """Re-read sources once for identifiers and exact aggregate row counts.

    Args:
        source_data_directory: Original directory recorded by the analysis run.

    Returns:
        Normalized nonblank source IDs and canonical table lengths keyed exactly
        like the public metadata row-count mapping.

    Side effects:
        Reads the four source files again; writes and mutations never occur.

    Statistical intent:
        Gives privacy and metadata gates one consistent source snapshot without
        retaining member identifiers inside aggregate ``AnalysisResult`` data.
    """
    data = load_case_study_data(source_data_directory)
    source_tables = {
        "body_weights": data.body_weights,
        "demographics": data.demographics,
        "engagement": data.engagement,
        "module_completions": data.module_completions,
    }
    identifiers: set[str] = set()
    for table in source_tables.values():
        for value in table["member_id"].dropna().astype(str):
            normalized = _normalize_identifier(value)
            if normalized:
                identifiers.add(normalized)
    row_counts = {
        name: int(len(table)) for name, table in source_tables.items()
    }
    return frozenset(identifiers), row_counts


def _contains_source_identifier(
    value: object,
    source_member_ids: frozenset[str],
) -> bool:
    """Detect a normalized source ID inside text or structured JSON-like data.

    Args:
        value: Candidate scalar or nested mapping/sequence from an output cell.
        source_member_ids: Normalized identities prohibited from outputs.

    Returns:
        ``True`` when any source ID occurs anywhere in string content.

    Side effects:
        None.

    Statistical intent:
        Closes prefix, suffix, free-text, JSON-key, and JSON-value disclosure
        paths without suppressing aggregate factor labels that contain no ID.
    """
    if isinstance(value, str):
        normalized = _normalize_identifier(value)
        if any(
            member_id in normalized for member_id in source_member_ids
        ):
            return True

        # A valid JSON cell can spell identifier characters with ``\uXXXX``
        # escapes. Decode once and recursively inspect keys, values, or a
        # top-level JSON string; ordinary malformed text remains raw-scanned.
        try:
            decoded_value = json.loads(value)
        except json.JSONDecodeError:
            return False
        # Although JSON decoding normally consumes syntax, this explicit guard
        # prevents an identical decoded string from recursing without progress.
        if isinstance(decoded_value, str) and decoded_value == value:
            return False
        return _contains_source_identifier(
            decoded_value,
            source_member_ids,
        )
    if isinstance(value, Mapping):
        return any(
            _contains_source_identifier(key, source_member_ids)
            or _contains_source_identifier(nested_value, source_member_ids)
            for key, nested_value in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_source_identifier(item, source_member_ids)
            for item in value
        )
    return False


def _validate_aggregate_table(
    table: pd.DataFrame,
    filename: str,
    expected_columns: tuple[str, ...],
    source_member_ids: frozenset[str],
) -> None:
    """Enforce one closed schema and reject source IDs anywhere in cell text.

    Args:
        table: Candidate aggregate dataframe to serialize.
        filename: Output filename used in focused validation errors.
        expected_columns: Complete permitted column sequence for this file.
        source_member_ids: Normalized identities prohibited from output cells.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Replaces incomplete identifier denylists with an exact artifact contract
        and a value-level member disclosure check.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{filename} must be a pandas DataFrame")
    actual_columns = tuple(str(column) for column in table.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{filename} must match its exact ordered schema; expected "
            f"{expected_columns}, received {actual_columns}"
        )

    # Inspect strings and JSON-like objects rather than guessing identifier-like
    # columns; legitimate factor labels remain unless they contain an actual ID.
    for value in table.to_numpy(dtype=object).ravel(order="C"):
        if _contains_source_identifier(value, source_member_ids):
            raise ValueError(
                f"{filename} contains a source member identifier value"
            )


def _validate_metadata_tree(
    value: object,
    source_member_ids: frozenset[str],
    path: str,
) -> None:
    """Reject metadata identifiers, nonfinite numbers, and unsupported values.

    Args:
        value: Current scalar or nested metadata container.
        source_member_ids: Normalized identities prohibited from metadata.
        path: Human-readable location used in validation errors.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Ensures metadata remains deterministic, finite, aggregate, and safely
        JSON-serializable before any output file is staged.
    """
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _contains_source_identifier(key, source_member_ids):
                raise ValueError(
                    f"{path} contains a source member identifier key"
                )
            _validate_metadata_tree(
                nested_value,
                source_member_ids,
                f"{path}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_metadata_tree(
                nested_value,
                source_member_ids,
                f"{path}[{index}]",
            )
        return
    if isinstance(value, str):
        if _contains_source_identifier(value, source_member_ids):
            raise ValueError(f"{path} contains a source member identifier")
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, Integral):
        return
    if isinstance(value, Real):
        if not np.isfinite(float(value)):
            raise ValueError(f"{path} contains a nonfinite number")
        return
    raise TypeError(f"{path} contains an unsupported metadata value")


def _validate_metadata(
    metadata: Mapping[str, object],
    source_member_ids: frozenset[str],
    expected_source_row_counts: Mapping[str, int],
    tables: Mapping[str, pd.DataFrame],
    locked_model: LockedModelResult,
) -> dict[str, object]:
    """Validate the exact metadata key, type, and value contract.

    Args:
        metadata: Candidate run metadata from ``AnalysisResult``.
        source_member_ids: Normalized identities prohibited from metadata.
        expected_source_row_counts: Canonical counts from reloaded sources.
        tables: Schema-validated scientific tables used for reconciliation.
        locked_model: In-memory locked fit supplying winner and inference labels.

    Returns:
        A plain validated dictionary ready for deterministic JSON serialization.

    Side effects:
        None.

    Statistical intent:
        Prevents schema drift, impossible controls, identifier disclosure, and
        nonfinite configuration values in the reproducibility record.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if set(metadata) != _METADATA_KEYS:
        missing = sorted(_METADATA_KEYS.difference(metadata))
        extra = sorted(set(metadata).difference(_METADATA_KEYS))
        raise ValueError(
            "metadata must match its exact key schema; "
            f"missing={missing}, extra={extra}"
        )
    validated = dict(metadata)
    _validate_metadata_tree(validated, source_member_ids, "metadata")

    integer_rules = {
        "seed": 0,
        "cv_folds": 2,
        "cv_repeats": 1,
        "stability_resamples": 1,
        "complete_pair_eligible_members": 1,
        "excluded_nonprimary_members": 0,
        "included_members": 1,
    }
    for key, minimum in integer_rules.items():
        value = validated[key]
        if isinstance(value, bool) or type(value) is not int or value < minimum:
            raise ValueError(f"metadata.{key} must be an integer >= {minimum}")

    ratios = validated["lambda_ratios"]
    if not isinstance(ratios, list) or not ratios:
        raise ValueError("metadata.lambda_ratios must be a nonempty list")
    numeric_ratios: list[float] = []
    for ratio in ratios:
        if isinstance(ratio, bool) or type(ratio) not in (int, float):
            raise ValueError("metadata.lambda_ratios values must be numeric")
        numeric_ratio = float(ratio)
        if not np.isfinite(numeric_ratio) or not 0.0 < numeric_ratio <= 1.0:
            raise ValueError("metadata.lambda_ratios values must lie in (0, 1]")
        numeric_ratios.append(numeric_ratio)
    if len(set(numeric_ratios)) != len(numeric_ratios):
        raise ValueError("metadata.lambda_ratios values must be unique")

    for key, include_one in (
        ("stability_subsample_fraction", False),
        ("stability_selection_threshold", True),
        ("longitudinal_exploratory_lambda_ratio", True),
        ("lasso_specification_tie_tolerance", True),
    ):
        value = validated[key]
        if isinstance(value, bool) or type(value) not in (int, float):
            raise ValueError(f"metadata.{key} must be numeric")
        numeric_value = float(value)
        upper_valid = numeric_value <= 1.0 if include_one else numeric_value < 1.0
        lower_valid = (
            numeric_value >= 0.0
            if key == "lasso_specification_tie_tolerance"
            else numeric_value > 0.0
        )
        if not np.isfinite(numeric_value) or not lower_valid or not upper_valid:
            raise ValueError(f"metadata.{key} is outside its probability range")

    responder_threshold = validated["responder_threshold_percentage"]
    if (
        isinstance(responder_threshold, bool)
        or type(responder_threshold) not in (int, float)
        or float(responder_threshold) != 5.0
    ):
        raise ValueError("metadata.responder_threshold_percentage must equal 5.0")

    source_counts = validated["source_row_counts"]
    expected_source_keys = {
        "body_weights",
        "demographics",
        "engagement",
        "module_completions",
    }
    if (
        not isinstance(source_counts, dict)
        or set(source_counts) != expected_source_keys
    ):
        raise ValueError("metadata.source_row_counts has an invalid schema")
    if any(
        isinstance(count, bool) or type(count) is not int or count < 0
        for count in source_counts.values()
    ):
        raise ValueError("metadata.source_row_counts must be nonnegative integers")
    if source_counts != dict(expected_source_row_counts):
        raise ValueError(
            "metadata.source_row_counts must equal the reloaded source counts"
        )

    if validated["longitudinal_exploratory_lambda_ratio"] != (
        _LONGITUDINAL_EXPLORATORY_LAMBDA_RATIO
    ):
        raise ValueError(
            "metadata.longitudinal_exploratory_lambda_ratio must equal 0.2"
        )
    cohort_flow = tables["nineam_cohort_flow.csv"]
    primary_rows = cohort_flow.loc[
        cohort_flow["stage_id"].eq("primary_member_types")
    ]
    if len(primary_rows) != 1:
        raise ValueError("cohort flow must contain one primary_member_types row")
    primary_row = primary_rows.iloc[0]
    reconciled_counts = {
        "complete_pair_eligible_members": int(primary_row["starting_n"]),
        "excluded_nonprimary_members": int(primary_row["excluded_n"]),
        "included_members": int(primary_row["retained_n"]),
    }
    for key, expected in reconciled_counts.items():
        if validated[key] != expected:
            raise ValueError(f"metadata.{key} does not reconcile to cohort flow")

    exact_values = {
        "analysis_schema_version": "2.0.0",
        "reference_member_type": REFERENCE_MEMBER_TYPE,
        "base_model_comparison_target": "raw_last_weight",
        "base_model_winner_rule": (
            "lowest_mean_rmse_then_lowest_mean_mae_then_"
            "percentage_loss_ols_on_exact_tie"
        ),
        "winning_base_model": locked_model.winning_base_model,
        "locked_covariance_estimator": "HC3",
        "locked_inference_status": "conditional_exploratory",
        "selection_interpretation": _SELECTION_INTERPRETATION,
        "longitudinal_lasso_interpretation": _LONGITUDINAL_INTERPRETATION,
        "lasso_fold_plan": (
            "paired_grouped_member_folds_shared_across_mean_and_domains"
        ),
    }
    for key, expected in exact_values.items():
        if type(validated[key]) is not str or validated[key] != expected:
            raise ValueError(f"metadata.{key} has an invalid value")
    if validated["base_model_comparison_metrics"] != ["rmse", "mae"]:
        raise ValueError("metadata.base_model_comparison_metrics is invalid")
    if validated["cross_family_likelihood_comparison"] is not False:
        raise ValueError("metadata cannot enable cross-family likelihood comparison")
    if validated["lasso_specification_tie_tolerance"] != 1e-12:
        raise ValueError("metadata.lasso_specification_tie_tolerance must equal 1e-12")

    fold_ids = set()
    for filename in (
        "nineam_lasso_mean_selection.csv",
        "nineam_lasso_domain_selection.csv",
    ):
        fold_ids.update(tables[filename]["fold_plan_id"].astype(str).unique())
    if fold_ids != {validated["lasso_fold_plan_id"]}:
        raise ValueError("metadata.lasso_fold_plan_id does not match LASSO tables")

    covariance = set(
        locked_model.coefficient_table["covariance_estimator"].astype(str)
    )
    inference = set(locked_model.coefficient_table["inference_status"].astype(str))
    if covariance != {"HC3"} or inference != {"conditional_exploratory"}:
        raise ValueError("locked model covariance or inference labels are invalid")
    if validated["output_files"] != sorted(
        path.as_posix() for path in _OUTPUT_RELATIVE_PATHS
    ):
        raise ValueError("metadata.output_files is invalid")
    return validated


def _validated_included_members(cohort_flow: pd.DataFrame) -> int:
    """Read the final included-members count from scientific cohort flow.

    Args:
        cohort_flow: Schema-validated scientific cohort-flow table.

    Returns:
        The unique nonnegative integer analysis denominator.

    Side effects:
        None.

    Statistical intent:
        Anchors metadata to the same reported cohort denominator instead of an
        independently mutable value.
    """
    included_rows = cohort_flow.loc[
        cohort_flow["stage_id"].eq("primary_member_types"),
        "retained_n",
    ]
    if len(included_rows) != 1:
        raise ValueError(
            "nineam_cohort_flow.csv must contain one primary_member_types row"
        )
    value = included_rows.iloc[0]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or float(value) < 0.0
        or not float(value).is_integer()
    ):
        raise ValueError(
            "nineam_cohort_flow.csv retained_n must be an integer"
        )
    return int(value)


def _serialize_output_payloads(
    result: AnalysisResult,
    source_member_ids: frozenset[str],
    expected_source_row_counts: Mapping[str, int],
) -> dict[Path, bytes]:
    """Prevalidate and serialize every output before final files are touched.

    Args:
        result: Completed aggregate analysis result.
        source_member_ids: Normalized identities prohibited from outputs.
        expected_source_row_counts: Canonical counts from reloaded sources.

    Returns:
        Relative-path-keyed UTF-8 payloads in deterministic output order.

    Side effects:
        None; serialization occurs entirely in memory.

    Statistical intent:
        Ensures a late schema, privacy, metadata, or serialization failure cannot
        partially publish a new analysis run.
    """
    tables = _result_tables(result)
    payloads: dict[Path, bytes] = {}
    for filename, expected_columns in _CSV_OUTPUT_SCHEMAS.items():
        table = tables[filename]
        _validate_aggregate_table(
            table,
            filename,
            expected_columns,
            source_member_ids,
        )
        # Stable row order comes from orchestration; fixed line and float formats
        # make independently staged payloads byte-identical across repeated runs.
        csv_text = table.to_csv(
            None,
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )
        payloads[Path("tables") / filename] = csv_text.encode("utf-8")

    expected_included_members = _validated_included_members(
        tables["nineam_cohort_flow.csv"]
    )
    if result.metadata.get("included_members") != expected_included_members:
        raise ValueError("metadata.included_members does not match cohort flow")
    metadata = _validate_metadata(
        result.metadata,
        source_member_ids,
        expected_source_row_counts,
        tables,
        result.locked_model,
    )
    metadata_text = json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    payloads[_METADATA_RELATIVE_PATH] = (
        metadata_text + "\n"
    ).encode("utf-8")
    return payloads


def _stage_payloads(
    destination: Path,
    payloads: Mapping[Path, bytes],
) -> dict[Path, Path]:
    """Write every serialized payload to a durable temporary sibling file.

    Args:
        destination: Existing aggregate output directory.
        payloads: Relative final paths and fully serialized bytes.

    Returns:
        A mapping from each final path to its staged sibling path.

    Side effects:
        Creates, flushes, and synchronizes temporary files; cleans them all if
        any staging write fails.

    Statistical intent:
        Prepares a complete run for transactional commit without exposing a
        mixture of old and partially serialized artifacts.
    """
    staged: dict[Path, Path] = {}
    try:
        for relative_path, payload in payloads.items():
            final_path = destination / relative_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.",
                suffix=".stage",
                dir=final_path.parent,
            )
            temporary_path = Path(temporary_name)
            staged[final_path] = temporary_path
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
        raise
    return staged


def _reserve_backup_path(final_path: Path) -> Path:
    """Reserve one temporary sibling path for an existing final-file backup.

    Args:
        final_path: Aggregate artifact that may need rollback restoration.

    Returns:
        An empty sibling path ready to receive an atomic replacement.

    Side effects:
        Creates and closes one empty temporary file.

    Statistical intent:
        Keeps each prior artifact recoverable until all staged files commit.
    """
    file_descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".backup",
        dir=final_path.parent,
    )
    os.close(file_descriptor)
    return Path(backup_name)


def _commit_staged_payloads(staged: Mapping[Path, Path]) -> None:
    """Commit staged artifacts with per-file replacement and set rollback.

    Args:
        staged: Ordered final-to-stage mapping for one complete output run.

    Returns:
        None.

    Side effects:
        Moves prior finals to sibling backups, atomically installs each stage,
        removes backups on success, and attempts to restore every prior final
        on failure. A backup that cannot be restored is retained for recovery.

    Statistical intent:
        Restores the prior complete artifact set after a filesystem replacement
        error. The multi-file commit is sequential, so it does not isolate
        concurrent readers during the short commit window.
    """
    backups: dict[Path, Path] = {}
    all_backup_paths: list[Path] = []
    committed_finals: list[Path] = []
    try:
        # Preserve every old artifact before installing any new payload so the
        # entire named set can roll back as one logical analysis publication.
        for final_path in staged:
            if final_path.exists():
                backup_path = _reserve_backup_path(final_path)
                all_backup_paths.append(backup_path)
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path
        for final_path, temporary_path in staged.items():
            os.replace(temporary_path, final_path)
            committed_finals.append(final_path)
    except BaseException as commit_error:
        rollback_errors: list[BaseException] = []
        retained_backup_paths: set[Path] = set()
        for final_path in committed_finals:
            try:
                final_path.unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(error)
        for final_path, backup_path in backups.items():
            try:
                if backup_path.exists():
                    os.replace(backup_path, final_path)
            except OSError as error:
                rollback_errors.append(error)
                # A failed restoration leaves this backup as the only known
                # recoverable copy, so cleanup must never remove it.
                if backup_path.exists():
                    retained_backup_paths.add(backup_path)
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)
        for backup_path in all_backup_paths:
            if backup_path not in retained_backup_paths:
                backup_path.unlink(missing_ok=True)
        if rollback_errors:
            retained_locations = ", ".join(
                str(path) for path in sorted(retained_backup_paths)
            )
            recovery_detail = (
                f" Retained recoverable backups: {retained_locations}."
                if retained_locations
                else ""
            )
            raise RuntimeError(
                "Output commit failed and rollback could not restore every file"
                f".{recovery_detail}"
            ) from commit_error
        raise
    else:
        for backup_path in all_backup_paths:
            backup_path.unlink(missing_ok=True)
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)


def write_analysis_outputs(
    result: AnalysisResult,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Publish deterministic scientific tables, metadata, and figures.

    Args:
        result: Completed in-memory aggregate analysis result.
        output_dir: Destination directory separate from the source data folder.

    Returns:
        Output paths in a fixed filename order.

    Side effects:
        Creates ``tables`` and ``figures`` subdirectories, publishes fifteen
        CSVs, one JSON metadata file, and eighteen deterministic image files;
        source files are never changed.

    Statistical intent:
        Serializes reproducible aggregate evidence without member rows,
        timestamps, or cross-family likelihood rankings.
    """
    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be an AnalysisResult")
    destination = Path(output_dir).resolve()
    source_directory = result.source_data_directory.resolve()
    if destination == source_directory or source_directory in destination.parents:
        raise ValueError(
            "output_dir cannot equal or be beneath the source data directory"
        )
    if destination.exists() and not destination.is_dir():
        raise ValueError("output_dir must be a directory")

    # Re-read exact IDs and serialize every artifact before creating stages or
    # modifying a final file; all privacy and schema failures are therefore safe.
    source_member_ids, source_row_counts = _source_privacy_context(
        source_directory
    )
    payloads = _serialize_output_payloads(
        result,
        source_member_ids,
        source_row_counts,
    )
    destination.mkdir(parents=True, exist_ok=True)
    staged = _stage_payloads(destination, payloads)
    _commit_staged_payloads(staged)

    # Plot only from the validated and already-published scientific tables.
    # Images are first rendered in an isolated directory, then the complete
    # eighteen-file set is committed with the same rollback mechanism.
    with tempfile.TemporaryDirectory(
        prefix=".nineam_figures.",
        dir=destination,
    ) as temporary_directory:
        rendered = write_analysis_figures(
            _result_tables(result),
            Path(temporary_directory),
        )
        expected_names = tuple(
            relative_path.name for relative_path in _FIGURE_RELATIVE_PATHS
        )
        if tuple(path.name for path in rendered) != expected_names:
            raise ValueError("Figure writer returned an unexpected artifact set")
        figure_directory = destination / "figures"
        figure_directory.mkdir(parents=True, exist_ok=True)
        staged_figures = {
            figure_directory / path.name: path for path in rendered
        }
        _commit_staged_payloads(staged_figures)
    return tuple(destination / path for path in _OUTPUT_RELATIVE_PATHS)
