"""Lock post-LASSO candidates and refit the confirmed HC3 percentage model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist
from types import MappingProxyType

import numpy as np
import pandas as pd
import statsmodels.api as sm

COEFFICIENT_COLUMNS = (
    "model_id", "term_order", "term", "term_label", "term_role",
    "support_scope", "contrast", "reference", "estimate", "standard_error",
    "test_statistic", "reference_distribution", "degrees_of_freedom", "p_value",
    "ci_lower", "ci_upper", "ci_level", "unit", "n_members",
    "covariance_estimator", "winning_base_model", "winning_module_spec",
    "selection_frequency", "selection_provenance", "inference_status",
)
FIT_STATISTICS_COLUMNS = (
    "model_id", "winning_base_model", "winning_module_spec", "analysis_population",
    "modeled_outcome", "model_formula", "reference_member_type",
    "duration_adjustment", "n_members", "n_parameters",
    "residual_degrees_of_freedom", "covariance_estimator", "r_squared",
    "adjusted_r_squared", "residual_rmse", "negative_two_log_likelihood", "aic",
    "bic", "likelihood_use", "standardized_design_condition_number",
    "maximum_leverage", "maximum_cooks_distance", "selection_rule",
    "inference_status", "fit_status",
)
DIAGNOSTIC_SUMMARY_COLUMNS = (
    "model_id", "diagnostic_order", "diagnostic_type", "row_type", "series",
    "bin_order", "bin_method", "bin_lower", "bin_upper", "bin_count", "x_value",
    "y_value", "y_lower", "y_upper", "metric", "value", "threshold", "flag",
    "status", "interpretation",
)
_REFERENCE = "Coaching Only"
_ACTIVE = "Active GLP-1 for Weight-loss"
_CONTINUOUS_CANDIDATES = (
    "engagement_volume_repeatable", "engagement_volume_repeatable_rate",
    "engagement_breadth", "tenure_days", "module_mean", "module_core",
    "module_mindset", "module_nutrition", "module_physical_activity",
)
_CANONICAL_CANDIDATE_ORDER = {
    name: index
    for index, name in enumerate(
        (*_CONTINUOUS_CANDIDATES, "sex[MALE]"), start=1
    )
}
_MEAN_CANDIDATES = {
    "engagement_volume_repeatable",
    "engagement_volume_repeatable_rate",
    "engagement_breadth",
    "tenure_days",
    "module_mean",
    "sex[MALE]",
}
_DOMAIN_CANDIDATES = {
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
_LIMITED_MODULE_SUPPORT = {
    "module_mean",
    "module_mindset",
    "module_nutrition",
    "module_physical_activity",
}


@dataclass(frozen=True, slots=True)
class LockedCandidateSelection:
    """Store one chosen LASSO specification and stable candidate provenance."""

    module_spec: str
    candidates: tuple[str, ...]
    cv_mean_mse: float
    selection_threshold: float
    selection_frequencies: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate a finite, internally aligned locked-candidate decision."""
        if self.module_spec not in ("mean", "domains"):
            raise ValueError("module_spec must be 'mean' or 'domains'")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Locked candidates cannot contain duplicate names")
        if not np.isfinite(self.cv_mean_mse):
            raise ValueError("cv_mean_mse must be finite")
        if not 0.0 < float(self.selection_threshold) <= 1.0:
            raise ValueError("selection_threshold must lie in (0, 1]")
        candidates = tuple(str(name) for name in self.candidates)
        if any(name.startswith("sex[") and name != "sex[MALE]" for name in candidates):
            raise ValueError("The canonical sex contrast is sex[MALE]")
        allowed = _MEAN_CANDIDATES if self.module_spec == "mean" else _DOMAIN_CANDIDATES
        invalid = sorted(set(candidates).difference(allowed))
        if invalid:
            raise ValueError(
                f"{self.module_spec} specification contains unsupported candidates: "
                + ", ".join(invalid)
            )
        canonical = tuple(sorted(candidates, key=_CANONICAL_CANDIDATE_ORDER.__getitem__))
        if candidates != canonical:
            raise ValueError("Locked candidates must follow canonical candidate order")
        frequencies = {
            str(name): float(value)
            for name, value in self.selection_frequencies.items()
        }
        if set(frequencies) != set(self.candidates):
            raise ValueError("selection_frequencies must match locked candidates")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in frequencies.values()):
            raise ValueError("selection frequencies must be finite values in [0, 1]")
        if any(value < float(self.selection_threshold) for value in frequencies.values()):
            raise ValueError("Every locked selection frequency must meet the threshold")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "selection_frequencies",
            MappingProxyType(frequencies.copy()),
        )


@dataclass(frozen=True, slots=True)
class LockedModelResult:
    """Expose aggregate HC3 results and no-identifier internal diagnostics."""

    primary_model_id: str
    winning_base_model: str
    winning_module_spec: str
    selected_candidates: tuple[str, ...]
    coefficient_table: pd.DataFrame
    fit_statistics: pd.DataFrame
    diagnostic_summary: pd.DataFrame
    diagnostic_points: pd.DataFrame


def choose_base_model_winner(base_model_cv: pd.DataFrame) -> str:
    """Choose the held-out winner by mean RMSE, mean MAE, and primary tie-break."""
    if not isinstance(base_model_cv, pd.DataFrame):
        raise TypeError("base_model_cv must be a pandas DataFrame")
    required = {"model", "rmse", "mae"}
    missing = sorted(required.difference(base_model_cv.columns))
    if missing:
        raise ValueError("base_model_cv is missing required columns: " + ", ".join(missing))
    table = base_model_cv.loc[:, ["model", "rmse", "mae"]].copy()
    table["model"] = table["model"].astype("string").str.strip()
    if table.empty or table["model"].isna().any() or table["model"].eq("").any():
        raise ValueError("base_model_cv must contain named model rows")
    try:
        metrics = table[["rmse", "mae"]].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("base_model_cv metrics must be numeric") from error
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError("base_model_cv metrics must be finite")
    scores = metrics.groupby(table["model"], sort=True).mean()
    minimum_rmse = float(scores["rmse"].min())
    eligible = scores.index[scores["rmse"].eq(minimum_rmse)]
    minimum_mae = float(scores.loc[eligible, "mae"].min())
    finalists = sorted(name for name in eligible if scores.loc[name, "mae"] == minimum_mae)
    if "percentage_loss_ols" in finalists:
        return "percentage_loss_ols"
    return finalists[0]


def _validate_selection_table(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize one LASSO table and require one consistent specification summary."""
    required = {
        "module_spec",
        "candidate",
        "cv_mean_mse",
        "selection_threshold",
        "selection_frequency",
    }
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{label} selection must be a pandas DataFrame")
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} selection is missing required columns: " + ", ".join(missing))
    if table.empty:
        raise ValueError(f"{label} selection cannot be empty")
    prepared = table.loc[:, sorted(required)].copy()
    if "candidate_order" in table.columns:
        prepared["candidate_order"] = table["candidate_order"]
    else:
        # Older in-memory LASSO summaries predate the reporting schema. Their
        # terms still receive the prespecified canonical order before locking.
        prepared["candidate_order"] = prepared["candidate"].map(
            _CANONICAL_CANDIDATE_ORDER
        ).fillna(len(_CANONICAL_CANDIDATE_ORDER) + 1)
    expected_spec = "mean" if label == "mean" else "domains"
    if set(prepared["module_spec"].astype(str)) != {expected_spec}:
        raise ValueError(f"{label} selection must contain only {expected_spec} rows")
    for column in ("cv_mean_mse", "selection_threshold", "selection_frequency"):
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    if not np.isfinite(prepared[["cv_mean_mse", "selection_threshold", "selection_frequency"]].to_numpy(float)).all():
        raise ValueError(f"{label} selection values must be finite")
    if prepared["candidate"].astype(str).str.strip().duplicated().any():
        raise ValueError(f"{label} selection candidates must be unique")
    if prepared["candidate_order"].duplicated().any():
        raise ValueError(f"{label} selection candidate order must be unique")
    candidate_names = tuple(prepared["candidate"].astype(str).str.strip())
    allowed = _MEAN_CANDIDATES if expected_spec == "mean" else _DOMAIN_CANDIDATES
    invalid = sorted(set(candidate_names).difference(allowed))
    if invalid:
        raise ValueError(
            f"{label} selection contains candidates incompatible with the "
            f"{expected_spec} specification: " + ", ".join(invalid)
        )
    expected_order = prepared["candidate"].map(_CANONICAL_CANDIDATE_ORDER)
    observed_order = pd.to_numeric(prepared["candidate_order"], errors="raise")
    if not np.array_equal(observed_order.to_numpy(), expected_order.to_numpy()):
        raise ValueError(f"{label} selection must use canonical candidate_order values")
    for column in ("cv_mean_mse", "selection_threshold"):
        values = prepared[column].to_numpy(float)
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} selection must have consistent {column}")
    if not prepared["selection_threshold"].between(0.0, 1.0, inclusive="right").all():
        raise ValueError(f"{label} selection threshold must lie in (0, 1]")
    if not prepared["selection_frequency"].between(0.0, 1.0).all():
        raise ValueError(f"{label} selection frequencies must lie in [0, 1]")
    return prepared.sort_values("candidate_order", kind="stable").reset_index(drop=True)


def select_locked_candidates(mean_selection: pd.DataFrame, domain_selection: pd.DataFrame, threshold: float = 0.75) -> LockedCandidateSelection:
    """Choose one CV-winning LASSO specification and its stable canonical terms."""
    if not isinstance(threshold, Real) or not np.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    mean = _validate_selection_table(mean_selection, "mean")
    domains = _validate_selection_table(domain_selection, "domains")
    for label, table in (("mean", mean), ("domains", domains)):
        table_threshold = float(table.loc[0, "selection_threshold"])
        if table_threshold != float(threshold):
            raise ValueError(
                f"{label} selection threshold does not match requested threshold"
            )
    mean_score = float(mean.loc[0, "cv_mean_mse"])
    domain_score = float(domains.loc[0, "cv_mean_mse"])
    chosen = mean if mean_score <= domain_score + 1e-12 else domains
    chosen_spec = str(chosen.loc[0, "module_spec"])
    stable = chosen.loc[chosen["selection_frequency"] >= float(threshold)]
    candidates = tuple(stable["candidate"].astype(str))
    frequencies = {str(row.candidate): float(row.selection_frequency) for row in stable.itertuples(index=False)}
    return LockedCandidateSelection(chosen_spec, candidates, float(chosen.loc[0, "cv_mean_mse"]), float(threshold), frequencies)


def _prepared_members(member_features: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    """Validate raw endpoint and selected-candidate values for a locked refit."""
    if not isinstance(member_features, pd.DataFrame):
        raise TypeError("member_features must be a pandas DataFrame")
    source_candidates = {
        "sex" if candidate.startswith("sex[") else candidate
        for candidate in candidates
    }
    required = {"member_type", "first_weight", "last_weight", "weight_days", *source_candidates}
    missing = sorted(required.difference(member_features.columns))
    if missing:
        raise ValueError("Locked model is missing required columns: " + ", ".join(missing))
    prepared = member_features.loc[:, sorted(required)].copy()
    prepared["member_type"] = prepared["member_type"].astype("string").str.strip()
    if not set(prepared["member_type"]) == {_REFERENCE, _ACTIVE}:
        raise ValueError("Locked model requires Coaching Only and Active GLP-1 for Weight-loss")
    numeric_columns = ["first_weight", "last_weight", "weight_days"] + [name for name in candidates if not name.startswith("sex[")]
    try:
        prepared[numeric_columns] = prepared[numeric_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Locked model numeric values must be numeric") from error
    if not np.isfinite(prepared[numeric_columns].to_numpy(float)).all():
        raise ValueError("Locked model contains nonfinite values")
    if (prepared[["first_weight", "last_weight"]] <= 0.0).any(axis=None):
        raise ValueError("Locked model endpoint weights must be positive")
    return prepared.reset_index(drop=True)


def _candidate_column(prepared: pd.DataFrame, candidate: str) -> tuple[np.ndarray, str]:
    """Return one selected continuous or sex treatment-coded candidate column."""
    if candidate in _CONTINUOUS_CANDIDATES:
        return prepared[candidate].to_numpy(float), candidate
    if candidate.startswith("sex[") and candidate.endswith("]"):
        level = candidate[4:-1]
        if "sex" not in prepared.columns:
            raise ValueError(f"unsupported locked candidate: {candidate}")
        sex = prepared["sex"].astype("string").str.strip()
        if sex.isna().any() or sex.eq("").any() or set(sex) != {"FEMALE", "MALE"}:
            raise ValueError(f"unsupported locked candidate: {candidate}")
        return sex.eq(level).to_numpy(float), candidate
    raise ValueError(f"unsupported locked candidate: {candidate}")


def _design(prepared: pd.DataFrame, candidates: tuple[str, ...], *, duration: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the explicit Coaching-reference primary or duration design matrix."""
    values = [np.ones(len(prepared)), prepared["first_weight"].to_numpy(float), prepared["member_type"].eq(_ACTIVE).to_numpy(float)]
    names = ["intercept", "first_weight", f"member_type[{_ACTIVE}]"]
    for candidate in candidates:
        value, name = _candidate_column(prepared, candidate)
        values.append(value)
        names.append(name)
    if duration:
        centered = prepared["weight_days"].to_numpy(float) - float(prepared["weight_days"].mean())
        values.extend((centered, centered**2))
        names.extend(("weight_days", "weight_days_squared"))
    design = np.column_stack(values).astype(float, copy=False)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Locked model design matrix is rank deficient")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Locked model requires positive residual degrees of freedom")
    return design, tuple(names)


def _term_metadata(
    term: str,
    model_support_scope: str,
) -> tuple[str, str, str, str, str, str]:
    """Return the controlled role and interpretable reporting metadata for a term."""
    member_type_term = f"member_type[{_ACTIVE}]"
    if term == "intercept":
        return (
            "Intercept",
            "intercept",
            model_support_scope,
            "",
            "",
            "percentage_points",
        )
    if term == "first_weight":
        return (
            "Baseline weight (per lb)",
            "base",
            model_support_scope,
            "",
            "",
            "percentage_points_per_lb",
        )
    if term == member_type_term:
        return (
            "Active GLP-1 for weight loss versus coaching only",
            "base",
            model_support_scope,
            f"{_ACTIVE} versus {_REFERENCE}",
            _REFERENCE,
            "percentage_points",
        )
    if term in ("weight_days", "weight_days_squared"):
        label = (
            "Weight observation interval (centered days)"
            if term == "weight_days"
            else "Weight observation interval squared (centered days squared)"
        )
        unit = (
            "percentage_points_per_day"
            if term == "weight_days"
            else "percentage_points_per_day_squared"
        )
        return (
            label,
            "sensitivity_adjustment",
            model_support_scope,
            "",
            "",
            unit,
        )

    labels_and_units = {
        "engagement_volume_repeatable": (
            "Repeatable engagement volume",
            "percentage_points_per_repeatable_event",
        ),
        "engagement_volume_repeatable_rate": (
            "Repeatable engagement volume rate",
            "percentage_points_per_repeatable_event_per_day",
        ),
        "engagement_breadth": (
            "Engagement breadth",
            "percentage_points_per_distinct_event_type",
        ),
        "tenure_days": ("Program tenure", "percentage_points_per_day"),
        "module_mean": (
            "Mean curriculum completion proportion",
            "percentage_points_per_proportion_unit",
        ),
        "module_core": (
            "Core module completion proportion",
            "percentage_points_per_proportion_unit",
        ),
        "module_mindset": (
            "Mindset module completion proportion",
            "percentage_points_per_proportion_unit",
        ),
        "module_nutrition": (
            "Nutrition module completion proportion",
            "percentage_points_per_proportion_unit",
        ),
        "module_physical_activity": (
            "Physical-activity module completion proportion",
            "percentage_points_per_proportion_unit",
        ),
        "sex[MALE]": ("Male versus female", "percentage_points"),
    }
    label, unit = labels_and_units[term]
    support_scope = (
        "glp1_observed_only_not_verifiable_in_coaching"
        if term in _LIMITED_MODULE_SUPPORT
        else model_support_scope
    )
    contrast = "MALE versus FEMALE" if term == "sex[MALE]" else ""
    reference = "FEMALE" if term == "sex[MALE]" else ""
    return (
        label,
        "selected_candidate",
        support_scope,
        contrast,
        reference,
        unit,
    )


def _coefficient_rows(
    model_id: str,
    fitted: object,
    names: tuple[str, ...],
    selection: LockedCandidateSelection,
    winning_base_model: str,
    support_scope: str,
) -> list[dict[str, object]]:
    """Format one HC3 result with term-specific scientific metadata."""
    result = fitted
    confidence = result.conf_int(alpha=0.05)
    rows = []
    for index, term in enumerate(names):
        selected = term in selection.selection_frequencies
        label, role, term_support, contrast, reference, unit = _term_metadata(
            term,
            support_scope,
        )
        rows.append(
            {
                "model_id": model_id,
                "term_order": index + 1,
                "term": term,
                "term_label": label,
                "term_role": role,
                "support_scope": term_support,
                "contrast": contrast,
                "reference": reference,
                "estimate": float(result.params[index]),
                "standard_error": float(result.bse[index]),
                "test_statistic": float(result.tvalues[index]),
                "reference_distribution": "normal",
                # HC3 inference uses an asymptotic normal reference, so a
                # finite t-distribution degrees of freedom is not applicable.
                "degrees_of_freedom": np.nan,
                "p_value": float(result.pvalues[index]),
                "ci_lower": float(confidence[index, 0]),
                "ci_upper": float(confidence[index, 1]),
                "ci_level": 0.95,
                "unit": unit,
                "n_members": int(result.nobs),
                "covariance_estimator": "HC3",
                "winning_base_model": winning_base_model,
                "winning_module_spec": selection.module_spec,
                "selection_frequency": selection.selection_frequencies.get(
                    term, np.nan
                ),
                "selection_provenance": (
                    "stable_lasso_frequency"
                    if selected
                    else "prespecified_base_or_sensitivity"
                ),
                "inference_status": "conditional_exploratory",
            }
        )
    return rows


def _diagnostics(model_id: str, fitted: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return no-ID point diagnostics and controlled aggregate plot summaries."""
    influence = fitted.get_influence()
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    standardized = np.asarray(influence.resid_studentized_internal, dtype=float)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    points = pd.DataFrame(
        {
            "model_id": model_id,
            "fitted_value": fitted_values,
            "residual": residuals,
            "standardized_residual": standardized,
            "leverage": leverage,
            "cooks_distance": cooks,
        }
    )
    rows: list[dict[str, object]] = []

    # Stable equal-count bins summarize shape without exporting member-level
    # observations. Sorting uses a stable algorithm for deterministic ties.
    n_bins = min(10, max(1, len(points) // 2))
    sorted_indices = np.argsort(fitted_values, kind="stable")
    for bin_order, indices in enumerate(np.array_split(sorted_indices, n_bins), start=1):
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_type": "residuals_vs_fitted",
                "row_type": "plot_summary",
                "series": "binned_mean",
                "bin_order": bin_order,
                "bin_method": "stable_equal_count",
                "bin_lower": float(np.min(fitted_values[indices])),
                "bin_upper": float(np.max(fitted_values[indices])),
                "bin_count": int(indices.size),
                "x_value": float(np.mean(fitted_values[indices])),
                "y_value": float(np.mean(residuals[indices])),
                "y_lower": np.nan,
                "y_upper": np.nan,
                "metric": "mean_residual",
                "value": float(np.mean(residuals[indices])),
                "threshold": 0.0,
                "flag": False,
                "status": "descriptive",
                "interpretation": "Equal-count aggregate residual summary",
            }
        )

    # A small fixed set of empirical quantiles provides an aggregate normal
    # Q-Q diagnostic without emitting one row per member.
    n_quantiles = min(9, len(points))
    probabilities = (np.arange(n_quantiles, dtype=float) + 0.5) / n_quantiles
    empirical = np.quantile(standardized, probabilities)
    normal = NormalDist()
    for bin_order, (probability, theoretical, observed) in enumerate(
        zip(
            probabilities,
            (normal.inv_cdf(float(value)) for value in probabilities),
            empirical,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_type": "normal_qq",
                "row_type": "plot_summary",
                "series": "quantile_pair",
                "bin_order": bin_order,
                "bin_method": "fixed_probability_quantiles",
                "bin_lower": float(probability),
                "bin_upper": float(probability),
                "bin_count": int(len(points)),
                "x_value": float(theoretical),
                "y_value": float(observed),
                "y_lower": np.nan,
                "y_upper": np.nan,
                "metric": "standardized_residual_quantile",
                "value": float(observed),
                "threshold": np.nan,
                "flag": False,
                "status": "descriptive",
                "interpretation": "Aggregate standardized-residual quantile pair",
            }
        )

    standardized_predictors = fitted.model.exog[:, 1:]
    standardized_predictors = (
        standardized_predictors - standardized_predictors.mean(axis=0)
    ) / standardized_predictors.std(axis=0, ddof=0)
    condition_number = float(
        np.linalg.cond(
            np.column_stack([np.ones(int(fitted.nobs)), standardized_predictors])
        )
    )
    summary_metrics = (
        (
            "leverage",
            "maximum_leverage",
            float(np.max(leverage)),
            2 * fitted.model.exog.shape[1] / fitted.nobs,
        ),
        (
            "cooks_distance",
            "maximum_cooks_distance",
            float(np.max(cooks)),
            4 / fitted.nobs,
        ),
        (
            "global_metric",
            "maximum_absolute_standardized_residual",
            float(np.max(np.abs(standardized))),
            2.0,
        ),
        (
            "global_metric",
            "standardized_condition_number",
            condition_number,
            30.0,
        ),
    )
    for diagnostic_type, metric, value, threshold in summary_metrics:
        flagged = bool(value > threshold)
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_type": diagnostic_type,
                "row_type": "summary",
                "series": "summary_metric",
                "bin_order": np.nan,
                "bin_method": "",
                "bin_lower": np.nan,
                "bin_upper": np.nan,
                "bin_count": int(len(points)),
                "x_value": np.nan,
                "y_value": np.nan,
                "y_lower": np.nan,
                "y_upper": np.nan,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "flag": flagged,
                "status": "review" if flagged else "within_threshold",
                "interpretation": "Aggregate diagnostic; member-level points remain internal",
            }
        )

    for order, row in enumerate(rows, start=1):
        row["diagnostic_order"] = order
    summary = pd.DataFrame.from_records(rows, columns=DIAGNOSTIC_SUMMARY_COLUMNS)
    return points, summary


def _fit_rows(model_id: str, fitted: object, names: tuple[str, ...], selection: LockedCandidateSelection, winning_base_model: str, *, duration: str, analysis_population: str) -> dict[str, object]:
    """Format one fitted HC3 equation into the closed fit-statistics schema."""
    influence = fitted.get_influence()
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    predictors = fitted.model.exog[:, 1:]
    standardized = (predictors - predictors.mean(axis=0)) / predictors.std(axis=0, ddof=0)
    condition_number = float(np.linalg.cond(np.column_stack([np.ones(int(fitted.nobs)), standardized])))
    return {"model_id": model_id, "winning_base_model": winning_base_model, "winning_module_spec": selection.module_spec, "analysis_population": analysis_population, "modeled_outcome": "percentage_weight_loss", "model_formula": "percentage_loss ~ " + " + ".join(names[1:]), "reference_member_type": _REFERENCE, "duration_adjustment": duration, "n_members": int(fitted.nobs), "n_parameters": int(fitted.model.exog.shape[1]), "residual_degrees_of_freedom": float(fitted.df_resid), "covariance_estimator": "HC3", "r_squared": float(fitted.rsquared), "adjusted_r_squared": float(fitted.rsquared_adj), "residual_rmse": float(np.sqrt(np.mean(np.asarray(fitted.resid, dtype=float) ** 2))), "negative_two_log_likelihood": float(-2 * fitted.llf), "aic": float(fitted.aic), "bic": float(fitted.bic), "likelihood_use": "within_percentage_loss_family_descriptive_only", "standardized_design_condition_number": condition_number, "maximum_leverage": float(np.max(leverage)), "maximum_cooks_distance": float(np.max(cooks)), "selection_rule": "selected_model_cv_mse_then_stability_frequency", "inference_status": "conditional_exploratory", "fit_status": "fitted"}


def fit_locked_percentage_model(member_features: pd.DataFrame, selection: LockedCandidateSelection, winning_base_model: str) -> LockedModelResult:
    """Refit primary and common-support duration sensitivity equations with HC3."""
    if not isinstance(selection, LockedCandidateSelection):
        raise TypeError("selection must be a LockedCandidateSelection")
    if winning_base_model != "percentage_loss_ols":
        raise ValueError("Locked HC3 path requires percentage_loss_ols as winning_base_model")
    if len(set(selection.candidates)) != len(selection.candidates):
        raise ValueError("Locked candidates cannot contain duplicate names")
    if any(
        candidate not in _CONTINUOUS_CANDIDATES
        and not (candidate.startswith("sex[") and candidate.endswith("]"))
        for candidate in selection.candidates
    ):
        raise ValueError("Locked model contains an unsupported candidate")
    prepared = _prepared_members(member_features, selection.candidates)
    response = 100.0 * (prepared["first_weight"].to_numpy(float) - prepared["last_weight"].to_numpy(float)) / prepared["first_weight"].to_numpy(float)
    primary_design, primary_names = _design(prepared, selection.candidates, duration=False)
    primary = sm.OLS(response, primary_design).fit(cov_type="HC3")
    group_ranges = prepared.groupby("member_type", sort=False)["weight_days"].agg(["min", "max"])
    lower = float(group_ranges["min"].max())
    upper = float(group_ranges["max"].min())
    common = prepared.loc[prepared["weight_days"].between(lower, upper)].copy()
    if common.empty or common["member_type"].nunique() != 2:
        raise ValueError("No common observed weight_days support across member types")
    sensitivity_design, sensitivity_names = _design(common, selection.candidates, duration=True)
    sensitivity_response = 100.0 * (common["first_weight"].to_numpy(float) - common["last_weight"].to_numpy(float)) / common["first_weight"].to_numpy(float)
    sensitivity = sm.OLS(sensitivity_response, sensitivity_design).fit(cov_type="HC3")
    primary_id = "locked_percentage_loss_primary"
    sensitivity_id = "locked_percentage_loss_duration_sensitivity"
    coefficient_table = pd.DataFrame.from_records(_coefficient_rows(primary_id, primary, primary_names, selection, winning_base_model, "primary_two_group") + _coefficient_rows(sensitivity_id, sensitivity, sensitivity_names, selection, winning_base_model, "common_weight_days_support"), columns=COEFFICIENT_COLUMNS)
    primary_points, primary_summary = _diagnostics(primary_id, primary)
    sensitivity_points, sensitivity_summary = _diagnostics(sensitivity_id, sensitivity)
    fit_statistics = pd.DataFrame.from_records([_fit_rows(primary_id, primary, primary_names, selection, winning_base_model, duration="none", analysis_population="primary_two_group"), _fit_rows(sensitivity_id, sensitivity, sensitivity_names, selection, winning_base_model, duration="centered_linear_quadratic_weight_days_common_support", analysis_population="common_weight_days_support")], columns=FIT_STATISTICS_COLUMNS)
    return LockedModelResult(primary_id, winning_base_model, selection.module_spec, selection.candidates, coefficient_table, fit_statistics, pd.concat([primary_summary, sensitivity_summary], ignore_index=True), pd.concat([primary_points, sensitivity_points], ignore_index=True))
