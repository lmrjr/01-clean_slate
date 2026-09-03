"""Render deterministic scientific figures from validated aggregate tables.

The plotting layer never reads member-level data or recomputes estimands. Each
figure is a direct visual encoding of one or two reporting tables so every mark
can be traced back to a CSV delivered with the analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import matplotlib

# Select a non-interactive backend before importing pyplot so CLI runs do not
# require a display server or inherit a user's desktop plotting configuration.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

FIGURE_STEMS = (
    "nineam_cohort_flow",
    "nineam_weight_loss_by_member_type",
    "nineam_responder_rate_by_member_type",
    "nineam_engagement_patterns",
    "nineam_module_completion",
    "nineam_base_model_performance",
    "nineam_lasso_stability",
    "nineam_locked_model_coefficients",
    "nineam_model_diagnostics",
)

_COACHING = "Coaching Only"
_GLP1 = "Active GLP-1 for Weight-loss"
_COLORS = {
    _COACHING: "#0072B2",
    _GLP1: "#D55E00",
    "selected": "#009E73",
    "neutral": "#4D4D4D",
    "grid": "#D9D9D9",
}
_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "nineam-health-analysis-20260901",
}

# Only columns directly encoded in a figure are required here. The reporting
# builder separately enforces each full closed CSV schema before this layer runs.
_SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "nineam_cohort_flow.csv": (
        "flow_order",
        "row_type",
        "stage_label",
        "starting_n",
        "excluded_n",
        "retained_n",
    ),
    "nineam_outcomes_by_member_type.csv": (
        "scope_order",
        "scope",
        "outcome",
        "outcome_label",
        "unit",
        "mean",
        "median",
        "q1",
        "q3",
        "percentage",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "ci_method",
    ),
    "nineam_engagement_by_member_type.csv": (
        "scope_order",
        "scope",
        "metric_order",
        "metric",
        "metric_label",
        "unit",
        "mean",
        "median",
        "q1",
        "q3",
    ),
    "nineam_modules_by_member_type.csv": (
        "scope_order",
        "scope",
        "module_order",
        "module_variable",
        "module_label",
        "mean_completion_proportion",
        "support_scope",
    ),
    "nineam_base_model_comparison_summary.csv": (
        "model_order",
        "model_id",
        "metric_order",
        "metric",
        "metric_unit",
        "mean_score",
        "standard_deviation",
        "is_winner",
    ),
    "nineam_lasso_mean_selection.csv": (
        "specification_order",
        "module_spec",
        "is_winning_specification",
        "candidate_order",
        "candidate_label",
        "selection_frequency",
        "selection_threshold",
        "selected_at_threshold",
        "eligible_for_locked_model",
    ),
    "nineam_lasso_domain_selection.csv": (
        "specification_order",
        "module_spec",
        "is_winning_specification",
        "candidate_order",
        "candidate_label",
        "selection_frequency",
        "selection_threshold",
        "selected_at_threshold",
        "eligible_for_locked_model",
    ),
    "nineam_locked_model_coefficients_hc3.csv": (
        "model_id",
        "term_order",
        "term_label",
        "term_role",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "unit",
        "covariance_estimator",
        "inference_status",
    ),
    "nineam_model_diagnostics.csv": (
        "model_id",
        "diagnostic_order",
        "diagnostic_type",
        "series",
        "bin_order",
        "bin_count",
        "x_value",
        "y_value",
        "y_lower",
        "y_upper",
    ),
}


def _validate_sources(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Validate required aggregate sources and return isolated deep copies.

    The function rejects missing tables or columns before creating an output
    directory. Deep copies ensure sorting and numeric conversion cannot mutate
    caller-owned reporting artifacts.
    """
    if not isinstance(tables, Mapping):
        raise TypeError("tables must be a mapping of CSV names to dataframes")
    validated: dict[str, pd.DataFrame] = {}
    for filename, required_columns in _SOURCE_COLUMNS.items():
        if filename not in tables:
            raise ValueError(f"Missing figure source table: {filename}")
        table = tables[filename]
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{filename} must be a pandas DataFrame")
        missing = [column for column in required_columns if column not in table]
        if missing:
            raise ValueError(
                f"{filename} is missing required columns: {', '.join(missing)}"
            )
        if table.empty:
            raise ValueError(f"{filename} cannot be empty")
        validated[filename] = table.copy(deep=True)
    return validated


def _new_figure(
    *,
    rows: int = 1,
    columns: int = 1,
) -> tuple[Figure, np.ndarray]:
    """Create a fixed 16:9 canvas and always return a flat axes array."""
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.0, 5.625),
        dpi=160,
        squeeze=False,
    )
    return figure, np.asarray(axes, dtype=object).reshape(-1)


def _style_axis(axis: object, *, grid_axis: str = "y") -> None:
    """Apply the shared restrained scientific-axis treatment."""
    axis.grid(axis=grid_axis, color=_COLORS["grid"], linewidth=0.7, alpha=0.8)
    axis.tick_params(colors="#333333")


def _scope_color(scope: str) -> str:
    """Return the fixed accessible color assigned to one member type."""
    return _COLORS.get(scope, _COLORS["neutral"])


def _format_model_name(model_id: str) -> str:
    """Convert internal model identifiers into short presentation labels."""
    labels = {
        "percentage_loss_ols": "Percentage-loss OLS",
        "log_compound_symmetry_gls": "Log-weight CS GLS",
    }
    return labels.get(model_id, model_id.replace("_", " ").title())


def _plot_cohort_flow(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot retained counts through the audited cohort stages."""
    table = tables["nineam_cohort_flow.csv"]
    table = table.loc[table["row_type"].eq("stage")].sort_values(
        "flow_order",
        kind="mergesort",
    )
    if table.empty:
        raise ValueError("nineam_cohort_flow.csv has no stage rows")
    figure, axes = _new_figure()
    axis = axes[0]
    y = np.arange(len(table))
    retained = pd.to_numeric(table["retained_n"], errors="raise").to_numpy()
    axis.barh(y, retained, color="#6C9BCF", edgecolor="#2F5597")
    axis.set_yticks(y, table["stage_label"].astype(str))
    axis.invert_yaxis()
    axis.set_xlabel("Members retained")
    axis.set_title("Cohort flow to the primary analysis", loc="left", weight="bold")
    for position, (_, row) in enumerate(table.iterrows()):
        excluded = int(row["excluded_n"])
        label = f"n={int(row['retained_n']):,}"
        if excluded:
            label += f"  (excluded {excluded:,})"
        axis.text(
            float(row["retained_n"]) + max(retained) * 0.012,
            position,
            label,
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, max(retained) * 1.28)
    _style_axis(axis, grid_axis="x")
    figure.subplots_adjust(left=0.28, right=0.96, top=0.86, bottom=0.15)
    return figure


def _member_type_outcome(
    tables: Mapping[str, pd.DataFrame],
    outcome: str,
) -> pd.DataFrame:
    """Select and stably order the two reportable member-type outcome rows."""
    table = tables["nineam_outcomes_by_member_type.csv"]
    selected = table.loc[
        table["outcome"].eq(outcome) & table["scope"].isin((_GLP1, _COACHING))
    ].sort_values("scope_order", kind="mergesort")
    if set(selected["scope"].astype(str)) != {_GLP1, _COACHING}:
        raise ValueError(f"Outcome {outcome} must contain both member types")
    return selected


def _plot_weight_loss(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot median percentage weight loss with IQR and separate mean markers."""
    table = _member_type_outcome(tables, "percentage_loss")
    figure, axes = _new_figure()
    axis = axes[0]
    x = np.arange(len(table))
    median = pd.to_numeric(table["median"], errors="raise").to_numpy(float)
    q1 = pd.to_numeric(table["q1"], errors="raise").to_numpy(float)
    q3 = pd.to_numeric(table["q3"], errors="raise").to_numpy(float)
    mean = pd.to_numeric(table["mean"], errors="raise").to_numpy(float)
    colors = [_scope_color(str(scope)) for scope in table["scope"]]
    # The vertical intervals are descriptive IQRs, not confidence intervals.
    for position, center, lower, upper, color in zip(
        x, median, q1, q3, colors, strict=True
    ):
        axis.errorbar(
            [position],
            [center],
            yerr=[[center - lower], [upper - center]],
            fmt="none",
            ecolor=color,
            elinewidth=3,
            capsize=8,
        )
    axis.scatter(x, median, color=colors, s=70, marker="o", label="Median (IQR)")
    axis.scatter(
        x,
        mean,
        facecolors="white",
        edgecolors=colors,
        s=75,
        marker="D",
        linewidths=1.8,
        label="Mean",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_xticks(x, table["scope"].astype(str))
    axis.set_ylabel("Percentage weight loss (percentage points)")
    axis.set_title("Weight loss differs by member type", loc="left", weight="bold")
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.25)
    return figure


def _plot_responder_rate(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot 5% response percentages with Wilson confidence intervals."""
    table = _member_type_outcome(tables, "weight_loss_success_5pct")
    if not table["ci_method"].eq("Wilson").all() or not np.allclose(
        pd.to_numeric(table["ci_level"], errors="raise"),
        0.95,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Responder intervals must be 95% Wilson intervals")
    figure, axes = _new_figure()
    axis = axes[0]
    x = np.arange(len(table))
    values = pd.to_numeric(table["percentage"], errors="raise").to_numpy(float)
    lower = pd.to_numeric(table["ci_lower"], errors="raise").to_numpy(float)
    upper = pd.to_numeric(table["ci_upper"], errors="raise").to_numpy(float)
    colors = [_scope_color(str(scope)) for scope in table["scope"]]
    axis.bar(x, values, color=colors, width=0.58)
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        fmt="none",
        ecolor="#222222",
        capsize=7,
        linewidth=1.5,
    )
    for position, value in zip(x, values, strict=True):
        axis.text(position, value + 2.0, f"{value:.1f}%", ha="center", weight="bold")
    axis.set_xticks(x, table["scope"].astype(str))
    axis.set_ylim(0.0, max(100.0, float(np.nanmax(upper)) * 1.15))
    axis.set_ylabel("Members reaching at least 5% weight loss (%)")
    axis.set_title("Clinically relevant weight-loss response", loc="left", weight="bold")
    axis.text(
        0.0,
        -0.20,
        "Whiskers: 95% Wilson confidence intervals",
        transform=axis.transAxes,
        color="#555555",
        fontsize=9,
    )
    _style_axis(axis)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.27)
    return figure


def _plot_engagement(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot four engagement summaries on separate unit-preserving panels."""
    table = tables["nineam_engagement_by_member_type.csv"]
    table = table.loc[table["scope"].isin((_GLP1, _COACHING))].sort_values(
        ["metric_order", "scope_order"],
        kind="mergesort",
    )
    metrics = table[["metric_order", "metric"]].drop_duplicates().head(4)
    if len(metrics) != 4:
        raise ValueError("Engagement figure requires exactly four ordered metrics")
    figure, axes = _new_figure(rows=2, columns=2)
    for axis, metric in zip(axes, metrics["metric"], strict=True):
        rows = table.loc[table["metric"].eq(metric)]
        x = np.arange(len(rows))
        median = pd.to_numeric(rows["median"], errors="raise").to_numpy(float)
        q1 = pd.to_numeric(rows["q1"], errors="raise").to_numpy(float)
        q3 = pd.to_numeric(rows["q3"], errors="raise").to_numpy(float)
        mean = pd.to_numeric(rows["mean"], errors="raise").to_numpy(float)
        colors = [_scope_color(str(scope)) for scope in rows["scope"]]
        for position, center, lower, upper, color in zip(
            x, median, q1, q3, colors, strict=True
        ):
            axis.errorbar(
                [position],
                [center],
                yerr=[[center - lower], [upper - center]],
                fmt="none",
                ecolor=color,
                elinewidth=2.5,
                capsize=5,
            )
        axis.scatter(x, median, color=colors, s=42)
        axis.scatter(x, mean, color=colors, marker="x", s=45, linewidths=1.8)
        axis.set_xticks(x, ["GLP-1", "Coaching"])
        axis.set_title(str(rows.iloc[0]["metric_label"]), loc="left", fontsize=10)
        axis.set_ylabel(str(rows.iloc[0]["unit"]), fontsize=8)
        _style_axis(axis)
    figure.suptitle("Engagement patterns by member type", x=0.08, ha="left", weight="bold")
    figure.text(0.08, 0.02, "Points show medians (IQR); x marks show means.", fontsize=8.5, color="#555555")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.14, wspace=0.30, hspace=0.48)
    return figure


def _plot_modules(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot observed mean completion proportions across curriculum domains."""
    table = tables["nineam_modules_by_member_type.csv"]
    table = table.loc[table["scope"].isin((_GLP1, _COACHING))].sort_values(
        ["module_order", "scope_order"],
        kind="mergesort",
    )
    modules = table[["module_order", "module_variable", "module_label"]].drop_duplicates()
    figure, axes = _new_figure()
    axis = axes[0]
    x = np.arange(len(modules))
    width = 0.36
    for offset, scope in ((-width / 2, _GLP1), (width / 2, _COACHING)):
        rows = table.loc[table["scope"].eq(scope)].set_index("module_variable")
        values = [
            100.0 * float(rows.loc[variable, "mean_completion_proportion"])
            for variable in modules["module_variable"]
        ]
        axis.bar(
            x + offset,
            values,
            width=width,
            color=_scope_color(scope),
            label="GLP-1" if scope == _GLP1 else "Coaching",
        )
    axis.set_xticks(x, modules["module_label"].astype(str), rotation=18, ha="right")
    axis.set_ylabel("Mean observed completion proportion (%)")
    axis.set_title("Observed module completion by member type", loc="left", weight="bold")
    axis.legend(frameon=False)
    axis.text(
        0.0,
        -0.27,
        "Extension-module availability is not verifiable from the extract; zeros reflect observed records.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    _style_axis(axis)
    figure.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.34)
    return figure


def _plot_base_performance(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot paired-fold predictive RMSE and MAE summaries with SD whiskers."""
    table = tables["nineam_base_model_comparison_summary.csv"].sort_values(
        ["metric_order", "model_order"],
        kind="mergesort",
    )
    metrics = table[["metric_order", "metric"]].drop_duplicates()
    figure, axes = _new_figure(rows=1, columns=len(metrics))
    for axis, metric in zip(axes, metrics["metric"], strict=True):
        rows = table.loc[table["metric"].eq(metric)]
        x = np.arange(len(rows))
        values = pd.to_numeric(rows["mean_score"], errors="raise").to_numpy(float)
        spread = pd.to_numeric(
            rows["standard_deviation"], errors="raise"
        ).to_numpy(float)
        colors = [
            _COLORS["selected"] if bool(winner) else "#8C8C8C"
            for winner in rows["is_winner"]
        ]
        bars = axis.bar(
            x,
            values,
            yerr=spread,
            capsize=6,
            color=colors,
            edgecolor="#333333",
            width=0.62,
        )
        for bar, winner in zip(bars, rows["is_winner"], strict=True):
            if bool(winner):
                bar.set_hatch("///")
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + float(np.nanmax(spread)) * 1.2,
                    "Winner",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    weight="bold",
                )
        axis.set_xticks(
            x,
            [_format_model_name(str(model)) for model in rows["model_id"]],
            rotation=14,
            ha="right",
        )
        axis.set_ylabel(str(rows.iloc[0]["metric_unit"]))
        axis.set_title(str(metric).upper(), loc="left")
        _style_axis(axis)
    figure.suptitle(
        "Held-out raw last-weight prediction performance",
        x=0.08,
        ha="left",
        weight="bold",
    )
    figure.text(0.08, 0.02, "Bars are fold means; whiskers are fold-score standard deviations. Lower is better.", fontsize=8.5, color="#555555")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.31, wspace=0.30)
    return figure


def _plot_lasso_stability(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot selection frequencies for both separate module specifications."""
    mean = tables["nineam_lasso_mean_selection.csv"].copy()
    domains = tables["nineam_lasso_domain_selection.csv"].copy()
    table = pd.concat((mean, domains), ignore_index=True).sort_values(
        ["specification_order", "candidate_order"],
        kind="mergesort",
    )
    labels = [
        f"[{spec}] {label}"
        for spec, label in zip(
            table["module_spec"],
            table["candidate_label"],
            strict=True,
        )
    ]
    frequency = pd.to_numeric(
        table["selection_frequency"], errors="raise"
    ).to_numpy(float)
    colors = [
        _COLORS["selected"] if bool(eligible) else "#9E9E9E"
        for eligible in table["eligible_for_locked_model"]
    ]
    figure, axes = _new_figure()
    axis = axes[0]
    y = np.arange(len(table))
    bars = axis.barh(y, frequency, color=colors, edgecolor="#555555")
    for bar, eligible in zip(
        bars,
        table["eligible_for_locked_model"],
        strict=True,
    ):
        if bool(eligible):
            bar.set_hatch("///")
            axis.text(
                min(float(bar.get_width()) + 0.015, 0.96),
                bar.get_y() + bar.get_height() / 2,
                "LOCKED",
                va="center",
                fontsize=7.5,
                weight="bold",
            )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    threshold_values = pd.to_numeric(
        table["selection_threshold"], errors="raise"
    ).unique()
    if len(threshold_values) != 1:
        raise ValueError("LASSO tables must share one selection threshold")
    axis.axvline(
        float(threshold_values[0]),
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"Stability threshold ({threshold_values[0]:.2f})",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Selection frequency across resamples")
    axis.set_title("LASSO stability by candidate specification", loc="left", weight="bold")
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis, grid_axis="x")
    # Reserve enough canvas for the longest domain label; fixed-width slide
    # exports otherwise crop the physical-activity label at the left edge.
    figure.subplots_adjust(left=0.42, right=0.97, top=0.86, bottom=0.16)
    return figure


def _plot_locked_coefficients(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot primary locked-model HC3 estimates and 95% confidence intervals."""
    table = tables["nineam_locked_model_coefficients_hc3.csv"]
    table = table.loc[
        table["model_id"].eq("locked_percentage_loss_primary")
        & ~table["term_role"].eq("intercept")
    ].sort_values("term_order", kind="mergesort")
    if table.empty:
        raise ValueError("Locked coefficient table has no primary non-intercept rows")
    if (
        not table["covariance_estimator"].eq("HC3").all()
        or not table["inference_status"].eq("conditional_exploratory").all()
        or not np.allclose(
            pd.to_numeric(table["ci_level"], errors="raise"),
            0.95,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(
            "Locked coefficient intervals must be 95% HC3 conditional-exploratory intervals"
        )
    estimate = pd.to_numeric(table["estimate"], errors="raise").to_numpy(float)
    lower = pd.to_numeric(table["ci_lower"], errors="raise").to_numpy(float)
    upper = pd.to_numeric(table["ci_upper"], errors="raise").to_numpy(float)
    figure, axes = _new_figure()
    axis = axes[0]
    y = np.arange(len(table))
    colors = [
        _COLORS["selected"]
        if role == "selected_candidate"
        else _COLORS["neutral"]
        for role in table["term_role"]
    ]
    for center, position, left, right, color in zip(
        estimate, y, lower, upper, colors, strict=True
    ):
        axis.errorbar(
            [center],
            [position],
            xerr=[[center - left], [right - center]],
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=5,
        )
    for center, position, color, role in zip(
        estimate,
        y,
        colors,
        table["term_role"],
        strict=True,
    ):
        axis.scatter(
            [center],
            [position],
            color=color,
            marker="D" if role == "selected_candidate" else "o",
            s=60,
            zorder=3,
        )
    axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    labels = [
        # Reporting tables retain machine-readable unit codes; figures convert
        # underscores to spaces so leadership-facing labels read naturally.
        f"{label}\n({str(unit).replace('_', ' ')})"
        for label, unit in zip(table["term_label"], table["unit"], strict=True)
    ]
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Estimated percentage-point difference")
    axis.set_title("Locked percentage-loss model coefficients (HC3)", loc="left", weight="bold")
    axis.text(
        0.0,
        -0.16,
        "Whiskers: 95% HC3 confidence intervals; inference is conditional and exploratory.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    _style_axis(axis, grid_axis="x")
    figure.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.22)
    return figure


def _plot_diagnostics(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot aggregate residual, Q-Q, leverage, and Cook's-distance summaries."""
    table = tables["nineam_model_diagnostics.csv"]
    table = table.loc[
        table["model_id"].eq("locked_percentage_loss_primary")
        & table["diagnostic_type"].isin(
            ("residuals_vs_fitted", "normal_qq", "leverage", "cooks_distance")
        )
        & table["series"].isin(("binned_mean", "quantile_pair"))
    ].sort_values(["diagnostic_order", "bin_order"], kind="mergesort")
    figure, axes = _new_figure(rows=2, columns=2)
    labels = {
        "residuals_vs_fitted": "Residuals vs fitted",
        "normal_qq": "Normal Q-Q",
        "leverage": "Leverage",
        "cooks_distance": "Cook's distance",
    }
    for axis, diagnostic_type in zip(axes, labels, strict=True):
        rows = table.loc[table["diagnostic_type"].eq(diagnostic_type)]
        if rows.empty:
            raise ValueError(f"Diagnostics table has no {diagnostic_type} plot rows")
        x = pd.to_numeric(rows["x_value"], errors="raise").to_numpy(float)
        y = pd.to_numeric(rows["y_value"], errors="raise").to_numpy(float)
        axis.plot(x, y, color="#2F5597", marker="o", linewidth=1.4)
        if diagnostic_type == "residuals_vs_fitted":
            lower = pd.to_numeric(rows["y_lower"], errors="coerce").to_numpy(float)
            upper = pd.to_numeric(rows["y_upper"], errors="coerce").to_numpy(float)
            if np.isfinite(lower).all() and np.isfinite(upper).all():
                axis.fill_between(x, lower, upper, color="#9DC3E6", alpha=0.35)
            axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
            axis.set_xlabel("Mean fitted value")
            axis.set_ylabel("Mean residual")
        elif diagnostic_type == "normal_qq":
            diagonal_min = float(min(np.nanmin(x), np.nanmin(y)))
            diagonal_max = float(max(np.nanmax(x), np.nanmax(y)))
            axis.plot(
                [diagonal_min, diagonal_max],
                [diagonal_min, diagonal_max],
                color="#777777",
                linestyle="--",
                linewidth=0.9,
            )
            axis.set_xlabel("Theoretical normal quantile")
            axis.set_ylabel("Observed residual quantile")
        else:
            axis.set_xlabel("Aggregate quantile/bin")
            axis.set_ylabel(labels[diagnostic_type])
        axis.set_title(labels[diagnostic_type], loc="left", fontsize=10)
        _style_axis(axis)
    figure.suptitle("Locked-model diagnostic summaries", x=0.08, ha="left", weight="bold")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.12, wspace=0.30, hspace=0.43)
    return figure


_PLOTTERS: tuple[
    tuple[str, Callable[[Mapping[str, pd.DataFrame]], Figure]], ...
] = (
    ("nineam_cohort_flow", _plot_cohort_flow),
    ("nineam_weight_loss_by_member_type", _plot_weight_loss),
    ("nineam_responder_rate_by_member_type", _plot_responder_rate),
    ("nineam_engagement_patterns", _plot_engagement),
    ("nineam_module_completion", _plot_modules),
    ("nineam_base_model_performance", _plot_base_performance),
    ("nineam_lasso_stability", _plot_lasso_stability),
    ("nineam_locked_model_coefficients", _plot_locked_coefficients),
    ("nineam_model_diagnostics", _plot_diagnostics),
)


def write_analysis_figures(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write nine aggregate-table figures in deterministic PNG and SVG forms.

    Args:
        tables: Validated scientific reporting tables keyed by CSV filename.
        output_dir: Dedicated figure directory; it is created when absent.

    Returns:
        Paths in stable figure order, with PNG followed by SVG for each stem.

    Side effects:
        Creates the destination and writes eighteen image files. Source tables
        are copied and never modified.

    Statistical intent:
        Encodes already-computed aggregate estimates, descriptive spreads, and
        confidence intervals without recomputing or changing any estimand.
    """
    validated = _validate_sources(tables)
    destination = Path(output_dir).resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError("output_dir must be a directory")
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with matplotlib.rc_context(_RC_PARAMS):
        for stem, plotter in _PLOTTERS:
            figure = plotter(validated)
            try:
                png_path = destination / f"{stem}.png"
                svg_path = destination / f"{stem}.svg"
                # Explicit metadata prevents timestamps from making otherwise
                # identical analytical runs produce different image bytes.
                figure.savefig(
                    png_path,
                    format="png",
                    dpi=160,
                    metadata={"Software": "nineam-health-analysis"},
                )
                figure.savefig(
                    svg_path,
                    format="svg",
                    metadata={
                        "Date": None,
                        "Creator": "nineam-health-analysis",
                        "Title": stem,
                    },
                )
                written.extend((png_path, svg_path))
            finally:
                # Closing every canvas prevents memory accumulation during CLI
                # reruns and keeps global pyplot state out of subsequent plots.
                plt.close(figure)
    return tuple(written)
