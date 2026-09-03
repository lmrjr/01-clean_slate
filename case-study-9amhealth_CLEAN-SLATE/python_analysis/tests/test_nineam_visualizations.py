"""Test deterministic scientific figures built only from aggregate tables."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from nineam_health_analysis.nineam_visualizations import (
    _RC_PARAMS,
    FIGURE_STEMS,
    _plot_lasso_stability,
    _plot_locked_coefficients,
    write_analysis_figures,
)


def _figure_source_tables() -> dict[str, pd.DataFrame]:
    """Return the smallest complete aggregate input for all nine figures."""
    scopes = ("Active GLP-1 for Weight-loss", "Coaching Only")
    outcomes: list[dict[str, object]] = []
    engagement: list[dict[str, object]] = []
    modules: list[dict[str, object]] = []
    for scope_order, scope in enumerate(scopes, start=2):
        shift = 4.0 if scope.startswith("Active") else 0.0
        outcomes.extend(
            [
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "outcome": "percentage_loss",
                    "outcome_label": "Percentage weight loss",
                    "unit": "percentage points",
                    "mean": 3.0 + shift,
                    "median": 2.8 + shift,
                    "q1": 1.0 + shift,
                    "q3": 4.5 + shift,
                    "percentage": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "ci_level": float("nan"),
                    "ci_method": "",
                },
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "outcome": "weight_loss_success_5pct",
                    "outcome_label": "At least 5% weight loss",
                    "unit": "percent",
                    "mean": float("nan"),
                    "median": float("nan"),
                    "q1": float("nan"),
                    "q3": float("nan"),
                    "percentage": 70.0 if shift else 15.0,
                    "ci_lower": 62.0 if shift else 10.0,
                    "ci_upper": 77.0 if shift else 22.0,
                    "ci_level": 0.95,
                    "ci_method": "Wilson",
                },
            ]
        )
        for metric_order, (metric, label, value) in enumerate(
            (
                ("engagement_breadth", "Activity breadth", 9.0 + shift),
                ("engagement_volume_repeatable", "Repeatable volume", 45.0 + shift),
                (
                    "engagement_volume_repeatable_rate",
                    "Repeatable volume per tenure day",
                    0.25 + shift / 100.0,
                ),
                (
                    "engagement_volume_weight_days_rate",
                    "Repeatable volume per observed weight day",
                    0.30 + shift / 100.0,
                ),
            ),
            start=1,
        ):
            engagement.append(
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "metric_order": metric_order,
                    "metric": metric,
                    "metric_label": label,
                    "unit": "events" if metric_order < 3 else "events/day",
                    "mean": value,
                    "median": value * 0.9,
                    "q1": value * 0.6,
                    "q3": value * 1.2,
                    "minimum": 0.0,
                    "maximum": value * 2.0,
                }
            )
        for module_order, (variable, label) in enumerate(
            (
                ("module_mean", "Four-domain mean"),
                ("module_core", "Core"),
                ("module_mindset", "Mindset"),
                ("module_nutrition", "Nutrition"),
                ("module_physical_activity", "Physical activity"),
            ),
            start=1,
        ):
            modules.append(
                {
                    "scope_order": scope_order,
                    "scope": scope,
                    "module_order": module_order,
                    "module_variable": variable,
                    "module_label": label,
                    "mean_completion_proportion": min(
                        1.0,
                        0.1 * module_order + shift / 10.0,
                    ),
                    "support_scope": (
                        "GLP-1-only observed support"
                        if module_order >= 3
                        else "both member types"
                    ),
                }
            )

    diagnostics: list[dict[str, object]] = []
    for diagnostic_order, diagnostic_type in enumerate(
        ("residuals_vs_fitted", "normal_qq", "leverage", "cooks_distance"),
        start=1,
    ):
        for bin_order in (1, 2):
            diagnostics.append(
                {
                    "model_id": "locked_percentage_loss_primary",
                    "diagnostic_order": diagnostic_order,
                    "diagnostic_type": diagnostic_type,
                    "row_type": "binned",
                    "series": (
                        "binned_mean"
                        if diagnostic_type == "residuals_vs_fitted"
                        else "quantile_pair"
                    ),
                    "bin_order": bin_order,
                    "bin_count": 10,
                    "x_value": float(bin_order),
                    "y_value": (-0.2 + 0.3 * bin_order),
                    "y_lower": -0.5 + 0.2 * bin_order,
                    "y_upper": 0.3 + 0.2 * bin_order,
                    "metric": "",
                    "value": float("nan"),
                    "threshold": float("nan"),
                    "flag": False,
                    "status": "review",
                }
            )

    lasso_common = {
        "specification_order": [1, 1],
        "is_winning_specification": [True, True],
        "candidate_order": [1, 2],
        "candidate_label": ["Engagement breadth", "Sex: male vs female"],
        "selection_frequency": [0.90, 0.55],
        "selection_threshold": [0.75, 0.75],
        "selected_at_threshold": [True, False],
        "eligible_for_locked_model": [True, False],
    }
    return {
        "nineam_cohort_flow.csv": pd.DataFrame(
            {
                "flow_order": [1, 2, 3],
                "row_type": ["stage", "stage", "stage"],
                "stage_id": ["enrolled", "complete_pair", "primary"],
                "stage_label": ["Enrolled", "Complete pair", "Primary analysis"],
                "starting_n": [865, 865, 633],
                "excluded_n": [0, 232, 99],
                "retained_n": [865, 633, 534],
            }
        ),
        "nineam_outcomes_by_member_type.csv": pd.DataFrame(outcomes),
        "nineam_engagement_by_member_type.csv": pd.DataFrame(engagement),
        "nineam_modules_by_member_type.csv": pd.DataFrame(modules),
        "nineam_base_model_comparison_summary.csv": pd.DataFrame(
            {
                "model_order": [1, 1, 2, 2],
                "model_id": ["percentage_loss_ols"] * 2
                + ["log_compound_symmetry_gls"] * 2,
                "metric_order": [1, 2, 1, 2],
                "metric": ["rmse", "mae", "rmse", "mae"],
                "metric_unit": ["lb"] * 4,
                "mean_score": [9.0, 6.0, 10.0, 7.0],
                "standard_deviation": [0.5, 0.4, 0.7, 0.5],
                "minimum_score": [8.4, 5.5, 9.1, 6.3],
                "maximum_score": [9.7, 6.6, 11.0, 7.8],
                "is_winner": [True, True, False, False],
            }
        ),
        "nineam_lasso_mean_selection.csv": pd.DataFrame(
            {"module_spec": ["mean", "mean"], **lasso_common}
        ),
        "nineam_lasso_domain_selection.csv": pd.DataFrame(
            {
                "module_spec": ["domains", "domains"],
                **{
                    **lasso_common,
                    "specification_order": [2, 2],
                    "is_winning_specification": [False, False],
                    "candidate_label": ["Core modules", "Nutrition modules"],
                    "selection_frequency": [0.65, 0.35],
                    "eligible_for_locked_model": [False, False],
                },
            }
        ),
        "nineam_locked_model_coefficients_hc3.csv": pd.DataFrame(
            {
                "model_id": ["locked_percentage_loss_primary"] * 3,
                "term_order": [1, 2, 3],
                "term": ["Intercept", "first_weight", "engagement_breadth"],
                "term_label": ["Intercept", "Baseline weight", "Engagement breadth"],
                "term_role": ["intercept", "base", "selected_candidate"],
                "estimate": [-1.0, 0.02, 0.30],
                "ci_lower": [-3.0, 0.01, 0.10],
                "ci_upper": [1.0, 0.03, 0.50],
                "ci_level": [0.95, 0.95, 0.95],
                "unit": ["percentage points", "percentage points per lb", "percentage points per type"],
                "p_value": [0.30, 0.01, 0.004],
                "covariance_estimator": ["HC3", "HC3", "HC3"],
                "inference_status": ["conditional_exploratory"] * 3,
            }
        ),
        "nineam_model_diagnostics.csv": pd.DataFrame(diagnostics),
    }


class AnalysisVisualizationsTestCase(unittest.TestCase):
    """Exercise figure completeness, validation, and byte determinism."""

    def test_writer_creates_png_and_svg_for_every_scientific_figure(self) -> None:
        """Create exactly two formats for each approved figure stem."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "figures"
            written = write_analysis_figures(_figure_source_tables(), output_dir)

            self.assertEqual(len(written), 2 * len(FIGURE_STEMS))
            self.assertEqual(
                {path.name for path in written},
                {
                    f"{stem}.{extension}"
                    for stem in FIGURE_STEMS
                    for extension in ("png", "svg")
                },
            )
            self.assertTrue(all(path.stat().st_size > 1_000 for path in written))

    def test_svg_and_png_outputs_are_byte_deterministic(self) -> None:
        """Render identical tables twice without timestamps or random IDs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_analysis_figures(_figure_source_tables(), root / "first")
            second = write_analysis_figures(_figure_source_tables(), root / "second")

            first_bytes = {path.name: path.read_bytes() for path in first}
            second_bytes = {path.name: path.read_bytes() for path in second}
            self.assertEqual(first_bytes, second_bytes)

    def test_writer_validates_sources_and_does_not_mutate_tables(self) -> None:
        """Fail on missing columns and preserve all caller-owned dataframes."""
        tables = _figure_source_tables()
        snapshots = {name: table.copy(deep=True) for name, table in tables.items()}
        with tempfile.TemporaryDirectory() as temporary_directory:
            write_analysis_figures(tables, Path(temporary_directory) / "valid")
            invalid = dict(tables)
            invalid["nineam_cohort_flow.csv"] = invalid[
                "nineam_cohort_flow.csv"
            ].drop(columns="retained_n")
            with self.assertRaisesRegex(ValueError, "retained_n"):
                write_analysis_figures(
                    invalid,
                    Path(temporary_directory) / "invalid",
                )

        for name, snapshot in snapshots.items():
            pd.testing.assert_frame_equal(tables[name], snapshot)

    def test_writer_rejects_mislabeled_interval_provenance(self) -> None:
        """Prevent Wilson and HC3 labels from being attached to other intervals."""
        tables = _figure_source_tables()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_wilson = dict(tables)
            invalid_wilson["nineam_outcomes_by_member_type.csv"] = tables[
                "nineam_outcomes_by_member_type.csv"
            ].copy()
            responder = invalid_wilson[
                "nineam_outcomes_by_member_type.csv"
            ]["outcome"].eq("weight_loss_success_5pct")
            invalid_wilson["nineam_outcomes_by_member_type.csv"].loc[
                responder,
                "ci_method",
            ] = "Wald"
            with self.assertRaisesRegex(ValueError, "Wilson"):
                write_analysis_figures(invalid_wilson, root / "wilson")

            invalid_hc3 = dict(tables)
            invalid_hc3["nineam_locked_model_coefficients_hc3.csv"] = tables[
                "nineam_locked_model_coefficients_hc3.csv"
            ].copy()
            invalid_hc3["nineam_locked_model_coefficients_hc3.csv"].loc[
                :, "covariance_estimator"
            ] = "nonrobust"
            with self.assertRaisesRegex(ValueError, "HC3"):
                write_analysis_figures(invalid_hc3, root / "hc3")

    def test_long_scientific_labels_remain_inside_the_canvas(self) -> None:
        """Keep long LASSO and coefficient labels visible in slide-ready exports."""
        tables = _figure_source_tables()
        domain = tables["nineam_lasso_domain_selection.csv"]
        domain.loc[domain.index[-1], "candidate_label"] = (
            "Physical-activity module completion proportion"
        )
        coefficients = tables["nineam_locked_model_coefficients_hc3.csv"]
        coefficients.loc[coefficients.index[-1], "term_label"] = (
            "Active GLP-1 for weight loss versus coaching only"
        )
        coefficients.loc[coefficients.index[-1], "unit"] = "percentage_points"

        with matplotlib.rc_context(_RC_PARAMS):
            figures = (
                _plot_lasso_stability(tables),
                _plot_locked_coefficients(tables),
            )
            try:
                for figure in figures:
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    for axis in figure.axes:
                        for label in axis.get_yticklabels():
                            self.assertGreaterEqual(
                                label.get_window_extent(renderer).x0,
                                0.0,
                                msg=f"Clipped label: {label.get_text()}",
                            )
                coefficient_labels = [
                    label.get_text()
                    for label in figures[1].axes[0].get_yticklabels()
                ]
                self.assertTrue(
                    all("_" not in label for label in coefficient_labels)
                )
            finally:
                for figure in figures:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
