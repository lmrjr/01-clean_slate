"""Tests for the locked post-LASSO percentage-loss model."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nineam_health_analysis.nineam_final_model import (
    COEFFICIENT_COLUMNS,
    DIAGNOSTIC_SUMMARY_COLUMNS,
    FIT_STATISTICS_COLUMNS,
    LockedCandidateSelection,
    choose_base_model_winner,
    fit_locked_percentage_model,
    select_locked_candidates,
)


def _members() -> pd.DataFrame:
    """Build a full-rank fixture with literal percentage-loss effects."""
    first_weight = np.array([100, 110, 120, 130, 105, 115, 125, 135], dtype=float)
    active = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    volume = np.array([1, 4, 2, 5, 3, 6, 7, 8], dtype=float)
    sex = np.array(
        ["FEMALE", "MALE", "FEMALE", "MALE", "FEMALE", "MALE", "FEMALE", "MALE"]
    )
    days = np.array([35, 50, 67, 80, 35, 48, 71, 80], dtype=float)
    residual = np.array([0.8, -0.5, -0.7, 0.4, -0.6, 0.9, 0.5, -0.8])
    percentage_loss = 4 + 0.05 * first_weight + 2 * active + 0.3 * volume + residual
    return pd.DataFrame(
        {
            "member_id": [f"member_{index}" for index in range(8)],
            "member_type": np.where(active == 1, "Active GLP-1 for Weight-loss", "Coaching Only"),
            "first_weight": first_weight,
            "last_weight": first_weight * (1 - percentage_loss / 100),
            "engagement_volume_repeatable": volume,
            "engagement_volume_repeatable_rate": volume / days,
            "engagement_breadth": volume + 1,
            "tenure_days": days,
            "weight_days": days,
            "module_mean": volume / 10,
            "module_core": volume / 11,
            "module_mindset": volume / 12,
            "module_nutrition": volume / 13,
            "module_physical_activity": volume / 14,
            "sex": sex,
        }
    )


class BaseModelWinnerTestCase(unittest.TestCase):
    """Exercise prespecified held-out prediction ranking."""

    def test_choose_winner_uses_mean_rmse_then_mean_mae(self) -> None:
        """A model with lower mean RMSE wins regardless of row order or likelihood."""
        comparison = pd.DataFrame(
            {
                "model": ["percentage_loss_ols", "percentage_loss_ols", "log_compound_symmetry_gls", "log_compound_symmetry_gls"],
                "rmse": [2.0, 4.0, 4.0, 4.0],
                "mae": [2.0, 3.0, 2.0, 2.0],
                "aic": [-1000.0, -1000.0, -1.0, -1.0],
            }
        )
        self.assertEqual(choose_base_model_winner(comparison.sample(frac=1, random_state=3)), "percentage_loss_ols")

    def test_choose_winner_does_not_treat_a_near_tie_as_an_exact_tie(self) -> None:
        """Any strictly lower mean RMSE wins before the prespecified tie-break."""
        comparison = pd.DataFrame(
            {
                "model": ["log_compound_symmetry_gls", "percentage_loss_ols"],
                "rmse": [3.0, 3.0 + 5e-13],
                "mae": [2.0, 2.0 + 5e-13],
            }
        )
        self.assertEqual(
            choose_base_model_winner(comparison), "log_compound_symmetry_gls"
        )

    def test_choose_winner_prefers_percentage_model_only_on_exact_metric_tie(
        self,
    ) -> None:
        """An exact tie on mean RMSE and mean MAE uses the primary-outcome rule."""
        comparison = pd.DataFrame(
            {
                "model": ["log_compound_symmetry_gls", "percentage_loss_ols"],
                "rmse": [3.0, 3.0],
                "mae": [2.0, 2.0],
            }
        )
        self.assertEqual(choose_base_model_winner(comparison), "percentage_loss_ols")


class LockedCandidateSelectionTestCase(unittest.TestCase):
    """Exercise module-specific post-LASSO candidate locking."""

    def test_selects_lower_cv_specification_without_unioning_candidates(self) -> None:
        """The winning domain spec retains only canonical stable domain terms."""
        mean = pd.DataFrame(
            {
                "module_spec": ["mean", "mean"],
                "candidate": ["engagement_breadth", "module_mean"],
                "candidate_order": [3, 5],
                "cv_mean_mse": [4.0, 4.0],
                "selection_threshold": [0.75, 0.75],
                "selection_frequency": [0.9, 0.8],
            }
        )
        domains = pd.DataFrame(
            {
                "module_spec": ["domains", "domains", "domains"],
                "candidate": ["module_nutrition", "engagement_breadth", "sex[MALE]"],
                "candidate_order": [8, 3, 10],
                "cv_mean_mse": [3.0, 3.0, 3.0],
                "selection_threshold": [0.75, 0.75, 0.75],
                "selection_frequency": [0.8, 0.9, 0.7],
            }
        )
        selected = select_locked_candidates(mean, domains, threshold=0.75)
        self.assertEqual(selected.module_spec, "domains")
        self.assertEqual(selected.candidates, ("engagement_breadth", "module_nutrition"))
        self.assertEqual(selected.selection_frequencies, {"engagement_breadth": 0.9, "module_nutrition": 0.8})

    def test_selects_mean_on_tolerance_tie_and_supports_base_only_result(self) -> None:
        """A tied module CV score favors mean and permits no stable candidates."""
        mean = pd.DataFrame({"module_spec": ["mean"], "candidate": ["module_mean"], "candidate_order": [5], "cv_mean_mse": [3.0], "selection_threshold": [0.75], "selection_frequency": [0.74]})
        domains = pd.DataFrame({"module_spec": ["domains"], "candidate": ["module_core"], "candidate_order": [6], "cv_mean_mse": [3.0 + 5e-13], "selection_threshold": [0.75], "selection_frequency": [0.2]})
        selected = select_locked_candidates(mean, domains, threshold=0.75)
        self.assertEqual(selected.module_spec, "mean")
        self.assertEqual(selected.candidates, ())

    def test_rejects_inconsistent_within_specification_metadata(self) -> None:
        """Contradictory selected-penalty CV evidence cannot be locked."""
        invalid = pd.DataFrame({"module_spec": ["mean", "mean"], "candidate": ["module_mean", "sex[MALE]"], "candidate_order": [5, 10], "cv_mean_mse": [3.0, 4.0], "selection_threshold": [0.75, 0.75], "selection_frequency": [0.8, 0.8]})
        with self.assertRaisesRegex(ValueError, "consistent"):
            select_locked_candidates(invalid, invalid, threshold=0.75)

    def test_rejects_threshold_mismatch_and_noncanonical_candidate_order(self) -> None:
        """The lock must use the requested threshold and prespecified term order."""
        mean = pd.DataFrame(
            {
                "module_spec": ["mean"],
                "candidate": ["module_mean"],
                "candidate_order": [5],
                "cv_mean_mse": [3.0],
                "selection_threshold": [0.80],
                "selection_frequency": [0.90],
            }
        )
        domains = pd.DataFrame(
            {
                "module_spec": ["domains"],
                "candidate": ["module_core"],
                "candidate_order": [6],
                "cv_mean_mse": [4.0],
                "selection_threshold": [0.75],
                "selection_frequency": [0.90],
            }
        )
        with self.assertRaisesRegex(ValueError, "requested threshold"):
            select_locked_candidates(mean, domains, threshold=0.75)

        mean.loc[0, "selection_threshold"] = 0.75
        mean.loc[0, "candidate_order"] = 99
        with self.assertRaisesRegex(ValueError, "canonical"):
            select_locked_candidates(mean, domains, threshold=0.75)

    def test_locked_selection_validates_spec_frequency_order_and_immutability(
        self,
    ) -> None:
        """Direct construction cannot bypass the LASSO locking contract."""
        with self.assertRaisesRegex(ValueError, "threshold"):
            LockedCandidateSelection(
                "mean", ("module_mean",), 3.0, 0.75, {"module_mean": 0.74}
            )
        with self.assertRaisesRegex(ValueError, "mean specification"):
            LockedCandidateSelection(
                "mean", ("module_core",), 3.0, 0.75, {"module_core": 0.90}
            )
        with self.assertRaisesRegex(ValueError, "domains specification"):
            LockedCandidateSelection(
                "domains", ("module_mean",), 3.0, 0.75, {"module_mean": 0.90}
            )
        with self.assertRaisesRegex(ValueError, "canonical"):
            LockedCandidateSelection(
                "mean",
                ("sex[MALE]", "engagement_breadth"),
                3.0,
                0.75,
                {"sex[MALE]": 0.90, "engagement_breadth": 0.90},
            )
        with self.assertRaisesRegex(ValueError, "canonical sex"):
            LockedCandidateSelection(
                "mean", ("sex[M]",), 3.0, 0.75, {"sex[M]": 0.90}
            )

        frequencies = {"module_mean": 0.90}
        selection = LockedCandidateSelection(
            "mean", ("module_mean",), 3.0, 0.75, frequencies
        )
        frequencies["module_mean"] = 0.10
        self.assertEqual(selection.selection_frequencies["module_mean"], 0.90)
        with self.assertRaises(TypeError):
            selection.selection_frequencies["module_mean"] = 0.20


class LockedModelTestCase(unittest.TestCase):
    """Exercise HC3 locked primary and duration-sensitivity refits."""

    def setUp(self) -> None:
        self.members = _members()
        self.selection = LockedCandidateSelection(
            module_spec="mean",
            candidates=("engagement_volume_repeatable", "sex[MALE]"),
            cv_mean_mse=4.0,
            selection_threshold=0.75,
            selection_frequencies={"engagement_volume_repeatable": 0.9, "sex[MALE]": 0.8},
        )

    def test_locked_model_uses_coaching_reference_hc3_and_closed_schemas(self) -> None:
        """Primary refit reports Coaching contrasts and real HC3 literal SEs."""
        result = fit_locked_percentage_model(self.members, self.selection, "percentage_loss_ols")
        coefficient = result.coefficient_table.query(
            "model_id == 'locked_percentage_loss_primary'"
        ).set_index("term")
        self.assertIn("member_type[Active GLP-1 for Weight-loss]", coefficient.index)
        self.assertNotIn("member_type[Coaching Only]", coefficient.index)
        self.assertEqual(list(result.coefficient_table.columns), list(COEFFICIENT_COLUMNS))
        self.assertEqual(list(result.fit_statistics.columns), list(FIT_STATISTICS_COLUMNS))
        self.assertEqual(list(result.diagnostic_summary.columns), list(DIAGNOSTIC_SUMMARY_COLUMNS))
        self.assertEqual(set(coefficient["covariance_estimator"]), {"HC3"})
        self.assertEqual(set(coefficient["inference_status"]), {"conditional_exploratory"})
        self.assertAlmostEqual(coefficient.loc["engagement_volume_repeatable", "standard_error"], 1.425886, places=5)
        self.assertTrue(result.fit_statistics["standardized_design_condition_number"].gt(0).all())
        self.assertEqual(
            set(result.coefficient_table["term_role"]),
            {"intercept", "base", "selected_candidate", "sensitivity_adjustment"},
        )
        self.assertTrue(
            result.coefficient_table["reference_distribution"].eq("normal").all()
        )
        self.assertTrue(result.coefficient_table["degrees_of_freedom"].isna().all())
        sex_row = coefficient.loc["sex[MALE]"]
        self.assertEqual(sex_row["contrast"], "MALE versus FEMALE")
        self.assertEqual(sex_row["reference"], "FEMALE")
        self.assertEqual(sex_row["term_label"], "Male versus female")
        self.assertEqual(
            coefficient.loc["first_weight", "unit"], "percentage_points_per_lb"
        )

    def test_locked_model_includes_common_support_duration_sensitivity_and_no_ids(self) -> None:
        """Sensitivity appends centered duration terms on its common-support sample."""
        result = fit_locked_percentage_model(self.members, self.selection, "percentage_loss_ols")
        self.assertEqual(set(result.fit_statistics["model_id"]), {"locked_percentage_loss_primary", "locked_percentage_loss_duration_sensitivity"})
        sensitivity = result.coefficient_table.query("model_id == 'locked_percentage_loss_duration_sensitivity'")
        self.assertIn("weight_days", set(sensitivity["term"]))
        self.assertIn("weight_days_squared", set(sensitivity["term"]))
        self.assertTrue(result.fit_statistics.loc[result.fit_statistics["model_id"].eq("locked_percentage_loss_duration_sensitivity"), "duration_adjustment"].eq("centered_linear_quadratic_weight_days_common_support").all())
        self.assertNotIn("member_id", result.diagnostic_points.columns)
        self.assertTrue({"leverage", "standardized_residual", "cooks_distance"}.issubset(result.diagnostic_points.columns))
        allowed_types = {
            "residuals_vs_fitted",
            "normal_qq",
            "leverage",
            "cooks_distance",
            "global_metric",
        }
        allowed_series = {"binned_mean", "quantile_pair", "summary_metric"}
        self.assertTrue(set(result.diagnostic_summary["diagnostic_type"]) <= allowed_types)
        self.assertTrue(set(result.diagnostic_summary["series"]) <= allowed_series)
        for model_id in result.fit_statistics["model_id"]:
            model_rows = result.diagnostic_summary.loc[
                result.diagnostic_summary["model_id"].eq(model_id)
            ]
            self.assertTrue(
                {"residuals_vs_fitted", "normal_qq"}.issubset(
                    model_rows["diagnostic_type"]
                )
            )

    def test_support_scope_is_assigned_per_term(self) -> None:
        """Module mean and extension domains retain their limited support label."""
        members = self.members.copy()
        selection = LockedCandidateSelection(
            "domains",
            ("module_mindset",),
            4.0,
            0.75,
            {"module_mindset": 0.9},
        )
        result = fit_locked_percentage_model(members, selection, "percentage_loss_ols")
        extension = result.coefficient_table.loc[
            result.coefficient_table["term"].eq("module_mindset")
        ]
        self.assertEqual(
            set(extension["support_scope"]),
            {"glp1_observed_only_not_verifiable_in_coaching"},
        )
        base = result.coefficient_table.loc[
            result.coefficient_table["term"].eq("first_weight")
        ]
        self.assertEqual(
            set(base["support_scope"]),
            {"primary_two_group", "common_weight_days_support"},
        )

        mean_selection = LockedCandidateSelection(
            "mean", ("module_mean",), 4.0, 0.75, {"module_mean": 0.9}
        )
        mean_result = fit_locked_percentage_model(
            members, mean_selection, "percentage_loss_ols"
        )
        self.assertEqual(
            set(
                mean_result.coefficient_table.loc[
                    mean_result.coefficient_table["term"].eq("module_mean"),
                    "support_scope",
                ]
            ),
            {"glp1_observed_only_not_verifiable_in_coaching"},
        )

    def test_rejects_unknown_duplicate_nonfinite_and_rank_deficient_candidates(self) -> None:
        """Locked refits fail loudly instead of silently changing their equation."""
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fit_locked_percentage_model(self.members, LockedCandidateSelection("mean", ("unknown",), 4.0, 0.75, {"unknown": 0.9}), "percentage_loss_ols")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_locked_percentage_model(self.members, LockedCandidateSelection("mean", ("sex[MALE]", "sex[MALE]"), 4.0, 0.75, {"sex[MALE]": 0.9}), "percentage_loss_ols")
        nonfinite = self.members.copy()
        nonfinite.loc[0, "engagement_volume_repeatable"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_locked_percentage_model(nonfinite, self.selection, "percentage_loss_ols")
        collinear = self.members.copy()
        collinear["engagement_breadth"] = collinear["first_weight"]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fit_locked_percentage_model(collinear, LockedCandidateSelection("mean", ("engagement_breadth",), 4.0, 0.75, {"engagement_breadth": 0.9}), "percentage_loss_ols")


if __name__ == "__main__":
    unittest.main()
