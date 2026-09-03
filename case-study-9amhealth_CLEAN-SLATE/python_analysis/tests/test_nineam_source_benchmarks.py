"""Lock the audited source cohort, feature totals, and model benchmarks."""

from __future__ import annotations

import unittest
from pathlib import Path

from nineam_health_analysis.nineam_cohort_selection import (
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
from nineam_health_analysis.nineam_resampling import compare_base_models
from nineam_health_analysis.nineam_statistical_models import (
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
)

# Resolve the supplied read-only extracts from the repository layout instead
# of embedding one analyst's absolute drive path in the regression tests.
_SOURCE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_source_analysis() -> tuple[CaseStudyData, CohortResult, FeatureResult]:
    """Load the supplied extracts and build the audited analysis tables.

    Args:
        None.

    Returns:
        Canonical source tables, the eligible cohort, and engineered features.

    Side effects:
        Reads the four source files without modifying or copying them.

    Statistical intent:
        Gives every source benchmark the same deterministic cohort and feature
        definitions used by the analysis pipeline.
    """
    data = load_case_study_data(_SOURCE_DATA_DIR)
    complete_pair = build_analysis_cohort(data)
    cohort = restrict_to_primary_member_types(complete_pair)
    features = build_member_features(data, cohort)
    return data, cohort, features


@unittest.skipUnless(
    _SOURCE_DATA_DIR.is_dir(),
    "Supplied case-study data are not available in this checkout",
)
class SourceBenchmarkTestCase(unittest.TestCase):
    """Protect audited source results while allowing data-free package installs."""

    def test_source_cohort_and_feature_totals_match_the_audit(self) -> None:
        """Reconcile exclusions and engineered totals to the audited extracts.

        Args:
            self: Test case providing exact and approximate assertions.

        Returns:
            None.

        Side effects:
            Reads the supplied source files; writes nothing.

        Statistical intent:
            Detects accidental changes to eligibility, temporal censoring,
            repeatable-event classification, or curriculum normalization.
        """
        _, cohort, features = _load_source_analysis()
        members = features.member_features

        self.assertEqual(len(members), 534)
        self.assertEqual(len(cohort.long_weights), 1068)
        self.assertEqual(cohort.audit_counts["source_demographic_rows"], 865)
        self.assertEqual(cohort.audit_counts["source_weight_rows"], 827)
        self.assertEqual(
            cohort.audit_counts["excluded_missing_body_weight_row"],
            38,
        )
        self.assertEqual(cohort.audit_counts["excluded_ineligible_status"], 2)
        self.assertEqual(
            cohort.audit_counts["excluded_nonpositive_weight_days"],
            192,
        )
        self.assertEqual(
            cohort.audit_counts["pre_member_type_restriction_members"],
            633,
        )
        self.assertEqual(
            cohort.audit_counts["excluded_nonprimary_member_type"],
            99,
        )
        self.assertEqual(
            cohort.audit_counts["excluded_active_glp1_diabetes"],
            7,
        )
        self.assertEqual(
            cohort.audit_counts[
                "excluded_active_generic_weight_loss_medication"
            ],
            57,
        )
        self.assertEqual(cohort.audit_counts["excluded_null_member_type"], 35)
        self.assertEqual(
            cohort.audit_counts["excluded_nonprimary_member_type"],
            cohort.audit_counts["excluded_active_glp1_diabetes"]
            + cohort.audit_counts[
                "excluded_active_generic_weight_loss_medication"
            ]
            + cohort.audit_counts["excluded_null_member_type"],
        )
        self.assertEqual(cohort.audit_counts["included_members"], 534)
        self.assertEqual(
            int(
                members["member_type"].eq("Active GLP-1 for Weight-loss").sum()
            ),
            331,
        )
        self.assertEqual(int(members["member_type"].eq("Coaching Only").sum()), 203)
        self.assertEqual(features.audit_counts["repeatable_event_types"], 10)
        self.assertEqual(features.audit_counts["retained_engagement_rows"], 37411)
        self.assertEqual(features.audit_counts["retained_module_rows"], 3201)
        self.assertEqual(features.audit_counts["distinct_module_completions"], 3187)

        # These sums jointly lock the supplied R-compatible event rule, its
        # exact per-day denominator, and the separate module-domain scaling.
        expected_totals = {
            "engagement_breadth": 6051.0,
            "engagement_volume_repeatable": 34759.0,
            "engagement_volume_repeatable_rate": 140.73586021084105,
            "tenure_days": 130063.0,
            "module_mean": 106.86111111111111,
        }
        for column, expected in expected_totals.items():
            with self.subTest(column=column):
                self.assertAlmostEqual(float(members[column].sum()), expected)

    def test_source_likelihood_and_regression_metrics_match_benchmarks(self) -> None:
        """Reproduce the four primary-cohort model summaries.

        Args:
            self: Test case providing numerical benchmark assertions.

        Returns:
            None.

        Side effects:
            Reads the source files and fits models in memory; writes nothing.

        Statistical intent:
            Locks likelihood constants, raw-scale lognormal Jacobians, factor
            coding, and the confirmed positive percentage-loss definition.
        """
        _, _, features = _load_source_analysis()
        members = features.member_features

        raw_unstructured = fit_longitudinal_gls(
            members,
            outcome_scale="raw",
            covariance_structure="unstructured",
        )
        log_compound_symmetry = fit_longitudinal_gls(
            members,
            outcome_scale="log",
            covariance_structure="compound_symmetry",
        )
        log_interaction = fit_longitudinal_gls(
            members,
            outcome_scale="log",
            covariance_structure="compound_symmetry",
            include_time_by_member_type=True,
        )
        percentage = fit_percentage_loss_ols(members)

        self.assertAlmostEqual(
            raw_unstructured.negative_two_log_likelihood,
            10213.22414647514,
            places=7,
        )
        self.assertAlmostEqual(raw_unstructured.aic, 10225.22414647514, places=7)
        self.assertAlmostEqual(
            log_compound_symmetry.negative_two_log_likelihood,
            10083.422914559282,
            places=7,
        )
        self.assertAlmostEqual(log_compound_symmetry.aic, 10093.422914559282, places=7)
        self.assertAlmostEqual(
            log_interaction.negative_two_log_likelihood,
            9927.515115980328,
            places=7,
        )
        self.assertAlmostEqual(log_interaction.aic, 9939.515115980328, places=7)
        self.assertAlmostEqual(
            percentage.negative_two_log_likelihood,
            3357.4730804692226,
            places=7,
        )
        self.assertAlmostEqual(percentage.r_squared, 0.2763807819582549, places=10)
        self.assertAlmostEqual(percentage.rmse, 5.611190194990324, places=10)

    def test_source_repeated_comparison_favors_percentage_model_on_rmse(self) -> None:
        """Smoke-test the prespecified common-target base-model comparison.

        Args:
            self: Test case providing paired comparison assertions.

        Returns:
            None.

        Side effects:
            Reads source files and runs deterministic in-memory resampling.

        Statistical intent:
            Confirms that percentage OLS has lower mean held-out raw last-weight
            RMSE than log-CS GLS without comparing incompatible likelihoods.
        """
        _, _, features = _load_source_analysis()
        comparison = compare_base_models(
            features.member_features,
            n_splits=5,
            n_repeats=2,
            random_state=2026,
        )
        mean_rmse = comparison.groupby("model", sort=True)["rmse"].mean()

        self.assertEqual(len(comparison), 20)
        self.assertNotIn("aic", {column.lower() for column in comparison.columns})
        self.assertLess(
            mean_rmse["percentage_loss_ols"],
            mean_rmse["log_compound_symmetry_gls"],
        )


if __name__ == "__main__":
    unittest.main()
