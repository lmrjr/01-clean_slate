"""Tests for deterministic aggregate scientific reporting tables."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from nineam_health_analysis.nineam_cohort_selection import (
    PRIMARY_MEMBER_TYPES,
    CohortResult,
)
from nineam_health_analysis.nineam_data_loading import CaseStudyData
from nineam_health_analysis.nineam_feature_engineering import FeatureResult
from nineam_health_analysis.nineam_final_model import (
    LockedCandidateSelection,
    fit_locked_percentage_model,
)
from nineam_health_analysis.nineam_reporting import (
    ACTIONABILITY_CLASSES,
    LIMITATION_IDS,
    REPORTING_SCHEMAS,
    ReportingResult,
    _aggregate_diagnostic_points,
    _benjamini_hochberg,
    _continuous_statistics,
    _fit_activity_association,
    build_reporting_tables,
    wilson_confidence_interval,
)

EVENT_TYPES = (
    "CHART_REVIEW",
    "COMPLETED_CONSULTATION",
    "COMPLETED_LAB_TEST",
    "CONSUMED_DIGITAL_CONTENT",
    "MEAL_PLAN_GENERATED",
    "MEDICAL_QUESTIONNAIRE_ANSWERED",
    "MEDICATION_CHANGE",
    "QUESTIONNAIRE_ANSWERED",
    "RECORD_BLOOD_GLUCOSE",
    "RECORD_BLOOD_GLUCOSE_WITH_REVIEW",
    "RECORD_BLOOD_PRESSURE",
    "RECORD_BLOOD_PRESSURE_WITH_REVIEW",
    "RECORD_BODY_WEIGHT",
    "RECORD_STEPS",
    "REGISTRATION",
    "SUBSCRIPTION_STARTED",
    "TEXT_MESSAGE_CARE_ONLY",
    "VIDEO_CALL_COMPLETED",
    "VOICE_MESSAGE_CARE_ONLY",
)

CANONICAL_CANDIDATE_ORDER = {
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


def _activity_oracle_members() -> tuple[pd.DataFrame, pd.Series]:
    """Return the fixed full-rank fixture used for an HC3 literal oracle."""
    first_weight = np.array(
        [100, 110, 120, 130, 140, 150, 105, 115, 125, 135, 145, 155],
        dtype=float,
    )
    days = np.array(
        [30, 45, 65, 80, 105, 125, 35, 55, 70, 95, 115, 135],
        dtype=float,
    )
    counts = pd.Series(
        [1, 2, 4, 1, 6, 3, 2, 5, 1, 7, 4, 8], dtype="int64"
    )
    percentage_loss = np.array(
        [5, 6.2, 4.9, 7.1, 6.8, 8, 7, 8.1, 6.9, 9.2, 8.7, 10.1],
        dtype=float,
    )
    members = pd.DataFrame(
        {
            "first_weight": first_weight,
            "member_type": ["Coaching Only"] * 6
            + ["Active GLP-1 for Weight-loss"] * 6,
            "weight_days": days,
            "percentage_loss": percentage_loss,
        }
    )
    return members, counts


def _member_fixture(n_members: int = 32) -> pd.DataFrame:
    """Build deterministic primary-member features with extension support limits."""
    index = np.arange(n_members)
    member_type = np.where(
        index % 2 == 0,
        "Active GLP-1 for Weight-loss",
        "Coaching Only",
    )
    first_weight = 180.0 + 3.0 * index + (index % 4)
    weight_days = 40.0 + 4.0 * index + (index % 3)
    volume = 1 + (index * 3) % 11
    breadth = 1 + (index * 5) % 8
    percentage_loss = (
        2.0
        + 0.025 * first_weight
        + 1.4 * (member_type == "Active GLP-1 for Weight-loss")
        + 0.18 * volume
        + 0.07 * ((index % 5) - 2)
    )
    last_weight = first_weight * (1.0 - percentage_loss / 100.0)
    core_count = (index % 7).astype(int)
    extension_support = member_type == "Active GLP-1 for Weight-loss"
    mindset_count = np.where(extension_support, index % 4, 0).astype(int)
    nutrition_count = np.where(extension_support, (index + 1) % 4, 0).astype(int)
    activity_count = np.where(extension_support, (index + 2) % 4, 0).astype(int)
    return pd.DataFrame(
        {
            "member_id": [f"member_{value:02d}" for value in index],
            "start_date": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(index, unit="D"),
            "subscription_status": np.where(index % 3, "ACTIVE", "FINISHED"),
            "cancellation_date": pd.NaT,
            "sex": np.where(index % 3, "FEMALE", "MALE"),
            "ethnicity": np.where(index % 4, "WHITE", "ASIAN"),
            "member_type": member_type,
            "first_weight_at": pd.Timestamp("2025-01-15")
            + pd.to_timedelta(index, unit="D"),
            "last_weight_at": pd.Timestamp("2025-04-15")
            + pd.to_timedelta(index, unit="D"),
            "first_weight": first_weight,
            "last_weight": last_weight,
            "weight_difference": last_weight - first_weight,
            "weight_days": weight_days,
            "tenure_days": weight_days + 12.0,
            "absolute_weight_loss": first_weight - last_weight,
            "percentage_loss": percentage_loss,
            "weight_loss_success_5pct": percentage_loss >= 5.0,
            "engagement_breadth": breadth,
            "engagement_volume_repeatable": volume,
            "engagement_volume_repeatable_rate": volume / (weight_days + 12.0),
            "module_core_count": core_count,
            "module_mindset_count": mindset_count,
            "module_nutrition_count": nutrition_count,
            "module_physical_activity_count": activity_count,
            "module_core": core_count / 9.0,
            "module_mindset": mindset_count / 4.0,
            "module_nutrition": nutrition_count / 4.0,
            "module_physical_activity": activity_count / 4.0,
        }
    ).assign(
        module_mean=lambda frame: frame[
            [
                "module_core",
                "module_mindset",
                "module_nutrition",
                "module_physical_activity",
            ]
        ].mean(axis="columns")
    )


def _lasso_input(module_spec: str, score: float) -> pd.DataFrame:
    """Create an enriched upstream LASSO fixture with explicit provenance."""
    if module_spec == "mean":
        candidates = (
            "engagement_volume_repeatable",
            "engagement_volume_repeatable_rate",
            "engagement_breadth",
            "tenure_days",
            "module_mean",
            "sex[MALE]",
        )
    else:
        candidates = (
            "engagement_volume_repeatable",
            "engagement_volume_repeatable_rate",
            "engagement_breadth",
            "tenure_days",
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
            "sex[MALE]",
        )
    records = []
    for candidate in candidates:
        frequency = 0.80 if candidate == "engagement_volume_repeatable" else 0.25
        records.append(
            {
                "module_spec": module_spec,
                "candidate_order": CANONICAL_CANDIDATE_ORDER[candidate],
                "candidate": candidate,
                "lambda_ratio": 0.25,
                "full_sample_lambda": 0.5,
                "full_sample_lambda_max": 2.0,
                "cv_selection_rule": "one_standard_error",
                "cv_mean_mse": score,
                "cv_standard_error": 0.2,
                "fold_plan_id": "shared_grouped_cv_seed_17",
                "n_resamples": 20,
                "subsample_fraction": 0.7,
                "selection_threshold": 0.75,
                "full_sample_coefficient": 0.4 if frequency >= 0.75 else 0.0,
                "full_sample_standardized_coefficient": (
                    0.2 if frequency >= 0.75 else 0.0
                ),
                "selection_count": int(20 * frequency),
                "selection_frequency": frequency,
                "selected_at_threshold": frequency >= 0.75,
                "excluded_from_full_fit": False,
            }
        )
    return pd.DataFrame.from_records(records)


class ReportingPrimitiveTestCase(unittest.TestCase):
    """Exercise hand-derived statistical helpers independently of orchestration."""

    def test_wilson_interval_matches_hand_checked_result(self) -> None:
        """Five successes in ten trials has the known 95% Wilson interval."""
        lower, upper = wilson_confidence_interval(5, 10)
        self.assertAlmostEqual(lower, 0.236593, places=6)
        self.assertAlmostEqual(upper, 0.763407, places=6)

    def test_continuous_statistics_use_sample_sd_and_linear_quartiles(self) -> None:
        """Literal values lock the approved descriptive conventions."""
        summary = _continuous_statistics(pd.Series([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(summary["n_observed"], 4)
        self.assertEqual(summary["n_missing"], 0)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["standard_deviation"], 1.290994449)
        self.assertAlmostEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["q1"], 1.75)
        self.assertAlmostEqual(summary["q3"], 3.25)

    def test_benjamini_hochberg_matches_literal_family(self) -> None:
        """The monotone reverse-minimum correction is hand-checkable."""
        adjusted = _benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

    def test_activity_hc3_matches_fixed_numeric_oracle(self) -> None:
        """The standardized log-count effect uses the full adjusted HC3 model."""
        members, counts = _activity_oracle_members()
        result = _fit_activity_association(members, counts)
        self.assertAlmostEqual(result["estimate"], 0.313229352, places=8)
        self.assertAlmostEqual(result["standard_error"], 0.505577116, places=8)
        self.assertAlmostEqual(result["p_value"], 0.53555533, places=8)
        self.assertAlmostEqual(result["ci_lower"], -0.67768359, places=8)
        self.assertAlmostEqual(result["ci_upper"], 1.30414229, places=8)
        self.assertEqual(result["covariance_estimator"], "HC3")
        self.assertEqual(result["reference_distribution"], "normal")
        self.assertTrue(pd.isna(result["degrees_of_freedom"]))

    def test_diagnostic_bins_are_aggregate_and_have_two_rows_minimum(self) -> None:
        """Four internal points become two deterministic residual bins."""
        points = pd.DataFrame(
            {
                "model_id": ["locked_percentage_loss_primary"] * 4,
                "fitted_value": [1.0, 2.0, 3.0, 4.0],
                "residual": [-2.0, -1.0, 1.0, 2.0],
                "standardized_residual": [-1.5, -0.5, 0.5, 1.5],
                "leverage": [0.1, 0.2, 0.3, 0.4],
                "cooks_distance": [0.01, 0.02, 0.03, 0.04],
            }
        )
        rows = _aggregate_diagnostic_points(points)
        residual = rows.query(
            "diagnostic_type == 'residuals_vs_fitted' and series == 'binned_mean'"
        ).iloc[0]
        self.assertEqual(int(residual["bin_count"]), 2)
        self.assertAlmostEqual(residual["x_value"], 1.5)
        self.assertAlmostEqual(residual["y_value"], -1.5)
        self.assertAlmostEqual(residual["y_lower"], -1.75)
        self.assertAlmostEqual(residual["y_upper"], -1.25)
        self.assertTrue(rows["bin_count"].ge(2).all())
        self.assertNotIn("member_id", rows.columns)


class ReportingBuilderTestCase(unittest.TestCase):
    """Exercise all fifteen closed reporting tables from aggregate fixtures."""

    def setUp(self) -> None:
        members = _member_fixture()
        demographics = members.loc[
            :,
            [
                "member_id",
                "start_date",
                "subscription_status",
                "cancellation_date",
                "sex",
                "ethnicity",
            ],
        ].copy()
        body_weights = members.loc[
            :,
            [
                "member_id",
                "member_type",
                "first_weight_at",
                "last_weight_at",
                "first_weight",
                "last_weight",
                "weight_difference",
                "weight_days",
                "tenure_days",
            ],
        ].copy()
        self.data = CaseStudyData(
            module_completions=pd.DataFrame(
                columns=["member_id", "questionnaire_title", "answered_at"]
            ),
            demographics=demographics,
            engagement=pd.DataFrame(
                columns=["member_id", "activity_at", "event_type"]
            ),
            body_weights=body_weights,
        )
        self.cohort = CohortResult(
            members=members.iloc[:, :17].copy(),
            long_weights=pd.DataFrame(),
            audit_counts={
                "source_weight_rows": 38,
                "source_demographic_rows": 40,
                "orphan_weight_rows": 0,
                "excluded_missing_body_weight_row": 2,
                "excluded_ineligible_status": 1,
                "excluded_missing_weights": 1,
                "excluded_nonpositive_weights": 1,
                "excluded_missing_weight_days": 1,
                "excluded_nonpositive_weight_days": 1,
                "pre_member_type_restriction_members": 33,
                "excluded_nonprimary_member_type": 1,
                "included_members": 32,
            },
        )
        repeatable = {
            "CHART_REVIEW",
            "COMPLETED_CONSULTATION",
            "CONSUMED_DIGITAL_CONTENT",
            "MEDICATION_CHANGE",
            "QUESTIONNAIRE_ANSWERED",
            "RECORD_BLOOD_PRESSURE",
            "RECORD_BODY_WEIGHT",
            "RECORD_STEPS",
            "TEXT_MESSAGE_CARE_ONLY",
            "VOICE_MESSAGE_CARE_ONLY",
        }
        event_summary = pd.DataFrame(
            {
                "event_type": EVENT_TYPES,
                "events": [100 + value for value in range(len(EVENT_TYPES))],
                "distinct_members": [32] * len(EVENT_TYPES),
                "events_per_member": [3.0] * len(EVENT_TYPES),
                "is_repeatable": [value in repeatable for value in EVENT_TYPES],
            }
        )
        event_records = [
            {
                "member_id": members.loc[index, "member_id"],
                "event_type": "RECORD_STEPS",
                "event_count": 1 + (index * 3) % 7,
                "is_repeatable": True,
            }
            for index in range(30)
        ]
        event_records.extend(
            [
                {
                    "member_id": members.loc[index, "member_id"],
                    "event_type": "CHART_REVIEW",
                    "event_count": index + 1,
                    "is_repeatable": True,
                }
                for index in range(2)
            ]
        )
        self.features = FeatureResult(
            member_features=members,
            event_type_summary=event_summary,
            member_event_counts=pd.DataFrame.from_records(event_records),
            audit_counts={"included_members": 32},
        )
        self.base_model_cv = pd.DataFrame(
            {
                "repeat": [0, 0, 0, 0, 1, 1, 1, 1],
                "fold": [0, 1, 0, 1, 0, 1, 0, 1],
                "model": [
                    "log_compound_symmetry_gls",
                    "log_compound_symmetry_gls",
                    "percentage_loss_ols",
                    "percentage_loss_ols",
                ]
                * 2,
                "n_test": [16] * 8,
                "rmse": [3.2, 3.4, 2.8, 3.0, 3.3, 3.5, 2.9, 3.1],
                "mae": [2.3, 2.4, 2.0, 2.1, 2.4, 2.5, 2.1, 2.2],
            }
        )
        self.lasso_mean = _lasso_input("mean", 2.0)
        self.lasso_domains = _lasso_input("domains", 3.0)
        selection = LockedCandidateSelection(
            module_spec="mean",
            candidates=("engagement_volume_repeatable",),
            cv_mean_mse=2.0,
            selection_threshold=0.75,
            selection_frequencies={"engagement_volume_repeatable": 0.8},
        )
        self.locked = fit_locked_percentage_model(
            members,
            selection,
            "percentage_loss_ols",
        )

    def _build(
        self,
        *,
        features: FeatureResult | None = None,
        base_model_cv: pd.DataFrame | None = None,
        lasso_mean: pd.DataFrame | None = None,
        lasso_domains: pd.DataFrame | None = None,
        locked_model=None,
    ):
        return build_reporting_tables(
            self.data,
            self.cohort,
            self.features if features is None else features,
            self.base_model_cv if base_model_cv is None else base_model_cv,
            self.lasso_mean if lasso_mean is None else lasso_mean,
            self.lasso_domains if lasso_domains is None else lasso_domains,
            self.locked if locked_model is None else locked_model,
        )

    def test_cohort_flow_uses_visualization_stage_row_type(self) -> None:
        """Every cohort-flow row uses the one literal consumed by figures."""
        flow = self._build().tables["nineam_cohort_flow.csv"]
        self.assertEqual(set(flow["row_type"]), {"stage"})

    def test_strict_base_winner_does_not_treat_near_rmse_as_a_tie(self) -> None:
        """Any literal RMSE advantage wins before MAE, even at 5e-13."""
        comparison = self.base_model_cv.copy()
        log_rows = comparison["model"].eq("log_compound_symmetry_gls")
        percentage_rows = comparison["model"].eq("percentage_loss_ols")
        comparison.loc[log_rows, "rmse"] = 3.0 - 5e-13
        comparison.loc[percentage_rows, "rmse"] = 3.0
        comparison.loc[log_rows, "mae"] = 2.5
        comparison.loc[percentage_rows, "mae"] = 2.0
        locked = replace(
            self.locked,
            winning_base_model="log_compound_symmetry_gls",
        )

        summary = self._build(
            base_model_cv=comparison,
            locked_model=locked,
        ).tables["nineam_base_model_comparison_summary.csv"]

        winners = summary.loc[summary["is_winner"], "model_id"].unique()
        self.assertEqual(tuple(winners), ("log_compound_symmetry_gls",))
        self.assertEqual(set(summary["tie_tolerance"]), {0.0})

    def test_reporting_result_isolates_source_and_accessed_dataframes(self) -> None:
        """Mutating either input or an accessed table cannot change held state."""
        source = {
            filename: pd.DataFrame(columns=schema)
            for filename, schema in REPORTING_SCHEMAS.items()
        }
        result = ReportingResult(source)
        first = "nineam_cohort_flow.csv"

        source[first].loc[0, "flow_order"] = 99
        self.assertTrue(result.tables[first].empty)

        accessed = result.tables[first]
        accessed.loc[0, "flow_order"] = 77
        self.assertTrue(result.tables[first].empty)

    def test_feature_member_types_must_match_cohort_by_member_id(self) -> None:
        """Swapping two labels while retaining both sets cannot evade alignment."""
        changed = self.features.member_features.copy()
        changed.loc[[0, 1], "member_type"] = changed.loc[
            [1, 0], "member_type"
        ].to_numpy()
        invalid = FeatureResult(
            member_features=changed,
            event_type_summary=self.features.event_type_summary,
            member_event_counts=self.features.member_event_counts,
            audit_counts=self.features.audit_counts,
        )

        with self.assertRaisesRegex(ValueError, "member_id-to-member_type"):
            self._build(features=invalid)

    def test_attrition_limitation_stops_before_member_type_restriction(self) -> None:
        """Complete-pair attrition is seven; the separate restriction removes one."""
        limitations = self._build().tables["nineam_limitations.csv"]
        attrition = limitations.loc[
            limitations["limitation_id"].eq("attrition_complete_pair"),
            "empirical_evidence",
        ].iloc[0]
        restriction = limitations.loc[
            limitations["limitation_id"].eq("two_condition_restriction"),
            "empirical_evidence",
        ].iloc[0]
        self.assertEqual(
            attrition,
            "7 of 40 enrolled members lack a usable complete outcome pair.",
        )
        self.assertEqual(
            restriction,
            "1 complete-pair members were removed by the member-type restriction.",
        )

    def test_module_mean_support_is_explicitly_limited_in_both_tables(self) -> None:
        """The mixed module mean never implies verified two-group availability."""
        tables = self._build().tables
        modules = tables["nineam_modules_by_member_type.csv"].query(
            "module_variable == 'module_mean'"
        )
        mean_lasso = tables["nineam_lasso_mean_selection.csv"].query(
            "candidate == 'module_mean'"
        )
        self.assertEqual(
            set(modules["availability_status"]),
            {"mixed_core_and_unverified_extensions"},
        )
        self.assertEqual(
            set(modules["support_scope"]),
            {"both_member_types_observed_with_extension_limitation"},
        )
        self.assertEqual(
            set(mean_lasso["support_scope"]),
            {"both_member_types_observed_with_extension_limitation"},
        )

    def test_lasso_count_frequency_identity_is_validated(self) -> None:
        """A count of 15 out of 20 cannot be reported as frequency 0.80."""
        invalid = self.lasso_mean.copy()
        invalid.loc[
            invalid["candidate"].eq("engagement_breadth"), "selection_count"
        ] = 15
        with self.assertRaisesRegex(
            ValueError,
            "selection_count / n_resamples",
        ):
            self._build(lasso_mean=invalid)

    def test_lasso_threshold_flag_is_derived_from_frequency_and_exclusion(self) -> None:
        """A nonexcluded 0.80-frequency candidate must carry a true flag."""
        invalid = self.lasso_mean.copy()
        target = invalid["candidate"].eq("engagement_breadth")
        invalid.loc[target, ["selection_count", "selection_frequency"]] = [16, 0.8]
        invalid.loc[target, "selected_at_threshold"] = False
        with self.assertRaisesRegex(ValueError, "selected_at_threshold"):
            self._build(lasso_mean=invalid)

    def test_excluded_lasso_candidate_remains_penalized_and_unselected(self) -> None:
        """Inestimable provenance is orthogonal to whether the term was penalized."""
        excluded = self.lasso_mean.copy()
        target = excluded["candidate"].eq("engagement_breadth")
        excluded.loc[target, ["selection_count", "selection_frequency"]] = [16, 0.8]
        excluded.loc[target, "selected_at_threshold"] = False
        excluded.loc[target, "excluded_from_full_fit"] = True

        row = self._build(lasso_mean=excluded).tables[
            "nineam_lasso_mean_selection.csv"
        ].loc[target].iloc[0]
        self.assertEqual(row["penalty_status"], "penalized")
        self.assertFalse(row["selected_at_threshold"])
        self.assertFalse(row["eligible_for_locked_model"])
        self.assertEqual(row["locked_model_status"], "inestimable_full_fit")

    def test_approved_lasso_candidates_require_global_canonical_order(self) -> None:
        """The module-mean candidate is fixed at order five, not any unique order."""
        invalid = self.lasso_mean.copy()
        invalid.loc[invalid["candidate"].eq("module_mean"), "candidate_order"] = 6
        with self.assertRaisesRegex(ValueError, "canonical candidate_order"):
            self._build(lasso_mean=invalid)

    def test_extra_sex_contrast_is_retained_but_never_lock_eligible(self) -> None:
        """An appended noncanonical sex contrast remains provenance-only."""
        extra = self.lasso_mean.iloc[[-1]].copy()
        extra.loc[:, "candidate"] = "sex[NON_BINARY]"
        extra.loc[:, "candidate_order"] = 11
        extra.loc[:, ["selection_count", "selection_frequency"]] = [16, 0.8]
        extra.loc[:, "selected_at_threshold"] = True
        mean = pd.concat([self.lasso_mean, extra], ignore_index=True)

        row = self._build(lasso_mean=mean).tables[
            "nineam_lasso_mean_selection.csv"
        ].query("candidate == 'sex[NON_BINARY]'").iloc[0]
        self.assertTrue(row["selected_at_threshold"])
        self.assertFalse(row["eligible_for_locked_model"])
        self.assertEqual(row["locked_model_status"], "ineligible_noncanonical_candidate")
        self.assertIn("canonical sex contrast", row["exclusion_reason"])

    def test_extra_sex_contrasts_require_alphabetic_appended_order(self) -> None:
        """Additional contrasts must occupy deterministic contiguous orders."""
        first = self.lasso_mean.iloc[[-1]].copy()
        first.loc[:, "candidate"] = "sex[NON_BINARY]"
        first.loc[:, "candidate_order"] = 12
        first.loc[:, ["selection_count", "selection_frequency"]] = [5, 0.25]
        first.loc[:, "selected_at_threshold"] = False
        second = first.copy()
        second.loc[:, "candidate"] = "sex[UNKNOWN]"
        second.loc[:, "candidate_order"] = 11
        invalid = pd.concat([self.lasso_mean, first, second], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "appended candidate_order"):
            self._build(lasso_mean=invalid)

    def test_domain_winner_drives_hypothesis_evidence_links_and_filters(self) -> None:
        """Engagement and module evidence both point to the winning domain table."""
        mean = self.lasso_mean.copy()
        domains = self.lasso_domains.copy()
        mean.loc[:, "cv_mean_mse"] = 3.0
        domains.loc[:, "cv_mean_mse"] = 2.0
        module = domains["candidate"].eq("module_core")
        domains.loc[module, ["selection_count", "selection_frequency"]] = [16, 0.8]
        domains.loc[module, "selected_at_threshold"] = True
        selection = LockedCandidateSelection(
            module_spec="domains",
            candidates=("engagement_volume_repeatable", "module_core"),
            cv_mean_mse=2.0,
            selection_threshold=0.75,
            selection_frequencies={
                "engagement_volume_repeatable": 0.8,
                "module_core": 0.8,
            },
        )
        locked = fit_locked_percentage_model(
            self.features.member_features,
            selection,
            "percentage_loss_ols",
        )

        hypotheses = self._build(
            lasso_mean=mean,
            lasso_domains=domains,
            locked_model=locked,
        ).tables["nineam_hypothesis_evidence.csv"].set_index("hypothesis_id")
        for hypothesis_id, role in (
            ("engagement_stability", "engagement"),
            ("module_stability", "module"),
        ):
            row = hypotheses.loc[hypothesis_id]
            self.assertEqual(
                row["evidence_table"],
                "nineam_lasso_domain_selection.csv",
            )
            self.assertEqual(
                row["evidence_filter"],
                f"module_spec=domains; candidate_role={role}; "
                "eligible_for_locked_model=true",
            )

    def test_all_tables_use_exact_closed_schemas_and_no_identifiers(self) -> None:
        """The result exposes all fifteen approved filename/column contracts."""
        tables = self._build().tables
        self.assertEqual(tuple(tables), tuple(REPORTING_SCHEMAS))
        for filename, schema in REPORTING_SCHEMAS.items():
            self.assertEqual(tuple(tables[filename].columns), schema)
            self.assertNotIn("member_id", tables[filename].columns)

    def test_outcomes_scopes_wilson_and_engagement_sensitivity_are_explicit(self) -> None:
        """Main summaries use fixed scopes and distinguish both rate denominators."""
        tables = self._build().tables
        outcomes = tables["nineam_outcomes_by_member_type.csv"]
        self.assertEqual(
            tuple(outcomes["scope"].drop_duplicates()),
            ("Overall", *PRIMARY_MEMBER_TYPES),
        )
        response = outcomes.query("outcome == 'weight_loss_success_5pct'")
        self.assertTrue(response["ci_method"].eq("Wilson").all())
        self.assertTrue(response["ci_lower"].between(0.0, 100.0).all())
        engagement = tables["nineam_engagement_by_member_type.csv"]
        sensitivity = engagement.query(
            "metric == 'engagement_volume_repeatable_rate_weight_days_sensitivity'"
        )
        self.assertEqual(len(sensitivity), 3)
        self.assertTrue(
            sensitivity["denominator_definition"].eq("max(weight_days, 7)").all()
        )

    def test_blank_categorical_values_collapse_into_one_missing_level(self) -> None:
        """Blank and null sex/ethnicity values share one explicit output level."""
        members = self.features.member_features.copy()
        members.loc[0, "sex"] = "   "
        members.loc[1, "sex"] = pd.NA
        members.loc[2, "ethnicity"] = "  "
        members.loc[3, "ethnicity"] = pd.NA
        features = FeatureResult(
            member_features=members,
            event_type_summary=self.features.event_type_summary,
            member_event_counts=self.features.member_event_counts,
            audit_counts=self.features.audit_counts,
        )

        sample = self._build(features=features).tables[
            "nineam_sample_characteristics.csv"
        ]
        self.assertFalse(sample["level"].astype(str).str.strip().eq("").any())
        overall = sample.loc[sample["scope"].eq("Overall")]
        sex = overall.loc[overall["characteristic"].eq("sex")]
        ethnicity = overall.loc[overall["characteristic"].eq("ethnicity")]
        self.assertEqual(tuple(sex["level"]), ("FEMALE", "MALE", "Missing"))
        self.assertEqual(
            tuple(ethnicity["level"]),
            ("ASIAN", "WHITE", "Missing"),
        )
        self.assertEqual(
            int(sex.loc[sex["level"].eq("Missing"), "count"].iloc[0]),
            2,
        )
        self.assertEqual(
            int(
                ethnicity.loc[
                    ethnicity["level"].eq("Missing"), "count"
                ].iloc[0]
            ),
            2,
        )
        missing_rows = sample.loc[
            sample["characteristic"].isin(["sex", "ethnicity"])
            & sample["level"].eq("Missing")
        ]
        self.assertTrue(
            missing_rows.groupby(["scope", "characteristic"]).size().eq(1).all()
        )

    def test_nonprimary_conditions_appear_only_in_accounting_and_limitations(self) -> None:
        """The three excluded condition labels remain explicit but nonanalytical."""
        tables = self._build().tables
        flow_definition = tables["nineam_cohort_flow.csv"].iloc[-1][
            "exclusion_definition"
        ]
        restriction = tables["nineam_limitations.csv"].query(
            "limitation_id == 'two_condition_restriction'"
        ).iloc[0]["limitation"]
        for label in (
            "Null",
            "Active GLP-1 for Diabetes",
            "Active Generic Medication for Weight-loss",
        ):
            self.assertIn(label, flow_definition)
            self.assertIn(label, restriction)

    def test_activity_zero_fill_threshold_hc3_bh_and_actionability(self) -> None:
        """All types are descriptive while only eligible reach enters one FDR family."""
        activity = self._build().tables[
            "nineam_engagement_activity_summary.csv"
        ]
        self.assertEqual(tuple(activity["event_type"]), EVENT_TYPES)
        self.assertEqual(len(activity), 19)
        self.assertTrue(
            activity["actionability_class"].isin(ACTIONABILITY_CLASSES).all()
        )
        steps = activity.set_index("event_type").loc["RECORD_STEPS"]
        self.assertEqual(int(steps["n_reached"]), 30)
        self.assertEqual(int(steps["n_zero"]), 2)
        self.assertEqual(steps["test_status"], "tested")
        self.assertEqual(steps["covariance_estimator"], "HC3")
        self.assertFalse(pd.isna(steps["fdr_adjusted_p_value"]))
        chart = activity.set_index("event_type").loc["CHART_REVIEW"]
        self.assertEqual(int(chart["n_reached"]), 2)
        self.assertEqual(int(chart["n_zero"]), 30)
        self.assertEqual(chart["test_status"], "descriptive_low_reach")
        self.assertTrue(pd.isna(chart["p_value"]))
        self.assertTrue(pd.isna(chart["fdr_adjusted_p_value"]))

    def test_module_extension_rows_keep_coaching_zeros_and_support_warning(self) -> None:
        """Observed zero extension completions are not converted to missing support."""
        modules = self._build().tables["nineam_modules_by_member_type.csv"]
        coaching_extensions = modules.query(
            "scope == 'Coaching Only' and module_group == 'extension'"
        )
        self.assertEqual(len(coaching_extensions), 3)
        self.assertTrue(coaching_extensions["total_unique_completions"].eq(0).all())
        self.assertTrue(coaching_extensions["zero_completion_group_flag"].all())
        self.assertTrue(
            coaching_extensions["support_scope"].eq("glp_1_weight_loss_only").all()
        )
        self.assertTrue(
            coaching_extensions["availability_status"].eq(
                "extension_availability_not_verifiable"
            ).all()
        )

    def test_lasso_expansion_uses_explicit_provenance_and_one_winning_spec(self) -> None:
        """Closed selection rows preserve upstream counts, lambdas, and fold plan."""
        tables = self._build().tables
        mean = tables["nineam_lasso_mean_selection.csv"]
        domains = tables["nineam_lasso_domain_selection.csv"]
        self.assertTrue(mean["is_winning_specification"].all())
        self.assertFalse(domains["is_winning_specification"].any())
        selected = mean.query("candidate == 'engagement_volume_repeatable'").iloc[0]
        self.assertEqual(int(selected["selection_count"]), 16)
        self.assertEqual(selected["fold_plan_id"], "shared_grouped_cv_seed_17")
        self.assertTrue(selected["eligible_for_locked_model"])
        self.assertEqual(selected["locked_model_status"], "locked")

    def test_diagnostics_are_binned_aggregate_rows_only(self) -> None:
        """Internal point diagnostics never cross the ReportingResult boundary."""
        diagnostics = self._build().tables["nineam_model_diagnostics.csv"]
        plotted = diagnostics.query("row_type == 'plot_summary'")
        self.assertFalse(plotted.empty)
        self.assertTrue(plotted["bin_count"].ge(2).all())
        self.assertTrue(
            set(plotted["diagnostic_type"]).issuperset(
                {"residuals_vs_fitted", "normal_qq", "leverage", "cooks_distance"}
            )
        )
        self.assertTrue(
            set(diagnostics["series"]).issubset(
                {"binned_mean", "quantile_pair", "summary_metric"}
            )
        )
        self.assertFalse(diagnostics["row_type"].eq("binned_series").any())
        self.assertNotIn("fitted_value", diagnostics.columns)
        self.assertNotIn("residual", diagnostics.columns)

    def test_evidence_tables_use_controlled_noncausal_vocabularies(self) -> None:
        """Interpretive artifacts remain evidence-linked and explicitly noncausal."""
        tables = self._build().tables
        hypotheses = tables["nineam_hypothesis_evidence.csv"]
        findings = tables["nineam_findings_and_implications.csv"]
        limitations = tables["nineam_limitations.csv"]
        self.assertTrue(hypotheses["causal_status"].eq("noncausal_observational").all())
        self.assertTrue(findings["causal_status"].eq("noncausal_observational").all())
        self.assertEqual(tuple(limitations["limitation_id"]), LIMITATION_IDS)
        self.assertEqual(set(limitations["direction_of_bias"]), {"indeterminate"})
        self.assertTrue(hypotheses["evidence_table"].str.endswith(".csv").all())

    def test_member_type_evidence_filter_resolves_one_coefficient_row(self) -> None:
        """The published key-value filter uniquely selects the primary contrast."""
        tables = self._build().tables
        hypothesis = tables["nineam_hypothesis_evidence.csv"].loc[
            lambda frame: frame["hypothesis_id"].eq("member_type_percentage_loss")
        ].iloc[0]
        expected_filter = (
            "model_id=locked_percentage_loss_primary; term_role=base; "
            "contrast=Active GLP-1 for Weight-loss versus Coaching Only"
        )
        self.assertEqual(hypothesis["evidence_filter"], expected_filter)

        coefficients = tables[hypothesis["evidence_table"]]
        resolved = coefficients
        for clause in str(hypothesis["evidence_filter"]).split("; "):
            column, value = clause.split("=", maxsplit=1)
            resolved = resolved.loc[resolved[column].astype(str).eq(value)]

        self.assertEqual(len(resolved), 1)
        self.assertEqual(
            resolved.iloc[0]["term"],
            "member_type[Active GLP-1 for Weight-loss]",
        )

    def test_shuffled_inputs_produce_identical_ordered_tables(self) -> None:
        """Source row order cannot change any aggregate reporting artifact."""
        baseline = self._build().tables
        shuffled_features = FeatureResult(
            member_features=self.features.member_features.sample(
                frac=1.0, random_state=7
            ),
            event_type_summary=self.features.event_type_summary.sample(
                frac=1.0, random_state=8
            ),
            member_event_counts=self.features.member_event_counts.sample(
                frac=1.0, random_state=9
            ),
            audit_counts=self.features.audit_counts,
        )
        shuffled = self._build(
            features=shuffled_features,
            base_model_cv=self.base_model_cv.sample(frac=1.0, random_state=10),
            lasso_mean=self.lasso_mean.sample(frac=1.0, random_state=11),
            lasso_domains=self.lasso_domains.sample(frac=1.0, random_state=12),
        ).tables
        for filename in REPORTING_SCHEMAS:
            assert_frame_equal(
                baseline[filename],
                shuffled[filename],
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_unknown_activity_type_fails_closed(self) -> None:
        """Unreviewed events cannot silently receive an actionability label."""
        invalid_summary = pd.concat(
            [
                self.features.event_type_summary,
                pd.DataFrame(
                    {
                        "event_type": ["UNKNOWN_ACTIVITY"],
                        "events": [1],
                        "distinct_members": [1],
                        "events_per_member": [1.0],
                        "is_repeatable": [False],
                    }
                ),
            ],
            ignore_index=True,
        )
        invalid = FeatureResult(
            self.features.member_features,
            invalid_summary,
            self.features.member_event_counts,
            self.features.audit_counts,
        )
        with self.assertRaisesRegex(ValueError, "actionability"):
            self._build(features=invalid)

    def test_missing_lasso_provenance_fails_with_explicit_contract(self) -> None:
        """Reporting never invents unavailable selection counts or lambda values."""
        with self.assertRaisesRegex(ValueError, "full_sample_lambda"):
            self._build(lasso_mean=self.lasso_mean.drop(columns="full_sample_lambda"))


if __name__ == "__main__":
    unittest.main()
