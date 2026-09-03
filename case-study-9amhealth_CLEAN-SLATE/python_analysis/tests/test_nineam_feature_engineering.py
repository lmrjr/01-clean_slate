"""Tests for engagement and module feature engineering."""

from __future__ import annotations

import unittest

import pandas as pd

from nineam_health_analysis import MODULE_DOMAIN_DENOMINATORS
from nineam_health_analysis.nineam_cohort_selection import CohortResult
from nineam_health_analysis.nineam_data_loading import CaseStudyData
from nineam_health_analysis.nineam_feature_engineering import (
    build_member_features,
    classify_repeatable_events,
)


def _make_analysis_inputs(
    member_rows: list[dict[str, object]],
    engagement_rows: list[dict[str, object]],
    module_rows: list[dict[str, object]],
) -> tuple[CaseStudyData, CohortResult]:
    """Build canonical in-memory inputs for feature tests.

    Args:
        member_rows: Eligible member records with cutoff dates and tenure.
        engagement_rows: Canonical event records to aggregate.
        module_rows: Canonical module-completion records to aggregate.

    Returns:
        A ``CaseStudyData`` and ``CohortResult`` pair accepted by the feature
        pipeline.

    Side effects:
        None; new dataframes are created without mutating the supplied rows.

    Statistical intent:
        Isolates feature formulas from file ingestion and cohort eligibility.
    """
    # Preserve the production schemas even for empty fixtures so tests exercise
    # real dataframe joins rather than special-purpose mocks.
    engagement = pd.DataFrame(
        engagement_rows,
        columns=["member_id", "activity_at", "event_type"],
    )
    modules = pd.DataFrame(
        module_rows,
        columns=["member_id", "questionnaire_title", "answered_at"],
    )
    members = pd.DataFrame(member_rows)

    # Convert fixture dates explicitly because censoring must compare datetimes,
    # not lexicographically ordered strings.
    members["last_weight_at"] = pd.to_datetime(members["last_weight_at"])
    if not engagement.empty:
        engagement["activity_at"] = pd.to_datetime(engagement["activity_at"])
    if not modules.empty:
        modules["answered_at"] = pd.to_datetime(modules["answered_at"])

    data = CaseStudyData(
        module_completions=modules,
        demographics=pd.DataFrame(),
        engagement=engagement,
        body_weights=pd.DataFrame(),
    )
    cohort = CohortResult(
        members=members,
        long_weights=pd.DataFrame(),
        audit_counts={"included_members": len(members)},
    )
    return data, cohort


class FeatureEngineeringTestCase(unittest.TestCase):
    """Exercise observable feature behavior with hand-checked fixtures."""

    def test_public_module_domain_denominators_are_fixed_and_immutable(self) -> None:
        """Expose the confirmed curriculum scales without writable state.

        The behavior would fail if a consumer receives a mutable mapping or if
        core or extension curriculum denominators drift from 9 and 4.
        """
        self.assertEqual(MODULE_DOMAIN_DENOMINATORS["core"], 9.0)
        self.assertEqual(MODULE_DOMAIN_DENOMINATORS["mindset"], 4.0)
        self.assertEqual(MODULE_DOMAIN_DENOMINATORS["nutrition"], 4.0)
        self.assertEqual(MODULE_DOMAIN_DENOMINATORS["physical_activity"], 4.0)
        with self.assertRaises(TypeError):
            MODULE_DOMAIN_DENOMINATORS["core"] = 1.0

    def test_member_event_counts_include_observed_type_counts(self) -> None:
        """Count retained event types sparsely at the member-event boundary.

        The behavior would fail if events are counted before censoring, absent
        member-event pairs are materialized as zero rows, or the full-source
        repeatability classification is discarded.
        """
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "A",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                }
            ],
            engagement_rows=[
                {
                    "member_id": "A",
                    "activity_at": "2025-01-08",
                    "event_type": "read",
                },
                {
                    "member_id": "A",
                    "activity_at": "2025-01-10",
                    "event_type": "read",
                },
                {
                    "member_id": "A",
                    "activity_at": "2025-01-11",
                    "event_type": "read",
                },
            ],
            module_rows=[],
        )

        result = build_member_features(data, cohort)
        row = result.member_event_counts.query(
            "member_id == 'A' and event_type == 'read'"
        )

        self.assertEqual(int(row.iloc[0]["event_count"]), 2)
        self.assertFalse(bool(row.iloc[0]["is_repeatable"]))
        self.assertListEqual(
            result.member_event_counts.columns.tolist(),
            ["member_id", "event_type", "event_count", "is_repeatable"],
        )

    def test_member_event_counts_are_sparse_ordered_and_typed(self) -> None:
        """Return only observed member-event pairs in stable sorted order.

        The behavior would fail if a crossed zero pair is materialized, rows
        are not deterministically sorted, or schema types drift at the
        reporting boundary.
        """
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "A",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                },
                {
                    "member_id": "B",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                },
            ],
            engagement_rows=[
                {
                    "member_id": "B",
                    "activity_at": "2025-01-09",
                    "event_type": "beta",
                },
                {
                    "member_id": "A",
                    "activity_at": "2025-01-09",
                    "event_type": "alpha",
                },
                {
                    "member_id": "B",
                    "activity_at": "2025-01-11",
                    "event_type": "alpha",
                },
            ],
            module_rows=[],
        )

        counts = build_member_features(data, cohort).member_event_counts

        self.assertListEqual(
            counts.loc[:, ["member_id", "event_type"]].values.tolist(),
            [["A", "alpha"], ["B", "beta"]],
        )
        observed_pairs = counts[["member_id", "event_type"]].values.tolist()
        self.assertNotIn(["A", "beta"], observed_pairs)
        self.assertNotIn(["B", "alpha"], observed_pairs)
        self.assertEqual(counts["member_id"].dtype, "string")
        self.assertEqual(counts["event_type"].dtype, "string")
        self.assertEqual(counts["event_count"].dtype, "int64")
        self.assertEqual(counts["is_repeatable"].dtype, "bool")

    def test_member_event_counts_preserve_empty_sparse_schema_and_types(self) -> None:
        """Keep the member-event table typed when no events survive censoring.

        The behavior would fail if an empty retained window loses its closed
        schema or changes the aggregate boundary dtypes.
        """
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "A",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                }
            ],
            engagement_rows=[
                {
                    "member_id": "A",
                    "activity_at": "2025-01-11",
                    "event_type": "after_cutoff",
                }
            ],
            module_rows=[],
        )

        counts = build_member_features(data, cohort).member_event_counts

        self.assertTrue(counts.empty)
        self.assertListEqual(
            counts.columns.tolist(),
            ["member_id", "event_type", "event_count", "is_repeatable"],
        )
        self.assertEqual(counts["member_id"].dtype, "string")
        self.assertEqual(counts["event_type"].dtype, "string")
        self.assertEqual(counts["event_count"].dtype, "int64")
        self.assertEqual(counts["is_repeatable"].dtype, "bool")

    def test_repeatable_classification_uses_full_canonical_engagement(self) -> None:
        """Fix repeatability before cohort and last-weight filtering.

        Args:
            self: Test case providing audit and feature assertions.

        Returns:
            None.

        Side effects:
            None; all event and cohort fixtures remain in memory.

        Statistical intent:
            Makes one type repeatable only in the full canonical table, thereby
            protecting the supplied R classification order before censoring.
        """
        member_rows = [
            {
                "member_id": "member_00",
                "last_weight_at": "2025-01-01",
                "tenure_days": 14.0,
            }
        ]

        # The full table has the exact 30-member/two-event boundary, but only
        # one event for the sole cohort member survives outcome-date censoring.
        engagement_rows = [
            {
                "member_id": f"member_{member_number:02d}",
                "activity_at": f"2025-01-{event_number + 1:02d}",
                "event_type": "FULL_TABLE_REPEATABLE",
            }
            for member_number in range(30)
            for event_number in range(2)
        ]
        data, cohort = _make_analysis_inputs(
            member_rows=member_rows,
            engagement_rows=engagement_rows,
            module_rows=[],
        )

        result = build_member_features(data, cohort)
        summary = result.event_type_summary.set_index("event_type")
        member = result.member_features.set_index("member_id").loc["member_00"]

        self.assertEqual(summary.loc["FULL_TABLE_REPEATABLE", "events"], 60)
        self.assertEqual(
            summary.loc["FULL_TABLE_REPEATABLE", "distinct_members"], 30
        )
        self.assertTrue(
            bool(summary.loc["FULL_TABLE_REPEATABLE", "is_repeatable"])
        )
        self.assertEqual(member["engagement_volume_repeatable"], 1)
        self.assertEqual(result.audit_counts["retained_engagement_rows"], 1)

    def test_repeatable_classification_rejects_missing_identity_values(self) -> None:
        """Reject null or blank member and event identifiers.

        Args:
            self: Test case providing validation-error assertions.

        Returns:
            None.

        Side effects:
            None; each corrupt fixture is created and discarded in memory.

        Statistical intent:
            Prevents anonymous rows and blank categories from changing event
            counts or distinct-member denominators.
        """
        invalid_cases = [
            ("member_id", None),
            ("member_id", "   "),
            ("event_type", None),
            ("event_type", "   "),
        ]

        for column, invalid_value in invalid_cases:
            with self.subTest(column=column, invalid_value=invalid_value):
                engagement = pd.DataFrame(
                    {"member_id": ["member_01"], "event_type": ["EVENT"]}
                )
                engagement.loc[0, column] = invalid_value

                with self.assertRaisesRegex(ValueError, column):
                    classify_repeatable_events(engagement)

    def test_repeatable_event_thresholds_are_inclusive(self) -> None:
        """Classify only types meeting both confirmed threshold boundaries.

        Args:
            self: Test case providing threshold assertions.

        Returns:
            None.

        Side effects:
            None; the synthetic event table exists only in memory.

        Statistical intent:
            Protects both inclusive cutoffs that define repeatable engagement
            volume using event types placed exactly at each boundary.
        """
        event_rows: list[dict[str, object]] = []

        # Exactly 30 members with two events each must qualify because both
        # thresholds are inclusive rather than strict inequalities.
        for member_number in range(30):
            for _ in range(2):
                event_rows.append(
                    {
                        "member_id": f"qualified_{member_number:02d}",
                        "event_type": "QUALIFIED",
                    }
                )

        # Two events per member are insufficient when only 29 distinct members
        # contribute to the event type.
        for member_number in range(29):
            for _ in range(2):
                event_rows.append(
                    {
                        "member_id": f"few_{member_number:02d}",
                        "event_type": "TOO_FEW_MEMBERS",
                    }
                )

        # Thirty contributing members are insufficient when average volume is
        # below two events per member.
        for member_number in range(30):
            event_rows.append(
                {
                    "member_id": f"sparse_{member_number:02d}",
                    "event_type": "TOO_SPARSE",
                }
            )

        summary = classify_repeatable_events(pd.DataFrame(event_rows))
        summary = summary.set_index("event_type")

        self.assertEqual(summary.loc["QUALIFIED", "events"], 60)
        self.assertEqual(summary.loc["QUALIFIED", "distinct_members"], 30)
        self.assertEqual(summary.loc["QUALIFIED", "events_per_member"], 2.0)
        self.assertTrue(bool(summary.loc["QUALIFIED", "is_repeatable"]))
        self.assertFalse(
            bool(summary.loc["TOO_FEW_MEMBERS", "is_repeatable"])
        )
        self.assertFalse(bool(summary.loc["TOO_SPARSE", "is_repeatable"]))

    def test_last_weight_date_is_an_inclusive_upper_cutoff_only(self) -> None:
        """Retain pre-cutoff records and exclude only records after follow-up.

        Args:
            self: Test case providing temporal-censoring assertions.

        Returns:
            None.

        Side effects:
            None; source-shaped fixtures are not written or mutated.

        Statistical intent:
            Prevents post-outcome information leakage while confirming that the
            requested rule does not invent a lower observation cutoff.
        """
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "member_01",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                }
            ],
            engagement_rows=[
                {
                    "member_id": "member_01",
                    "activity_at": "2024-01-01",
                    "event_type": "BEFORE",
                },
                {
                    "member_id": "member_01",
                    "activity_at": "2025-01-10",
                    "event_type": "ON_CUTOFF",
                },
                {
                    "member_id": "member_01",
                    "activity_at": "2025-01-11",
                    "event_type": "AFTER",
                },
            ],
            module_rows=[
                {
                    "member_id": "member_01",
                    "questionnaire_title": "01_Core before",
                    "answered_at": "2024-01-01",
                },
                {
                    "member_id": "member_01",
                    "questionnaire_title": "05_Core on cutoff",
                    "answered_at": "2025-01-10",
                },
                {
                    "member_id": "member_01",
                    "questionnaire_title": "06_Core after",
                    "answered_at": "2025-01-11",
                },
            ],
        )

        result = build_member_features(data, cohort)
        member = result.member_features.set_index("member_id").loc["member_01"]

        self.assertEqual(member["engagement_breadth"], 2)
        self.assertEqual(member["module_core_count"], 2)
        self.assertEqual(result.audit_counts["retained_engagement_rows"], 2)
        self.assertEqual(
            result.audit_counts["excluded_engagement_after_last_weight"], 1
        )
        self.assertEqual(result.audit_counts["retained_module_rows"], 2)
        self.assertEqual(
            result.audit_counts["excluded_module_after_last_weight"], 1
        )

    def test_engagement_rate_protects_tenure_below_seven_days(self) -> None:
        """Divide repeatable volume by exactly ``max(tenure_days, 7)``.

        Args:
            self: Test case providing exact-rate assertions.

        Returns:
            None.

        Side effects:
            None; the 30-member fixture remains in memory.

        Statistical intent:
            Protects the requested denominator below and above seven days and
            guards against an unintended multiplication by seven.
        """
        member_rows = [
            {
                "member_id": f"member_{member_number:02d}",
                "last_weight_at": "2025-02-01",
                "tenure_days": 3.0 if member_number == 0 else 14.0,
            }
            for member_number in range(30)
        ]

        # Two retained events for each of 30 members make the event type
        # repeatable at both exact classification thresholds.
        engagement_rows = [
            {
                "member_id": f"member_{member_number:02d}",
                "activity_at": f"2025-01-{event_number + 1:02d}",
                "event_type": "REPEATED_EVENT",
            }
            for member_number in range(30)
            for event_number in range(2)
        ]
        data, cohort = _make_analysis_inputs(
            member_rows=member_rows,
            engagement_rows=engagement_rows,
            module_rows=[],
        )

        result = build_member_features(data, cohort)
        members = result.member_features.set_index("member_id")

        self.assertEqual(
            members.loc["member_00", "engagement_volume_repeatable"], 2
        )
        self.assertEqual(members.loc["member_00", "engagement_breadth"], 1)
        self.assertAlmostEqual(
            members.loc[
                "member_00", "engagement_volume_repeatable_rate"
            ],
            2.0 / 7.0,
        )
        self.assertAlmostEqual(
            members.loc[
                "member_01", "engagement_volume_repeatable_rate"
            ],
            2.0 / 14.0,
        )

    def test_modules_are_deduplicated_normalized_and_averaged(self) -> None:
        """Count unique member-title completions and derive domain features.

        Args:
            self: Test case providing curriculum-feature assertions.

        Returns:
            None.

        Side effects:
            None; module fixtures are neither written nor mutated.

        Statistical intent:
            Prevents repeated completions from inflating exposure while locking
            fixed denominators and the requested four-domain row mean.
        """
        module_rows = [
            {
                "member_id": "member_01",
                "questionnaire_title": "01_Core title",
                "answered_at": "2025-01-01",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "01_Core title",
                "answered_at": "2025-01-02",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "05_Second core title",
                "answered_at": "2025-01-03",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "MINDSET W01: Mindset title",
                "answered_at": "2025-01-04",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "NUTRITION W01: Nutrition title",
                "answered_at": "2025-01-05",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "NUTRITION W02: Nutrition title",
                "answered_at": "2025-01-06",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "PHYSICAL ACTIVITY W01: Title",
                "answered_at": "2025-01-07",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "PHYSICAL ACTIVITY W02: Title",
                "answered_at": "2025-01-08",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "PHYSICAL ACTIVITY W03: Title",
                "answered_at": "2025-01-09",
            },
            {
                "member_id": "member_01",
                "questionnaire_title": "PHYSICAL ACTIVITY W04: Title",
                "answered_at": "2025-01-10",
            },
        ]
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "member_01",
                    "last_weight_at": "2025-01-31",
                    "tenure_days": 30.0,
                }
            ],
            engagement_rows=[],
            module_rows=module_rows,
        )

        result = build_member_features(data, cohort)
        member = result.member_features.set_index("member_id").loc["member_01"]

        self.assertEqual(member["module_core_count"], 2)
        self.assertEqual(member["module_mindset_count"], 1)
        self.assertEqual(member["module_nutrition_count"], 2)
        self.assertEqual(member["module_physical_activity_count"], 4)
        self.assertAlmostEqual(member["module_core"], 2.0 / 9.0)
        self.assertAlmostEqual(member["module_mindset"], 1.0 / 4.0)
        self.assertAlmostEqual(member["module_nutrition"], 2.0 / 4.0)
        self.assertAlmostEqual(member["module_physical_activity"], 1.0)

        # The exact requested row mean retains all four domain proportions,
        # including their known deterministic linear dependency.
        expected_mean = (2.0 / 9.0 + 1.0 / 4.0 + 2.0 / 4.0 + 1.0) / 4.0
        self.assertAlmostEqual(member["module_mean"], expected_mean)
        self.assertEqual(result.audit_counts["duplicate_module_rows"], 1)

    def test_only_confirmed_module_identifiers_are_mapped(self) -> None:
        """Exclude unknown title identifiers from fixed-domain proportions.

        Args:
            self: Test case providing mapping and audit assertions.

        Returns:
            None.

        Side effects:
            None; all known and future-title fixtures remain in memory.

        Statistical intent:
            Prevents future curriculum additions from silently changing the
            confirmed fixed-scale module covariates.
        """
        valid_core_identifiers = [
            "01",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]
        invalid_core_identifiers = ["02", "03", "04", "13"]

        # Core identifiers are not a continuous numeric range: only 01 and
        # 05-12 belong to the confirmed nine-title curriculum denominator.
        module_rows = [
            {
                "member_id": "member_01",
                "questionnaire_title": f"{identifier}_Core title",
                "answered_at": "2025-01-01",
            }
            for identifier in (
                valid_core_identifiers + invalid_core_identifiers
            )
        ]

        # Each extension curriculum contains exactly W01-W04; W05 is included
        # as a future title that must remain explicitly unmapped.
        for prefix in ["MINDSET", "NUTRITION", "PHYSICAL ACTIVITY"]:
            for week in range(1, 6):
                module_rows.append(
                    {
                        "member_id": "member_01",
                        "questionnaire_title": (
                            f"{prefix} W{week:02d}: Curriculum title"
                        ),
                        "answered_at": "2025-01-01",
                    }
                )

        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "member_01",
                    "last_weight_at": "2025-01-31",
                    "tenure_days": 30.0,
                }
            ],
            engagement_rows=[],
            module_rows=module_rows,
        )

        result = build_member_features(data, cohort)
        member = result.member_features.set_index("member_id").loc["member_01"]

        self.assertEqual(member["module_core_count"], 9)
        self.assertEqual(member["module_mindset_count"], 4)
        self.assertEqual(member["module_nutrition_count"], 4)
        self.assertEqual(member["module_physical_activity_count"], 4)
        self.assertEqual(member["module_core"], 1.0)
        self.assertEqual(member["module_mindset"], 1.0)
        self.assertEqual(member["module_nutrition"], 1.0)
        self.assertEqual(member["module_physical_activity"], 1.0)
        self.assertEqual(member["module_mean"], 1.0)
        self.assertEqual(result.audit_counts["unmapped_module_completions"], 7)

    def test_zero_fill_audits_and_inputs_are_preserved(self) -> None:
        """Retain inactive members, reconcile audits, and avoid input mutation.

        Args:
            self: Test case providing zero-fill, audit, and immutability checks.

        Returns:
            None.

        Side effects:
            None; explicit equality checks confirm all inputs remain unchanged.

        Statistical intent:
            Preserves the cohort denominator and makes noncohort, post-outcome,
            and zero-activity feature accounting fully reconcilable.
        """
        data, cohort = _make_analysis_inputs(
            member_rows=[
                {
                    "member_id": "member_01",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 5.0,
                },
                {
                    "member_id": "member_02",
                    "last_weight_at": "2025-01-10",
                    "tenure_days": 10.0,
                },
            ],
            engagement_rows=[
                {
                    "member_id": "member_01",
                    "activity_at": "2025-01-09",
                    "event_type": "BEFORE",
                },
                {
                    "member_id": "member_01",
                    "activity_at": "2025-01-11",
                    "event_type": "AFTER",
                },
                {
                    "member_id": "outside_member",
                    "activity_at": "2025-01-09",
                    "event_type": "OUTSIDE",
                },
            ],
            module_rows=[
                {
                    "member_id": "member_01",
                    "questionnaire_title": "01_Core before",
                    "answered_at": "2025-01-09",
                },
                {
                    "member_id": "member_01",
                    "questionnaire_title": "05_Core after",
                    "answered_at": "2025-01-11",
                },
                {
                    "member_id": "outside_member",
                    "questionnaire_title": "06_Outside core",
                    "answered_at": "2025-01-09",
                },
            ],
        )
        engagement_before = data.engagement.copy(deep=True)
        modules_before = data.module_completions.copy(deep=True)
        members_before = cohort.members.copy(deep=True)

        result = build_member_features(data, cohort)
        members = result.member_features.set_index("member_id")
        audit = result.audit_counts

        zero_filled_columns = [
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
        ]
        self.assertEqual(list(members.index), ["member_01", "member_02"])
        self.assertTrue((members.loc["member_02", zero_filled_columns] == 0).all())

        # Source rows partition into noncohort and cohort rows; cohort rows then
        # partition into post-outcome, missing-date, and retained records.
        self.assertEqual(
            audit["source_engagement_rows"],
            audit["excluded_noncohort_engagement_rows"]
            + audit["cohort_engagement_rows"],
        )
        self.assertEqual(
            audit["cohort_engagement_rows"],
            audit["excluded_engagement_after_last_weight"]
            + audit["excluded_engagement_missing_date"]
            + audit["retained_engagement_rows"],
        )
        self.assertEqual(
            audit["source_module_rows"],
            audit["excluded_noncohort_module_rows"]
            + audit["cohort_module_rows"],
        )
        self.assertEqual(
            audit["cohort_module_rows"],
            audit["excluded_module_after_last_weight"]
            + audit["excluded_module_missing_date"]
            + audit["retained_module_rows"],
        )
        self.assertEqual(
            audit["retained_module_rows"],
            audit["duplicate_module_rows"]
            + audit["distinct_module_completions"],
        )

        pd.testing.assert_frame_equal(data.engagement, engagement_before)
        pd.testing.assert_frame_equal(data.module_completions, modules_before)
        pd.testing.assert_frame_equal(cohort.members, members_before)


if __name__ == "__main__":
    unittest.main()
