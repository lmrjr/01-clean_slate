"""Tests for source-data loading and analysis-cohort construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nineam_health_analysis.nineam_cohort_selection import (
    PRIMARY_MEMBER_TYPES,
    build_analysis_cohort,
    restrict_to_primary_member_types,
)
from nineam_health_analysis.nineam_data_loading import (
    CaseStudyData,
    load_case_study_data,
)

MODULE_FILE = (
    "Data Analyst Case Study Doc 1 - "
    "12 Weeks Weight Loss Modules Completion.csv"
)
DEMOGRAPHICS_FILE = "Data Analyst Case Study Doc 2 - Demographics.csv"
ENGAGEMENT_FILE = "Data Analyst Case Study Doc 3 - Engagement Data.csv"
WEIGHT_FILE = "Data Analyst Case Study Doc 4 - BW_Detail.csv"


class DataAndCohortTestCase(unittest.TestCase):
    """Exercise observable loading and filtering behavior with real files."""

    def _write_source_files(
        self,
        data_dir: Path,
        demographics_rows: list[dict[str, object]],
        weight_rows: list[dict[str, object]],
        *,
        omitted_weight_columns: tuple[str, ...] = (),
    ) -> None:
        """Create realistic, isolated source extracts for a test.

        Args:
            data_dir: Temporary directory that receives the four source files.
            demographics_rows: Member enrollment records to serialize.
            weight_rows: Member-level paired-weight records to serialize.
            omitted_weight_columns: Required fields to remove for schema tests.

        Returns:
            None.

        Side effects:
            Writes four UTF-16 tab-delimited files under ``data_dir``; input
            records are not mutated.

        Statistical intent:
            Controls cohort conditions without reading production member data.
        """
        # Ancillary tables are required by the loader but do not affect cohort
        # eligibility, so one enrolled member is enough to exercise ingestion.
        first_member = str(demographics_rows[0]["Readable Id"])
        tables = {
            MODULE_FILE: pd.DataFrame(
                [
                    {
                        "Readable Id": first_member,
                        "Questionnaire Title (All Questionnaire Records)": (
                            "01_Introduction&Goal Setting"
                        ),
                        "Day of Answered At (All Questionnaire Records)": (
                            "21 February 2025"
                        ),
                        "": None,
                    }
                ]
            ),
            DEMOGRAPHICS_FILE: pd.DataFrame(demographics_rows).assign(
                **{"": None}
            ),
            ENGAGEMENT_FILE: pd.DataFrame(
                [
                    {
                        "Readable Id": first_member,
                        "Day of Activity Timestamp": "February 13, 2025",
                        "Type": "REGISTRATION",
                        "": None,
                    }
                ]
            ),
            WEIGHT_FILE: pd.DataFrame(weight_rows)
            .drop(columns=list(omitted_weight_columns), errors="ignore")
            .assign(**{"": None}),
        }

        # Preserve the source system's UTF-16 tab-delimited format and its
        # harmless trailing empty field so tests cover the real ingestion path.
        for filename, table in tables.items():
            table.to_csv(
                data_dir / filename,
                sep="\t",
                encoding="utf-16",
                index=False,
            )

    @staticmethod
    def _demographic_row(
        member_id: str | None,
        status: str = "ACTIVE",
    ) -> dict[str, object]:
        """Build one minimally complete enrollment record.

        Args:
            member_id: Identifier to place in the source-shaped record.
            status: Subscription status used by cohort eligibility tests.

        Returns:
            A new source-shaped demographic dictionary.

        Side effects:
            None; inputs are not mutated.

        Statistical intent:
            Defines the enrollment denominator and eligibility stratum.
        """
        return {
            "Readable Id": member_id,
            "Day of Start Date": "February 13, 2025",
            "Status (Subscriptions)": status,
            "cancellation_date": None,
            "Sex": "FEMALE",
            "ethnicity": "WHITE",
        }

    @staticmethod
    def _weight_row(
        member_id: str | None,
        *,
        member_type: str = "Active GLP-1 for Weight-loss",
        first_weight: float | None = 200.0,
        last_weight: float | None = 180.0,
        weight_days: float | None = 84.0,
    ) -> dict[str, object]:
        """Build one source-shaped paired-weight record.

        Args:
            member_id: Identifier that links weights to enrollment.
            first_weight: Baseline outcome, or ``None`` for missingness tests.
            last_weight: Follow-up outcome, or ``None`` for missingness tests.
            weight_days: Interval between outcomes, including invalid fixtures.

        Returns:
            A new dictionary containing paired outcomes and their interval.

        Side effects:
            None; inputs are not mutated.

        Statistical intent:
            Represents one member's paired longitudinal response.
        """
        # The source difference is positive for weight loss; calculating it in
        # the fixture keeps internally consistent records without testing it.
        difference = (
            None
            if first_weight is None or last_weight is None
            else first_weight - last_weight
        )
        return {
            "User Id": member_id,
            "Weight-loss Member Type": member_type,
            "Day of BW First Measurement Effective": "February 12, 2025",
            "Day of BW Last Measurement Effective": "May 7, 2025",
            "First": first_weight,
            "Last": last_weight,
            "Diff": difference,
            "BW Days between First/Last": weight_days,
            "days in weight-loss program": 90.0,
        }

    def test_load_case_study_data_reads_and_canonicalizes_source_files(self) -> None:
        """Verify source files load into canonical typed schemas.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Protects the typed variables that define later model inputs.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [self._demographic_row("member-1")],
                [self._weight_row("member-1")],
            )

            data = load_case_study_data(data_dir)

        self.assertIsInstance(data, CaseStudyData)
        self.assertListEqual(
            list(data.module_completions.columns),
            ["member_id", "questionnaire_title", "answered_at"],
        )
        self.assertListEqual(
            list(data.demographics.columns),
            [
                "member_id",
                "start_date",
                "subscription_status",
                "cancellation_date",
                "sex",
                "ethnicity",
            ],
        )
        self.assertListEqual(
            list(data.engagement.columns),
            ["member_id", "activity_at", "event_type"],
        )
        self.assertListEqual(
            list(data.body_weights.columns),
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
        )
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                data.module_completions["answered_at"]
            )
        )
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(data.demographics["start_date"])
        )
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                data.engagement["activity_at"]
            )
        )
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                data.body_weights["last_weight_at"]
            )
        )
        self.assertEqual(data.body_weights.loc[0, "first_weight"], 200.0)

    def test_load_case_study_data_reports_missing_required_columns(self) -> None:
        """Verify missing analytical fields cause a clear schema failure.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Prevents models from running with silently incomplete variables.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [self._demographic_row("member-1")],
                [self._weight_row("member-1")],
                omitted_weight_columns=("First",),
            )

            with self.assertRaisesRegex(ValueError, "First"):
                load_case_study_data(data_dir)

    def test_build_analysis_cohort_keeps_only_eligible_statuses(self) -> None:
        """Verify only ACTIVE and FINISHED subscriptions are retained.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Fixes the analysis population to the confirmed eligibility rule.
        """
        demographics = [
            self._demographic_row("active", "ACTIVE"),
            self._demographic_row("finished", "FINISHED"),
            self._demographic_row("paused", "PAUSED"),
        ]
        weights = [
            self._weight_row("active"),
            self._weight_row("finished"),
            self._weight_row("paused"),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(
            cohort.members["member_id"].tolist(),
            ["active", "finished"],
        )
        self.assertEqual(cohort.audit_counts["excluded_ineligible_status"], 1)
        self.assertEqual(cohort.audit_counts["included_members"], 2)

    def test_build_analysis_cohort_excludes_missing_weights(self) -> None:
        """Verify members missing either paired outcome are excluded.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Ensures every retained member contributes a complete outcome pair.
        """
        member_ids = ["valid", "missing-first", "missing-last"]
        demographics = [self._demographic_row(value) for value in member_ids]
        weights = [
            self._weight_row("valid"),
            self._weight_row("missing-first", first_weight=None),
            self._weight_row("missing-last", last_weight=None),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.members["member_id"].tolist(), ["valid"])
        self.assertEqual(cohort.audit_counts["excluded_missing_weights"], 2)

    def test_build_analysis_cohort_excludes_nonpositive_weights(self) -> None:
        """Verify zero and negative endpoint weights are excluded.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Keeps percentage loss defined on plausible positive outcomes.
        """
        member_ids = ["valid", "zero-first", "negative-last"]
        demographics = [self._demographic_row(value) for value in member_ids]
        weights = [
            self._weight_row("valid"),
            self._weight_row("zero-first", first_weight=0.0),
            self._weight_row("negative-last", last_weight=-1.0),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.members["member_id"].tolist(), ["valid"])
        self.assertEqual(cohort.audit_counts["excluded_nonpositive_weights"], 2)

    def test_build_analysis_cohort_excludes_nonpositive_weight_intervals(self) -> None:
        """Verify zero and reversed measurement intervals are excluded.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Preserves a valid temporal ordering for longitudinal change.
        """
        member_ids = ["valid", "zero-days", "negative-days"]
        demographics = [self._demographic_row(value) for value in member_ids]
        weights = [
            self._weight_row("valid", weight_days=84.0),
            self._weight_row("zero-days", weight_days=0.0),
            self._weight_row("negative-days", weight_days=-1.0),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.members["member_id"].tolist(), ["valid"])
        self.assertEqual(
            cohort.audit_counts["excluded_nonpositive_weight_days"],
            2,
        )

    def test_build_analysis_cohort_counts_members_without_weight_rows(self) -> None:
        """Verify attrition counts enrolled members lacking a weight row.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Anchors attrition to the full enrollment denominator.
        """
        demographics = [
            self._demographic_row("has-weight"),
            self._demographic_row("no-weight"),
        ]
        weights = [self._weight_row("has-weight")]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(
            cohort.members["member_id"].tolist(),
            ["has-weight"],
        )
        self.assertEqual(cohort.audit_counts["source_demographic_rows"], 2)
        self.assertEqual(
            cohort.audit_counts["excluded_missing_body_weight_row"],
            1,
        )
        self.assertEqual(cohort.audit_counts["included_members"], 1)

    def test_build_analysis_cohort_excludes_missing_weight_days(self) -> None:
        """Verify pairs with an unknown measurement interval are excluded.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Requires observable follow-up timing for longitudinal inference.
        """
        demographics = [
            self._demographic_row("valid"),
            self._demographic_row("missing-days"),
        ]
        weights = [
            self._weight_row("valid"),
            self._weight_row("missing-days", weight_days=None),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.members["member_id"].tolist(), ["valid"])
        self.assertEqual(
            cohort.audit_counts["excluded_missing_weight_days"],
            1,
        )

    def test_build_analysis_cohort_reports_orphan_weight_rows(self) -> None:
        """Verify weight rows without enrollment are reported as orphans.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Separates source-quality defects from enrolled-member attrition.
        """
        demographics = [self._demographic_row("matched")]
        weights = [
            self._weight_row("matched"),
            self._weight_row("orphan"),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.members["member_id"].tolist(), ["matched"])
        self.assertEqual(cohort.audit_counts["orphan_weight_rows"], 1)
        self.assertEqual(
            cohort.audit_counts["excluded_missing_body_weight_row"],
            0,
        )

    def test_build_analysis_cohort_rejects_duplicate_member_ids(self) -> None:
        """Verify duplicated member-level records are rejected.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Enforces one enrollment and one outcome pair per member.
        """
        cases = [
            (
                [
                    self._demographic_row("duplicate"),
                    self._demographic_row("duplicate"),
                ],
                [self._weight_row("duplicate")],
                "Demographics",
            ),
            (
                [self._demographic_row("duplicate")],
                [
                    self._weight_row("duplicate"),
                    self._weight_row("duplicate"),
                ],
                "Body weights",
            ),
        ]

        for demographics, weights, source_name in cases:
            with self.subTest(source_name=source_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    data_dir = Path(temporary_directory)
                    self._write_source_files(data_dir, demographics, weights)
                    data = load_case_study_data(data_dir)

                    with self.assertRaisesRegex(
                        ValueError,
                        f"{source_name}.*duplicate",
                    ):
                        build_analysis_cohort(data)

    def test_build_analysis_cohort_rejects_null_or_blank_member_ids(self) -> None:
        """Verify null and blank member identifiers are rejected.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Prevents silent member loss or false linkage across sources.
        """
        cases = [
            (
                [self._demographic_row(None)],
                [self._weight_row("valid")],
                "Demographics",
            ),
            (
                [self._demographic_row("valid")],
                [self._weight_row("   ")],
                "Body weights",
            ),
        ]

        for demographics, weights, source_name in cases:
            with self.subTest(source_name=source_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    data_dir = Path(temporary_directory)
                    self._write_source_files(data_dir, demographics, weights)
                    data = load_case_study_data(data_dir)

                    with self.assertRaisesRegex(
                        ValueError,
                        f"{source_name}.*null or blank",
                    ):
                        build_analysis_cohort(data)

    def test_build_analysis_cohort_creates_two_long_rows_per_member(self) -> None:
        """Verify retained members yield ordered baseline and follow-up rows.

        Args:
            self: Test case providing fixture and assertion helpers.

        Returns:
            None.

        Side effects:
            Creates and removes temporary source files.

        Statistical intent:
            Protects the paired response and percentage-loss definitions.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [self._demographic_row("member-1", "FINISHED")],
                [
                    self._weight_row(
                        "member-1",
                        first_weight=200.0,
                        last_weight=180.0,
                    )
                ],
            )
            cohort = build_analysis_cohort(load_case_study_data(data_dir))

        self.assertListEqual(cohort.long_weights["time"].tolist(), [0, 1])
        self.assertListEqual(
            cohort.long_weights["occasion"].tolist(),
            ["first", "last"],
        )
        self.assertListEqual(
            cohort.long_weights["weight"].tolist(),
            [200.0, 180.0],
        )
        self.assertAlmostEqual(cohort.members.loc[0, "percentage_loss"], 10.0)

    def test_primary_cohort_keeps_only_confirmed_comparable_types(self) -> None:
        """Restrict complete pairs without altering their original audit flow.

        The behavior would fail if a nonprimary type is retained or the
        restriction overwrites the complete-pair denominator.
        """
        member_types = [
            "Active GLP-1 for Weight-loss",
            "Coaching Only",
            "GLP-1 for Diabetes",
            "Active Weight-loss Medication",
            "",
        ]
        demographics = [
            self._demographic_row(f"member-{index}")
            for index in range(len(member_types))
        ]
        weights = [
            self._weight_row(
                f"member-{index}",
                member_type=member_type,
            )
            for index, member_type in enumerate(member_types)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(data_dir, demographics, weights)
            complete_pair = build_analysis_cohort(load_case_study_data(data_dir))
            cohort = restrict_to_primary_member_types(complete_pair)

        self.assertEqual(set(cohort.members["member_type"]), set(PRIMARY_MEMBER_TYPES))
        self.assertEqual(
            cohort.audit_counts["pre_member_type_restriction_members"],
            5,
        )
        self.assertEqual(
            cohort.audit_counts["excluded_nonprimary_member_type"],
            3,
        )
        self.assertEqual(cohort.audit_counts["included_members"], 2)
        self.assertEqual(len(cohort.long_weights), 4)
        self.assertEqual(complete_pair.audit_counts["included_members"], 5)

    def test_primary_cohort_restriction_does_not_alias_complete_pair_tables(self) -> None:
        """Return independent cohort tables while preserving complete pairs.

        The behavior would fail if the restriction mutates the original
        complete-pair result or returns either dataframe or audit mapping by
        reference.
        """
        member_types = [
            "Active GLP-1 for Weight-loss",
            "Coaching Only",
            "GLP-1 for Diabetes",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [
                    self._demographic_row(f"member-{index}")
                    for index in range(len(member_types))
                ],
                [
                    self._weight_row(
                        f"member-{index}",
                        member_type=member_type,
                    )
                    for index, member_type in enumerate(member_types)
                ],
            )
            complete_pair = build_analysis_cohort(load_case_study_data(data_dir))
            original_members = complete_pair.members.copy(deep=True)
            original_long_weights = complete_pair.long_weights.copy(deep=True)
            original_audit = dict(complete_pair.audit_counts)
            restricted = restrict_to_primary_member_types(complete_pair)

        self.assertIsNot(restricted.members, complete_pair.members)
        self.assertIsNot(restricted.long_weights, complete_pair.long_weights)
        self.assertIsNot(restricted.audit_counts, complete_pair.audit_counts)
        restricted.members.loc[0, "member_type"] = "changed"
        restricted.long_weights.loc[0, "weight"] = -1.0
        restricted.audit_counts["included_members"] = 999
        pd.testing.assert_frame_equal(complete_pair.members, original_members)
        pd.testing.assert_frame_equal(
            complete_pair.long_weights,
            original_long_weights,
        )
        self.assertEqual(complete_pair.audit_counts, original_audit)

    def test_primary_cohort_audits_exact_excluded_member_type_labels(self) -> None:
        """Count each approved excluded source label without hard-coding totals.

        The behavior would fail if a source label is misspelled, collapsed into
        a generic exclusion count, or omitted from the reconcilable subtype
        audit.
        """
        member_types = [
            "Active GLP-1 for Weight-loss",
            "Coaching Only",
            "Active GLP-1 for Diabetes",
            "Active GLP-1 for Diabetes",
            "Active Generic Medication for Weight-loss (NOT on GLP-1 for weight-loss)",
            "Active Generic Medication for Weight-loss (NOT on GLP-1 for weight-loss)",
            "Active Generic Medication for Weight-loss (NOT on GLP-1 for weight-loss)",
            "Null",
            "Null",
            "Null",
            "Null",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [
                    self._demographic_row(f"member-{index}")
                    for index in range(len(member_types))
                ],
                [
                    self._weight_row(
                        f"member-{index}",
                        member_type=member_type,
                    )
                    for index, member_type in enumerate(member_types)
                ],
            )
            cohort = restrict_to_primary_member_types(
                build_analysis_cohort(load_case_study_data(data_dir))
            )

        audit = cohort.audit_counts
        self.assertEqual(audit["excluded_active_glp1_diabetes"], 2)
        self.assertEqual(
            audit["excluded_active_generic_weight_loss_medication"],
            3,
        )
        self.assertEqual(audit["excluded_null_member_type"], 4)
        self.assertEqual(
            audit["excluded_nonprimary_member_type"],
            audit["excluded_active_glp1_diabetes"]
            + audit["excluded_active_generic_weight_loss_medication"]
            + audit["excluded_null_member_type"],
        )

    def test_weight_loss_fields_use_confirmed_directions(self) -> None:
        """Expose pounds lost and the inclusive five-percent response rule.

        The behavior would fail if the outcome direction is reversed or the
        response threshold becomes strict rather than inclusive.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [self._demographic_row("member-1")],
                [
                    self._weight_row(
                        "member-1",
                        first_weight=200.0,
                        last_weight=190.0,
                    )
                ],
            )
            member = build_analysis_cohort(
                load_case_study_data(data_dir)
            ).members.iloc[0]

        self.assertEqual(member["absolute_weight_loss"], 10.0)
        self.assertEqual(member["percentage_loss"], 5.0)
        self.assertTrue(member["weight_loss_success_5pct"])

    def test_weight_loss_success_uses_the_unrounded_inclusive_threshold(self) -> None:
        """Classify response directly from percentage loss at five percent.

        The behavior would fail if percentages are rounded before response
        classification or if the threshold changes from inclusive to strict.
        """
        weight_pairs = {
            "below": (1_000_000.0, 950_000.01),
            "at": (1_000_000.0, 950_000.0),
            "above": (1_000_000.0, 949_999.99),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._write_source_files(
                data_dir,
                [self._demographic_row(member_id) for member_id in weight_pairs],
                [
                    self._weight_row(
                        member_id,
                        first_weight=first_weight,
                        last_weight=last_weight,
                    )
                    for member_id, (first_weight, last_weight) in weight_pairs.items()
                ],
            )
            members = build_analysis_cohort(
                load_case_study_data(data_dir)
            ).members.set_index("member_id")

        self.assertAlmostEqual(members.loc["below", "percentage_loss"], 4.999999)
        self.assertEqual(members.loc["below", "weight_loss_success_5pct"], False)
        self.assertEqual(members.loc["at", "percentage_loss"], 5.0)
        self.assertEqual(members.loc["at", "weight_loss_success_5pct"], True)
        self.assertAlmostEqual(members.loc["above", "percentage_loss"], 5.000001)
        self.assertEqual(members.loc["above", "weight_loss_success_5pct"], True)


if __name__ == "__main__":
    unittest.main()
