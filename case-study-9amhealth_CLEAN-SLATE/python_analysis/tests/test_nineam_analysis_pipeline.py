"""Test deterministic aggregate orchestration and the command-line entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd

from nineam_health_analysis import nineam_analysis_pipeline as pipeline_module
from nineam_health_analysis.nineam_analysis_pipeline import (
    AnalysisConfig,
    AnalysisResult,
    run_analysis,
    write_analysis_outputs,
)
from nineam_health_analysis.nineam_reporting import REPORTING_SCHEMAS
from nineam_health_analysis.nineam_visualizations import FIGURE_STEMS

MODULE_FILE = (
    "Data Analyst Case Study Doc 1 - "
    "12 Weeks Weight Loss Modules Completion.csv"
)
DEMOGRAPHICS_FILE = "Data Analyst Case Study Doc 2 - Demographics.csv"
ENGAGEMENT_FILE = "Data Analyst Case Study Doc 3 - Engagement Data.csv"
WEIGHT_FILE = "Data Analyst Case Study Doc 4 - BW_Detail.csv"
ACTIVITY_TYPES = (
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
EXPECTED_OUTPUT_PATHS = {
    *(Path("tables") / filename for filename in REPORTING_SCHEMAS),
    *(
        Path("figures") / f"{stem}.{extension}"
        for stem in FIGURE_STEMS
        for extension in ("png", "svg")
    ),
    Path("nineam_analysis_metadata.json"),
}


def _write_pipeline_fixture(
    data_dir: Path,
    n_members: int = 40,
    *,
    include_rare_subgroups: bool = False,
) -> None:
    """Write a complete source-shaped fixture with estimable model variation.

    Args:
        data_dir: Empty directory that receives four case-study extracts.
        n_members: Number of eligible members; at least 30 supports one
            repeatable engagement type under the confirmed source rule.
        include_rare_subgroups: Whether one member receives unique sex and
            ethnicity labels for no-suppression and locking-boundary tests.

    Returns:
        None.

    Side effects:
        Writes four UTF-16, tab-delimited source files under ``data_dir``.

    Statistical intent:
        Provides paired outcomes, crossed factors, engagement variation, and
        separate curriculum patterns so every pipeline model is estimable.
    """
    if n_members < 30:
        raise ValueError("Pipeline fixture requires at least 30 members")
    member_number = np.arange(n_members)
    member_ids = [f"fixture_member_{index:03d}" for index in member_number]
    member_types = np.where(
        member_number % 2 == 0,
        "Coaching Only",
        "Active GLP-1 for Weight-loss",
    )
    sexes = np.where(member_number % 4 < 2, "FEMALE", "MALE").astype(object)
    ethnicities = np.resize(
        np.array(["Group 1", "Group 2", "Group 3"], dtype=object),
        n_members,
    )
    first_weight = (
        165.0
        + 2.2 * member_number
        + 9.0 * (member_types == "Active GLP-1 for Weight-loss")
    )
    percentage_loss = (
        5.0
        + 0.028 * (first_weight - 180.0)
        + 1.4 * (member_types == "Active GLP-1 for Weight-loss")
        + np.resize(np.array([-0.7, 0.2, 0.8, -0.3, 0.5]), n_members)
    )
    if include_rare_subgroups:
        # Approved scientific summaries retain observed categories even for a
        # one-member level; the generic contrast is excluded only from locking.
        sexes[0] = "SINGLETON_SEX_PRIVATE"
        ethnicities[0] = "SINGLETON_ETHNICITY_PRIVATE"
    last_weight = first_weight * (1.0 - percentage_loss / 100.0)

    demographics = pd.DataFrame(
        {
            "Readable Id": member_ids,
            "Day of Start Date": "January 1, 2025",
            "Status (Subscriptions)": "ACTIVE",
            "cancellation_date": None,
            "Sex": sexes,
            "ethnicity": ethnicities,
            "": None,
        }
    )
    body_weights = pd.DataFrame(
        {
            "User Id": member_ids,
            "Weight-loss Member Type": member_types,
            "Day of BW First Measurement Effective": "January 2, 2025",
            "Day of BW Last Measurement Effective": "April 2, 2025",
            "First": first_weight,
            "Last": last_weight,
            "Diff": first_weight - last_weight,
            "BW Days between First/Last": 70.0 + (7 * member_number) % 41,
            "days in weight-loss program": 70.0 + member_number,
            "": None,
        }
    )

    engagement_rows: list[dict[str, object]] = []
    for index, member_id in enumerate(member_ids):
        # Two or more step events for every member make one type repeatable.
        # Each remaining approved type appears sparsely so reporting can
        # zero-fill the complete prespecified 19-activity dictionary.
        for event_number in range(2 + index % 3):
            engagement_rows.append(
                {
                    "Readable Id": member_id,
                    "Day of Activity Timestamp": (
                        f"February {1 + event_number}, 2025"
                    ),
                    "Type": "RECORD_STEPS",
                    "": None,
                }
            )
        sparse_types = tuple(
            event_type for event_type in ACTIVITY_TYPES if event_type != "RECORD_STEPS"
        )
        if index < len(sparse_types):
            engagement_rows.append(
                {
                    "Readable Id": member_id,
                    "Day of Activity Timestamp": "March 1, 2025",
                    "Type": sparse_types[index],
                    "": None,
                }
            )
    engagement = pd.DataFrame(engagement_rows)

    module_rows: list[dict[str, object]] = []
    module_patterns = (
        ("01_Introduction",),
        ("01_Introduction", "05_Core Practice"),
        ("01_Introduction", "MINDSET W01: Practice"),
        ("01_Introduction", "NUTRITION W01: Practice"),
        ("01_Introduction", "PHYSICAL ACTIVITY W01: Practice"),
        ("01_Introduction", "05_Core Practice", "NUTRITION W01: Practice"),
    )
    for index, member_id in enumerate(member_ids):
        # Cycling six literal curricula avoids making any domain a deterministic
        # copy of member type or sex in the penalized design.
        for title in module_patterns[index % len(module_patterns)]:
            module_rows.append(
                {
                    "Readable Id": member_id,
                    "Questionnaire Title (All Questionnaire Records)": title,
                    "Day of Answered At (All Questionnaire Records)": (
                        "March 10, 2025"
                    ),
                    "": None,
                }
            )
    modules = pd.DataFrame(module_rows)

    for filename, table in (
        (MODULE_FILE, modules),
        (DEMOGRAPHICS_FILE, demographics),
        (ENGAGEMENT_FILE, engagement),
        (WEIGHT_FILE, body_weights),
    ):
        # Match the source system's actual UTF-16 TSV representation despite
        # the CSV extension; the trailing blank field tests loader cleanup.
        table.to_csv(
            data_dir / filename,
            sep="\t",
            encoding="utf-16",
            index=False,
        )


def _load_cli_module() -> ModuleType:
    """Load the repository CLI script as an importable module for real calls.

    Args:
        None.

    Returns:
        The executed ``nineam_run_analysis`` script module.

    Side effects:
        Executes module-level imports but does not invoke its ``main`` function.

    Statistical intent:
        Tests the actual entry point without replacing analytical dependencies
        with mocks or starting a separate environment-dependent process.
    """
    script_path = Path(__file__).parents[1] / "scripts" / "nineam_run_analysis.py"
    specification = importlib.util.spec_from_file_location(
        "nineam_run_analysis_test_module",
        script_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Unable to load the analysis CLI module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _UnserializableCell:
    """Provide a controlled late CSV serialization failure for rollback tests."""

    def __str__(self) -> str:
        """Raise when pandas attempts to serialize this diagnostic cell.

        Args:
            self: The controlled failing value.

        Returns:
            Never returns normally.

        Side effects:
            Raises a runtime error during CSV text conversion.

        Statistical intent:
            Verifies all payloads serialize before existing aggregate artifacts
            are touched, without mocking pandas behavior.
        """
        raise RuntimeError("controlled late serialization failure")


class AnalysisPipelineTestCase(unittest.TestCase):
    """Exercise pipeline validation, orchestration, and aggregate-only outputs."""

    def test_pipeline_exposes_locked_model_and_scientific_reporting_tables(
        self,
    ) -> None:
        """Integrate the primary cohort, locked HC3 fit, and all closed tables."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=67,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )

        self.assertEqual(result.metadata["included_members"], 40)
        self.assertEqual(result.metadata["complete_pair_eligible_members"], 40)
        self.assertEqual(result.metadata["excluded_nonprimary_members"], 0)
        self.assertEqual(result.metadata["reference_member_type"], "Coaching Only")
        self.assertEqual(result.metadata["responder_threshold_percentage"], 5.0)
        self.assertEqual(result.metadata["locked_covariance_estimator"], "HC3")
        self.assertEqual(result.metadata["locked_inference_status"], "conditional_exploratory")
        self.assertEqual(
            result.metadata["base_model_winner_rule"],
            "lowest_mean_rmse_then_lowest_mean_mae_then_percentage_loss_ols_on_exact_tie",
        )
        self.assertEqual(
            result.locked_model.winning_base_model,
            result.metadata["winning_base_model"],
        )
        self.assertEqual(tuple(result.reporting_tables), tuple(REPORTING_SCHEMAS))
        for filename, schema in REPORTING_SCHEMAS.items():
            table = result.reporting_tables[filename]
            self.assertEqual(tuple(table.columns), schema)
            self.assertFalse(
                any("member_id" in str(column).casefold() for column in table.columns)
            )
            self.assertFalse(
                table.astype("string")
                .eq("<SUPPRESSED_RARE_LEVELS>")
                .any(axis=None)
            )

    def test_writer_publishes_exact_nested_set_and_deterministic_bytes(self) -> None:
        """Publish fifteen CSVs, eighteen figures, and one metadata file twice."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            first_output = root / "first"
            second_output = root / "second"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            config = AnalysisConfig(
                seed=69,
                cv_folds=2,
                cv_repeats=1,
                stability_resamples=2,
                lambda_ratios=(1.0, 0.2),
            )
            first = write_analysis_outputs(run_analysis(data_dir, config=config), first_output)
            second = write_analysis_outputs(run_analysis(data_dir, config=config), second_output)

            first_relative = {path.relative_to(first_output) for path in first}
            second_relative = {path.relative_to(second_output) for path in second}
            self.assertEqual(first_relative, EXPECTED_OUTPUT_PATHS)
            self.assertEqual(second_relative, EXPECTED_OUTPUT_PATHS)
            self.assertEqual(len(first), 34)
            first_hashes = {
                path.relative_to(first_output): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in first
            }
            second_hashes = {
                path.relative_to(second_output): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in second
            }
            self.assertEqual(first_hashes, second_hashes)
            for path in first:
                self.assertNotIn(b"fixture_member_", path.read_bytes())

            metadata = json.loads(
                (first_output / "nineam_analysis_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("minimum_reporting_cell_size", metadata)
            self.assertNotIn("rare_level_reporting_label", metadata)
            self.assertNotIn("SUPPRESSED", json.dumps(metadata))
            self.assertEqual(metadata["output_files"], [
                path.as_posix()
                for path in sorted(EXPECTED_OUTPUT_PATHS, key=lambda value: value.as_posix())
            ])

    def test_noncanonical_sex_contrast_is_reported_but_never_locked(self) -> None:
        """Retain generic design rows while locking only approved candidates."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir, include_rare_subgroups=True)
            weight_path = data_dir / WEIGHT_FILE
            weights = pd.read_csv(weight_path, sep="\t", encoding="utf-16")
            weights.loc[0, "Last"] = float(weights.loc[0, "First"]) * (1.0 - 0.0388)
            weights.loc[0, "Diff"] = (
                float(weights.loc[0, "First"]) - float(weights.loc[0, "Last"])
            )
            weights.to_csv(
                weight_path,
                sep="\t",
                encoding="utf-16",
                index=False,
            )
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=71,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )

        extra_candidate = "sex[SINGLETON_SEX_PRIVATE]"
        reported = pd.concat(
            [
                result.reporting_tables["nineam_lasso_mean_selection.csv"],
                result.reporting_tables["nineam_lasso_domain_selection.csv"],
            ],
            ignore_index=True,
        ).query("candidate == @extra_candidate")
        self.assertEqual(len(reported), 2)
        self.assertFalse(reported["eligible_for_locked_model"].any())
        self.assertTrue(reported["exclusion_reason"].str.contains("canonical").all())
        self.assertNotIn(extra_candidate, result.locked_model.selected_candidates)

    def test_pipeline_restricts_models_to_the_two_confirmed_member_types(self) -> None:
        """Keep nonprimary complete-pair members only in the cohort-flow audit.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Creates and removes isolated source fixture files.

        Statistical intent:
            Proves the pipeline applies the confirmed comparability restriction
            before feature construction, resampling, or model fitting.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            weight_path = data_dir / WEIGHT_FILE
            weights = pd.read_csv(weight_path, sep="\t", encoding="utf-16")
            weights.loc[:2, "Weight-loss Member Type"] = [
                "Active GLP-1 for Diabetes",
                "Active Generic Weight-loss Medication",
                "Null",
            ]
            weights.to_csv(
                weight_path,
                sep="\t",
                encoding="utf-16",
                index=False,
            )

            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=73,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )

        cohort_counts = result.cohort_summary.set_index("metric")["count"]
        self.assertEqual(cohort_counts["pre_member_type_restriction_members"], 40)
        self.assertEqual(cohort_counts["excluded_nonprimary_member_type"], 3)
        self.assertEqual(cohort_counts["included_members"], 37)
        self.assertTrue(result.model_summary["n_members"].eq(37).all())
        feature_count = result.feature_summary.loc[
            result.feature_summary["category"].eq("feature_audit")
            & result.feature_summary["variable"].eq("included_members"),
            "value",
        ].iloc[0]
        self.assertEqual(feature_count, 37.0)

    def test_analysis_config_rejects_invalid_resampling_controls(self) -> None:
        """Reject degenerate seeds, folds, repeats, resamples, and lambda grids.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents invalid resampling designs and non-identifiable tuning grids
            before any source data are loaded.
        """
        invalid_arguments = (
            {"seed": -1},
            {"cv_folds": 1},
            {"cv_repeats": 0},
            {"stability_resamples": 0},
            {"lambda_ratios": (0.5, 0.5)},
            {"lambda_ratios": (0.0, 0.5)},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    AnalysisConfig(**arguments)

    def test_small_fixture_runs_every_model_and_aggregate_diagnostic(self) -> None:
        """Run all confirmed paths and expose no member-level result table.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Creates and removes isolated source fixture files.

        Statistical intent:
            Verifies common-fold base comparison, separate module LASSO runs,
            and explicitly exploratory fixed-prewhitening longitudinal LASSO.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            config = AnalysisConfig(
                seed=71,
                cv_folds=2,
                cv_repeats=1,
                stability_resamples=4,
                lambda_ratios=(1.0, 0.4, 0.1),
            )

            with (
                patch(
                    "nineam_health_analysis.nineam_analysis_pipeline."
                    "select_lambda_by_grouped_cv",
                    wraps=pipeline_module.select_lambda_by_grouped_cv,
                ) as cv_spy,
                patch(
                    "nineam_health_analysis.nineam_analysis_pipeline."
                    "stability_select_lasso",
                    wraps=pipeline_module.stability_select_lasso,
                ) as stability_spy,
            ):
                result = run_analysis(data_dir, config=config)

        self.assertIsInstance(result, AnalysisResult)
        included = result.cohort_summary.loc[
            result.cohort_summary["metric"].eq("included_members"),
            "count",
        ].iloc[0]
        self.assertEqual(included, 40)
        self.assertEqual(
            set(result.model_summary["model"]),
            {
                "raw_un_additive_gls",
                "log_cs_additive_gls",
                "log_cs_time_by_member_type_sensitivity",
                "percentage_loss_ols",
            },
        )
        model_contracts = result.model_summary.set_index("model").loc[
            :,
            [
                "outcome_scale",
                "covariance_structure",
                "include_time_by_member_type",
            ],
        ]
        self.assertEqual(
            tuple(model_contracts.loc["raw_un_additive_gls"]),
            ("raw", "unstructured", False),
        )
        self.assertEqual(
            tuple(model_contracts.loc["log_cs_additive_gls"]),
            ("log", "compound_symmetry", False),
        )
        self.assertEqual(
            tuple(
                model_contracts.loc[
                    "log_cs_time_by_member_type_sensitivity"
                ]
            ),
            ("log", "compound_symmetry", True),
        )
        self.assertEqual(
            tuple(model_contracts.loc["percentage_loss_ols"]),
            ("percentage_loss", "independent", False),
        )
        self.assertEqual(
            set(result.base_model_cv["model"]),
            {"log_compound_symmetry_gls", "percentage_loss_ols"},
        )
        self.assertEqual(
            list(result.base_model_cv.columns),
            ["repeat", "fold", "model", "n_test", "rmse", "mae"],
        )
        paired_cv = result.base_model_cv.groupby(["repeat", "fold"])
        self.assertTrue(paired_cv.size().eq(2).all())
        self.assertTrue(paired_cv["model"].nunique().eq(2).all())
        self.assertTrue(paired_cv["n_test"].nunique().eq(1).all())
        self.assertTrue(
            np.isfinite(result.base_model_cv[["rmse", "mae"]]).all().all()
        )
        self.assertTrue((result.base_model_cv[["rmse", "mae"]] >= 0.0).all().all())
        for selection in (
            result.lasso_mean_selection,
            result.lasso_domain_selection,
        ):
            required_provenance = {
                "candidate_order",
                "full_sample_lambda",
                "full_sample_lambda_max",
                "cv_selection_rule",
                "fold_plan_id",
                "n_resamples",
                "subsample_fraction",
                "selection_count",
            }
            self.assertTrue(required_provenance.issubset(selection.columns))
            self.assertTrue(selection["cv_selection_rule"].eq("one_standard_error").all())
            self.assertTrue(selection["n_resamples"].eq(4).all())
            self.assertTrue(selection["subsample_fraction"].eq(0.7).all())
            self.assertTrue(
                np.allclose(
                    selection["full_sample_lambda"],
                    selection["lambda_ratio"]
                    * selection["full_sample_lambda_max"],
                )
            )
            self.assertTrue(
                np.allclose(
                    selection["selection_count"]
                    / selection["n_resamples"],
                    selection["selection_frequency"],
                )
            )
            self.assertTrue(
                selection["interpretation"]
                .eq("exploratory_stability_not_significance")
                .all()
            )
            self.assertFalse(
                any("p_value" in column.lower() for column in selection.columns)
            )
        self.assertEqual(
            result.lasso_mean_selection["fold_plan_id"].unique().tolist(),
            result.lasso_domain_selection["fold_plan_id"].unique().tolist(),
        )
        self.assertEqual(len(cv_spy.call_args_list), 2)
        self.assertEqual(
            [call.kwargs["random_state"] for call in cv_spy.call_args_list],
            [1071, 1071],
        )
        np.testing.assert_array_equal(
            cv_spy.call_args_list[0].args[0].group_ids,
            cv_spy.call_args_list[1].args[0].group_ids,
        )
        np.testing.assert_array_equal(
            cv_spy.call_args_list[0].args[0].strata,
            cv_spy.call_args_list[1].args[0].strata,
        )
        self.assertEqual(len(stability_spy.call_args_list), 2)
        self.assertEqual(
            [
                call.kwargs["random_state"]
                for call in stability_spy.call_args_list
            ],
            [11071, 12071],
        )
        self.assertEqual(
            tuple(result.lasso_mean_selection["candidate"]),
            (
                "engagement_volume_repeatable",
                "engagement_volume_repeatable_rate",
                "engagement_breadth",
                "tenure_days",
                "module_mean",
                "sex[MALE]",
            ),
        )
        self.assertEqual(
            tuple(result.lasso_mean_selection["candidate_order"]),
            (1, 2, 3, 4, 5, 10),
        )
        self.assertEqual(
            tuple(result.lasso_domain_selection["candidate"]),
            (
                "engagement_volume_repeatable",
                "engagement_volume_repeatable_rate",
                "engagement_breadth",
                "tenure_days",
                "module_core",
                "module_mindset",
                "module_nutrition",
                "module_physical_activity",
                "sex[MALE]",
            ),
        )
        self.assertEqual(
            tuple(result.lasso_domain_selection["candidate_order"]),
            (1, 2, 3, 4, 6, 7, 8, 9, 10),
        )
        mean_candidates = set(result.lasso_mean_selection["candidate"])
        domain_candidates = set(result.lasso_domain_selection["candidate"])
        module_domains = {
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
        }
        self.assertIn("module_mean", mean_candidates)
        self.assertNotIn("module_mean", domain_candidates)
        self.assertTrue(module_domains.isdisjoint(mean_candidates))
        self.assertTrue(module_domains.issubset(domain_candidates))
        self.assertEqual(
            set(result.diagnostics["diagnostic_type"]),
            {
                "missingness",
                "subgroup",
                "collinearity",
                "longitudinal_lasso_exploratory",
            },
        )
        longitudinal_details = result.diagnostics.loc[
            result.diagnostics["diagnostic_type"].eq(
                "longitudinal_lasso_exploratory"
            ),
            "detail",
        ]
        self.assertTrue(
            longitudinal_details.eq(
                "retrospective_two_stage_fixed_prewhitening_not_performance"
            ).all()
        )
        null_relations = result.diagnostics.loc[
            result.diagnostics["detail"].str.startswith(
                "null_relation_",
                na=False,
            )
        ]
        for _, relation in null_relations.groupby("detail", sort=True):
            self.assertGreater(float(relation.iloc[0]["value"]), 0.0)
        for table in (
            result.cohort_summary,
            result.feature_summary,
            result.model_summary,
            result.base_model_cv,
            result.lasso_mean_selection,
            result.lasso_domain_selection,
            result.diagnostics,
        ):
            self.assertNotIn("member_id", table.columns)

    def test_outputs_are_deterministic_aggregate_only_and_sources_are_read_only(
        self,
    ) -> None:
        """Write byte-identical summaries without changing input extracts.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Writes aggregate output files in two isolated directories.

        Statistical intent:
            Makes analysis artifacts reproducible while preventing row-level
            member disclosure or accidental mutation of supplied source data.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            first_output = root / "first_output"
            second_output = root / "second_output"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            source_before = {
                path.name: path.read_bytes() for path in sorted(data_dir.iterdir())
            }
            config = AnalysisConfig(
                seed=101,
                cv_folds=2,
                cv_repeats=1,
                stability_resamples=3,
                lambda_ratios=(1.0, 0.3, 0.05),
            )

            first_result = run_analysis(data_dir, config=config)
            second_result = run_analysis(data_dir, config=config)
            first_paths = write_analysis_outputs(first_result, first_output)
            second_paths = write_analysis_outputs(second_result, second_output)

            self.assertEqual(
                {path.relative_to(first_output) for path in first_paths},
                EXPECTED_OUTPUT_PATHS,
            )
            self.assertEqual(
                {path.relative_to(second_output) for path in second_paths},
                EXPECTED_OUTPUT_PATHS,
            )
            source_after = {
                path.name: path.read_bytes() for path in sorted(data_dir.iterdir())
            }
            self.assertEqual(source_before, source_after)
            for relative_path in EXPECTED_OUTPUT_PATHS:
                first_bytes = (first_output / relative_path).read_bytes()
                second_bytes = (second_output / relative_path).read_bytes()
                self.assertEqual(first_bytes, second_bytes)
                self.assertNotIn(b"fixture_member_", first_bytes)

            metadata = json.loads(
                (first_output / "nineam_analysis_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("generated_at", metadata)
            self.assertEqual(metadata["seed"], 101)
            self.assertEqual(
                metadata["selection_interpretation"],
                "exploratory_stability_not_significance",
            )

    def test_rare_subgroup_labels_are_retained_without_suppression(
        self,
    ) -> None:
        """Report approved descriptive levels without small-cell suppression.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Writes isolated source fixtures and aggregate output artifacts.

        Statistical intent:
            Preserves the approved no-suppression analytical reporting contract
            while keeping member identifiers outside every artifact.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir, include_rare_subgroups=True)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=17,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )
            write_analysis_outputs(result, output_dir)

            characteristics = result.reporting_tables[
                "nineam_sample_characteristics.csv"
            ]
            levels = set(characteristics["level"].dropna().astype(str))
            self.assertIn("SINGLETON_SEX_PRIVATE", levels)
            self.assertIn("SINGLETON_ETHNICITY_PRIVATE", levels)
            payload = (
                output_dir / "tables" / "nineam_sample_characteristics.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("SINGLETON_SEX_PRIVATE", payload)
            self.assertIn("SINGLETON_ETHNICITY_PRIVATE", payload)
            self.assertNotIn("<SUPPRESSED_RARE_LEVELS>", payload)
            self.assertNotIn("rare_level_reporting_label", result.metadata)

    def test_writer_enforces_closed_schemas_member_ids_and_metadata_contract(
        self,
    ) -> None:
        """Reject identifier columns/cells and malformed run metadata.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Builds isolated fixtures; rejected writes create no final artifacts.

        Statistical intent:
            Treats every aggregate file and metadata field as a closed disclosure
            contract instead of relying on an incomplete identifier denylist.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=19,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )

            def replace_reporting_table(
                filename: str,
                table: pd.DataFrame,
            ) -> AnalysisResult:
                """Return a result with one intentionally corrupted CSV table."""
                reporting_tables = {
                    name: result.reporting_tables[name].copy(deep=True)
                    for name in REPORTING_SCHEMAS
                }
                reporting_tables[filename] = table
                return replace(result, reporting_tables=reporting_tables)

            for column_name in (
                "user_id",
                "readable_id",
                "User_ID",
                "READABLE_ID",
            ):
                with self.subTest(column_name=column_name):
                    corrupted = result.reporting_tables[
                        "nineam_cohort_flow.csv"
                    ].copy()
                    corrupted[column_name] = "not_an_identifier"
                    with self.assertRaisesRegex(ValueError, "exact ordered schema"):
                        write_analysis_outputs(
                            replace_reporting_table(
                                "nineam_cohort_flow.csv",
                                corrupted,
                            ),
                            root / f"schema_{column_name}",
                        )

            identifier_cell = result.reporting_tables[
                "nineam_cohort_flow.csv"
            ].copy()
            identifier_cell.loc[0, "notes"] = " fixture_member_000 "
            with self.assertRaisesRegex(ValueError, "source member identifier"):
                write_analysis_outputs(
                    replace_reporting_table(
                        "nineam_cohort_flow.csv",
                        identifier_cell,
                    ),
                    root / "identifier_cell",
                )

            for embedded_value in (
                "prefix_FIXTURE_MEMBER_000",
                "fixture_member_000_suffix",
            ):
                with self.subTest(embedded_value=embedded_value):
                    embedded_cell = result.reporting_tables[
                        "nineam_cohort_flow.csv"
                    ].copy()
                    embedded_cell.loc[0, "notes"] = embedded_value
                    with self.assertRaisesRegex(
                        ValueError,
                        "source member identifier",
                    ):
                        write_analysis_outputs(
                            replace_reporting_table(
                                "nineam_cohort_flow.csv",
                                embedded_cell,
                            ),
                            root / f"embedded_{embedded_value}",
                        )

            structured_value = result.reporting_tables[
                "nineam_cohort_flow.csv"
            ].copy()
            structured_value["notes"] = structured_value["notes"].astype(object)
            structured_value.at[0, "notes"] = {
                "safe_key": ["prefix_fixture_member_000_suffix"]
            }
            with self.assertRaisesRegex(ValueError, "source member identifier"):
                write_analysis_outputs(
                    replace_reporting_table(
                        "nineam_cohort_flow.csv",
                        structured_value,
                    ),
                    root / "structured_identifier_value",
                )

            json_value = result.reporting_tables[
                "nineam_cohort_flow.csv"
            ].copy()
            json_value.loc[0, "notes"] = json.dumps(
                {"coef_fixture_member_000_suffix": 0.0}
            )
            with self.assertRaisesRegex(ValueError, "source member identifier"):
                write_analysis_outputs(
                    replace_reporting_table(
                        "nineam_cohort_flow.csv",
                        json_value,
                    ),
                    root / "json_identifier_key",
                )

            escaped_json_payloads = (
                '{"coef_fixture_member_\\u0030\\u0030\\u0030_suffix":0.0}',
                (
                    '{"safe":{"nested":"prefix_fixture_member_'
                    '\\u0030\\u0030\\u0030_suffix"}}'
                ),
                '"prefix_fixture_member_\\u0030\\u0030\\u0030_suffix"',
            )
            for case_number, escaped_payload in enumerate(
                escaped_json_payloads
            ):
                with self.subTest(escaped_json_case=case_number):
                    escaped_value = result.reporting_tables[
                        "nineam_cohort_flow.csv"
                    ].copy()
                    escaped_value.loc[0, "notes"] = (
                        escaped_payload
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "source member identifier",
                    ):
                        write_analysis_outputs(
                            replace_reporting_table(
                                "nineam_cohort_flow.csv",
                                escaped_value,
                            ),
                            root / f"escaped_json_{case_number}",
                        )

            metadata_cases: list[dict[str, object]] = []
            extra = dict(result.metadata)
            extra["extra_key"] = "unexpected"
            metadata_cases.append(extra)
            missing = dict(result.metadata)
            missing.pop("seed")
            metadata_cases.append(missing)
            nonfinite = dict(result.metadata)
            nonfinite["stability_subsample_fraction"] = float("nan")
            metadata_cases.append(nonfinite)
            identifier = dict(result.metadata)
            identifier["analysis_schema_version"] = "fixture_member_000"
            metadata_cases.append(identifier)
            wrong_value = dict(result.metadata)
            wrong_value["cross_family_likelihood_comparison"] = True
            metadata_cases.append(wrong_value)
            for case_number, metadata in enumerate(metadata_cases):
                with self.subTest(case_number=case_number):
                    with self.assertRaises(ValueError):
                        write_analysis_outputs(
                            replace(result, metadata=metadata),
                            root / f"metadata_{case_number}",
                        )

    def test_metadata_reconciles_fixed_and_source_derived_values(self) -> None:
        """Reject fixed-contract and source-reconciliation metadata probes.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Builds an isolated fixture; rejected writes create no artifacts.

        Statistical intent:
            Prevents internally plausible but false run metadata from diverging
            from fixed privacy settings, the cohort summary, or source tables.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=23,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )

            # These probes pass broad range/type checks but
            # contradict the analysis contract or reloaded aggregate sources.
            metadata_probes: list[tuple[str, dict[str, object]]] = []
            complete_pair = dict(result.metadata)
            complete_pair["complete_pair_eligible_members"] = 1
            metadata_probes.append(("complete_pair_1", complete_pair))
            longitudinal_ratio = dict(result.metadata)
            longitudinal_ratio["longitudinal_exploratory_lambda_ratio"] = 0.9
            metadata_probes.append(("longitudinal_ratio_0_9", longitudinal_ratio))
            included = dict(result.metadata)
            included["included_members"] = 1
            metadata_probes.append(("included_1", included))
            source_counts = dict(result.metadata)
            source_counts["source_row_counts"] = {
                "body_weights": 0,
                "demographics": 0,
                "engagement": 0,
                "module_completions": 0,
            }
            metadata_probes.append(("zero_source_counts", source_counts))

            for label, metadata in metadata_probes:
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        write_analysis_outputs(
                            replace(result, metadata=metadata),
                            root / label,
                        )

    def test_transaction_rolls_back_serialization_and_late_commit_failures(
        self,
    ) -> None:
        """Leave prior outputs unchanged after staged or atomic commit failure.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Writes valid outputs, then injects controlled serialization and OS
            replacement failures against the isolated output directory.

        Statistical intent:
            Ensures a partial analysis publication cannot mix artifacts from
            different runs or leave ambiguous temporary payloads.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=29,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )
            write_analysis_outputs(result, output_dir)
            original_bytes = {
                path.relative_to(output_dir): path.read_bytes()
                for path in output_dir.rglob("*")
                if path.is_file()
            }

            changed_flow = result.reporting_tables[
                "nineam_cohort_flow.csv"
            ].copy()
            changed_flow.loc[0, "notes"] = "changed for rollback probe"
            unserializable = result.reporting_tables[
                "nineam_limitations.csv"
            ].copy()
            unserializable["limitation"] = unserializable[
                "limitation"
            ].astype(object)
            unserializable.at[unserializable.index[-1], "limitation"] = (
                _UnserializableCell()
            )
            invalid_tables = {
                name: result.reporting_tables[name].copy(deep=True)
                for name in REPORTING_SCHEMAS
            }
            invalid_tables["nineam_cohort_flow.csv"] = changed_flow
            invalid_tables["nineam_limitations.csv"] = unserializable
            with self.assertRaisesRegex(
                RuntimeError,
                "controlled late serialization failure",
            ):
                write_analysis_outputs(
                    replace(
                        result,
                        reporting_tables=invalid_tables,
                    ),
                    output_dir,
                )
            self.assertEqual(
                original_bytes,
                {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                },
            )

            real_replace = os.replace
            replace_calls = 0

            def fail_once_during_late_commit(
                source: str | bytes | Path,
                destination: str | bytes | Path,
            ) -> None:
                """Fail one late replace while delegating every other real move.

                Args:
                    source: Existing staged or backup path.
                    destination: Target final or backup path.

                Returns:
                    None.

                Side effects:
                    Performs real atomic replacements except on call twenty.

                Statistical intent:
                    Exercises rollback after several final files were committed,
                    while allowing subsequent restoration operations to proceed.
                """
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 20:
                    raise OSError("controlled late atomic replace failure")
                real_replace(source, destination)

            changed_tables = {
                name: result.reporting_tables[name].copy(deep=True)
                for name in REPORTING_SCHEMAS
            }
            changed_tables["nineam_cohort_flow.csv"] = changed_flow

            with patch(
                "nineam_health_analysis.nineam_analysis_pipeline.os.replace",
                side_effect=fail_once_during_late_commit,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "controlled late atomic replace failure",
                ):
                    write_analysis_outputs(
                        replace(result, reporting_tables=changed_tables),
                        output_dir,
                    )
            self.assertEqual(
                original_bytes,
                {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                },
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in output_dir.rglob("*"))
            )

            figure_installs = 0

            def fail_once_during_figure_commit(
                source: str | bytes | Path,
                destination: str | bytes | Path,
            ) -> None:
                """Fail after three staged images while allowing rollback."""
                nonlocal figure_installs
                source_path = Path(source)
                if source_path.parent.name.startswith(".nineam_figures."):
                    figure_installs += 1
                    if figure_installs == 4:
                        raise OSError("controlled figure replace failure")
                real_replace(source, destination)

            with patch(
                "nineam_health_analysis.nineam_analysis_pipeline.os.replace",
                side_effect=fail_once_during_figure_commit,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "controlled figure replace failure",
                ):
                    write_analysis_outputs(result, output_dir)
            self.assertEqual(
                original_bytes,
                {
                    path.relative_to(output_dir): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                },
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in output_dir.rglob("*"))
            )

    def test_transaction_retains_backup_if_restoration_fails(self) -> None:
        """Preserve the prior artifact when its rollback replacement fails.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Writes valid outputs, then injects one commit failure and one
            persistent restoration failure in an isolated output directory.

        Statistical intent:
            Ensures a failed rollback never deletes the only recoverable copy
            of a previously published aggregate analysis artifact.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=31,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )
            write_analysis_outputs(result, output_dir)

            prior_final = output_dir / "tables" / "nineam_cohort_flow.csv"
            prior_bytes = prior_final.read_bytes()
            changed_flow = result.reporting_tables[
                "nineam_cohort_flow.csv"
            ].copy()
            changed_flow.loc[0, "notes"] = "changed for recovery probe"
            changed_tables = {
                name: result.reporting_tables[name].copy(deep=True)
                for name in REPORTING_SCHEMAS
            }
            changed_tables["nineam_cohort_flow.csv"] = changed_flow
            real_replace = os.replace
            staged_install_count = 0
            failed_backup_path: Path | None = None

            def fail_commit_and_one_restoration(
                source: str | bytes | Path,
                destination: str | bytes | Path,
            ) -> None:
                """Inject a late commit failure and a persistent restore failure.

                Args:
                    source: Existing staged or backup path.
                    destination: Intended final or backup destination.

                Returns:
                    None.

                Side effects:
                    Performs all other real replacements while recording the
                    backup whose restoration is intentionally blocked.

                Statistical intent:
                    Reaches the recovery branch where preserving the prior
                    aggregate matters more than removing every temporary file.
                """
                nonlocal staged_install_count, failed_backup_path
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path.suffix == ".stage":
                    staged_install_count += 1
                    if staged_install_count == 4:
                        raise OSError("controlled late commit failure")
                if (
                    source_path.suffix == ".backup"
                    and destination_path == prior_final
                ):
                    failed_backup_path = source_path
                    raise OSError("controlled persistent restoration failure")
                real_replace(source, destination)

            with patch(
                "nineam_health_analysis.nineam_analysis_pipeline.os.replace",
                side_effect=fail_commit_and_one_restoration,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "recoverable backup",
                ) as caught:
                    write_analysis_outputs(
                        replace(result, reporting_tables=changed_tables),
                        output_dir,
                    )

            self.assertIsNotNone(failed_backup_path)
            assert failed_backup_path is not None
            self.assertTrue(failed_backup_path.exists())
            self.assertEqual(prior_bytes, failed_backup_path.read_bytes())
            self.assertIn(str(failed_backup_path), str(caught.exception))
            self.assertFalse(prior_final.exists())
            self.assertFalse(
                any(path.suffix == ".stage" for path in output_dir.rglob("*"))
            )

    def test_writer_rejects_source_directory_as_output_target(self) -> None:
        """Refuse to place generated summaries beside source extracts.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Creates and removes isolated source fixture files; writes no output.

        Statistical intent:
            Enforces a read-only analytical boundary around supplied data files.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            result = run_analysis(
                data_dir,
                config=AnalysisConfig(
                    seed=4,
                    cv_folds=2,
                    cv_repeats=1,
                    stability_resamples=2,
                    lambda_ratios=(1.0, 0.2),
                ),
            )
            for destination in (data_dir, data_dir / "nested" / "output"):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(
                        ValueError,
                        "source data directory",
                    ):
                        write_analysis_outputs(result, destination)

    def test_cli_uses_required_defaults_and_runs_with_explicit_directories(
        self,
    ) -> None:
        """Expose documented defaults and execute the real CLI main function.

        Args:
            self: Test case providing temporary directories and assertions.

        Returns:
            None.

        Side effects:
            Loads the CLI module and writes aggregate outputs to a temp folder.

        Statistical intent:
            Ensures command-line controls map directly to deterministic pipeline
            resampling parameters rather than hidden global state.
        """
        cli = _load_cli_module()
        parser = cli.build_argument_parser()
        defaults = parser.parse_args([])
        project_root = Path(__file__).parents[1]
        self.assertEqual(defaults.data_dir, project_root.parent / "data")
        self.assertEqual(defaults.output_dir, project_root / "outputs")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            _write_pipeline_fixture(data_dir)
            exit_code = cli.main(
                [
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "88",
                    "--cv-folds",
                    "2",
                    "--cv-repeats",
                    "1",
                    "--stability-resamples",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {
                    path.relative_to(output_dir)
                    for path in output_dir.rglob("*")
                    if path.is_file()
                },
                EXPECTED_OUTPUT_PATHS,
            )

    def test_supplied_source_smoke_reproduces_cohort_and_common_target_ordering(
        self,
    ) -> None:
        """Run a small configured smoke check against supplied source extracts.

        Args:
            self: Test case providing repository-path discovery and assertions.

        Returns:
            None.

        Side effects:
            Reads supplied source files; writes no outputs.

        Statistical intent:
            Protects the 534-member primary cohort and confirms the percentage base model
            has lower average held-out raw-weight RMSE in this fixed smoke split.
        """
        source_data = Path(__file__).parents[2] / "data"
        if not source_data.is_dir():
            self.skipTest("Supplied case-study data are not available")
        result = run_analysis(
            source_data,
            config=AnalysisConfig(
                seed=20260901,
                cv_folds=2,
                cv_repeats=1,
                stability_resamples=2,
                lambda_ratios=(1.0, 0.2),
            ),
        )

        included = result.cohort_summary.loc[
            result.cohort_summary["metric"].eq("included_members"),
            "count",
        ].iloc[0]
        mean_rmse = result.base_model_cv.groupby("model")["rmse"].mean()
        cohort_counts = result.cohort_summary.set_index("metric")["count"]
        self.assertEqual(included, 534)
        self.assertEqual(cohort_counts["pre_member_type_restriction_members"], 633)
        self.assertEqual(cohort_counts["excluded_nonprimary_member_type"], 99)
        self.assertEqual(result.metadata["complete_pair_eligible_members"], 633)
        self.assertEqual(result.metadata["excluded_nonprimary_members"], 99)
        self.assertEqual(result.metadata["included_members"], 534)
        self.assertEqual(result.metadata["reference_member_type"], "Coaching Only")
        self.assertEqual(result.metadata["responder_threshold_percentage"], 5.0)
        self.assertEqual(result.metadata["locked_covariance_estimator"], "HC3")
        self.assertEqual(
            result.metadata["locked_inference_status"],
            "conditional_exploratory",
        )
        self.assertLess(
            mean_rmse["percentage_loss_ols"],
            mean_rmse["log_compound_symmetry_gls"],
        )


if __name__ == "__main__":
    unittest.main()
