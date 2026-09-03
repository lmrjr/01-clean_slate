"""Test member-level resampling, penalized designs, and model comparison."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nineam_health_analysis.nineam_resampling import (
    GroupedFold,
    LambdaCVResult,
    PenalizedDesign,
    StabilitySelectionResult,
    build_longitudinal_penalized_design,
    build_percentage_penalized_design,
    compare_base_models,
    make_grouped_repeated_folds,
    select_lambda_by_grouped_cv,
    stability_select_lasso,
)
from nineam_health_analysis.nineam_statistical_models import (
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
    predict_last_weight_longitudinal,
    predict_last_weight_percentage_loss,
)


def _make_member_features(n_members: int = 16) -> pd.DataFrame:
    """Create deterministic member features for design and resampling tests.

    Args:
        n_members: Number of unique member rows to create.

    Returns:
        A dataframe with endpoint weights, factors, and every candidate feature.

    Side effects:
        None; a new dataframe is returned on each call.

    Statistical intent:
        Supplies non-collinear factor and continuous patterns with a percentage-
        loss signal that can be checked independently of production builders.
    """
    member_number = np.arange(n_members)
    member_type = np.where(member_number % 2 == 0, "A", "B")
    first_weight = 160.0 + 3.5 * member_number + 12.0 * (member_type == "B")
    percentage_loss = (
        6.0
        + 0.035 * (first_weight - 180.0)
        + 1.8 * (member_type == "B")
        + np.resize(np.array([-0.8, 0.3, 0.9, -0.4]), n_members)
    )
    last_weight = first_weight * (1.0 - percentage_loss / 100.0)
    module_core = (member_number % 9) / 9.0
    module_mindset = (member_number % 4) / 4.0
    module_nutrition = ((member_number + 1) % 4) / 4.0
    module_physical = ((member_number + 2) % 4) / 4.0

    return pd.DataFrame(
        {
            "member_id": [f"member_{value:02d}" for value in member_number],
            "member_type": member_type,
            "first_weight": first_weight,
            "last_weight": last_weight,
            "engagement_volume_repeatable": 2.0 + member_number,
            "engagement_volume_repeatable_rate": (
                0.1 + (member_number % 5) / 10.0
            ),
            "engagement_breadth": 1.0 + (member_number % 7),
            "tenure_days": 30.0 + 4.0 * member_number,
            "module_mean": np.mean(
                np.column_stack(
                    [
                        module_core,
                        module_mindset,
                        module_nutrition,
                        module_physical,
                    ]
                ),
                axis=1,
            ),
            "module_core": module_core,
            "module_mindset": module_mindset,
            "module_nutrition": module_nutrition,
            "module_physical_activity": module_physical,
            # This pattern deliberately differs from member type so the two
            # factors represent separate candidate and base information.
            "sex": np.where(member_number % 4 < 2, "F", "M"),
        }
    )


def _make_signal_design(n_members: int = 40) -> PenalizedDesign:
    """Create a grouped LASSO problem with one reproducible true signal.

    Args:
        n_members: Number of independent one-row member groups.

    Returns:
        An immutable penalized design with signal and noise candidates.

    Side effects:
        None; random values come from a local fixed-seed generator.

    Statistical intent:
        Makes the first candidate explain outcome beyond an intercept while the
        second candidate is independent noise, enabling selection checks.
    """
    generator = np.random.default_rng(918)
    signal = generator.normal(size=n_members)
    noise = generator.normal(size=n_members)
    residual = generator.normal(scale=0.12, size=n_members)
    return PenalizedDesign(
        outcome=2.0 + 3.5 * signal + residual,
        base_design=np.ones((n_members, 1)),
        candidate_design=np.column_stack([signal, noise]),
        base_names=("intercept",),
        candidate_names=("signal", "noise"),
        group_ids=np.array([f"g{index:02d}" for index in range(n_members)]),
        strata=np.where(np.arange(n_members) % 2 == 0, "A", "B"),
    )


class GroupedFoldTestCase(unittest.TestCase):
    """Exercise deterministic member-level repeated fold construction."""

    def test_groups_remain_whole_and_test_once_per_repeat(self) -> None:
        """Keep paired rows together and test each member exactly once.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents within-member leakage while retaining exhaustive repeated
            cross-validation coverage.
        """
        group_ids = np.repeat([f"g{index}" for index in range(9)], 2)
        strata = np.repeat(["A"] * 5 + ["B"] * 4, 2)
        folds = make_grouped_repeated_folds(
            group_ids,
            strata=strata,
            n_splits=3,
            n_repeats=2,
            random_state=42,
        )

        self.assertEqual(len(folds), 6)
        for repeat in range(2):
            tested_groups: list[str] = []
            for fold in [item for item in folds if item.repeat == repeat]:
                train_groups = set(group_ids[fold.train_indices])
                test_groups = set(group_ids[fold.test_indices])
                self.assertFalse(train_groups.intersection(test_groups))
                self.assertGreater(fold.train_indices.size, 0)
                self.assertGreater(fold.test_indices.size, 0)
                tested_groups.extend(test_groups)
            self.assertCountEqual(tested_groups, np.unique(group_ids))

    def test_folds_are_deterministic_and_arrays_are_immutable(self) -> None:
        """Return identical read-only indices for an identical seed.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Makes repeated-validation membership exactly reproducible.
        """
        group_ids = np.repeat(["a", "b", "c", "d"], 2)
        first = make_grouped_repeated_folds(
            group_ids,
            n_splits=2,
            n_repeats=2,
            random_state=7,
        )
        second = make_grouped_repeated_folds(
            group_ids,
            n_splits=2,
            n_repeats=2,
            random_state=7,
        )

        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left.train_indices, right.train_indices)
            np.testing.assert_array_equal(left.test_indices, right.test_indices)
        with self.assertRaises(ValueError):
            first[0].test_indices[0] = 99

    def test_strata_must_be_constant_within_member(self) -> None:
        """Reject a member assigned to more than one stratum.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Avoids ambiguous stratification that could silently split a group.
        """
        with self.assertRaisesRegex(ValueError, "vary within group"):
            make_grouped_repeated_folds(
                ["a", "a", "b", "b"],
                strata=["A", "B", "A", "A"],
                n_splits=2,
                n_repeats=1,
                random_state=1,
            )

    def test_grouped_fold_rejects_fractional_or_duplicate_indices(self) -> None:
        """Reject unsafe integer casts and repeated row membership.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents silent truncation or duplicate weighting from changing a
            grouped validation sample.
        """
        invalid_folds = (
            {
                "repeat": 0.5,
                "fold": 0,
                "train_indices": [0, 1],
                "test_indices": [2],
            },
            {
                "repeat": 0,
                "fold": 1.5,
                "train_indices": [0, 1],
                "test_indices": [2],
            },
            {
                "repeat": 0,
                "fold": 0,
                "train_indices": [0.2, 1.0],
                "test_indices": [2],
            },
            {
                "repeat": 0,
                "fold": 0,
                "train_indices": [0, 0, 1],
                "test_indices": [2],
            },
            {
                "repeat": 0,
                "fold": 0,
                "train_indices": [0, 1],
                "test_indices": [2, 2],
            },
        )
        for arguments in invalid_folds:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    GroupedFold(**arguments)


class PenalizedDesignTestCase(unittest.TestCase):
    """Exercise percentage and longitudinal penalized design construction."""

    def test_confirmed_member_types_default_to_coaching_reference(self) -> None:
        """Both penalized paths preserve the confirmed treatment contrast."""
        members = _make_member_features(8).replace(
            {"A": "Coaching Only", "B": "Active GLP-1 for Weight-loss"}
        )
        percentage = build_percentage_penalized_design(members)
        fitted = fit_longitudinal_gls(members, covariance_structure="unstructured")
        longitudinal = build_longitudinal_penalized_design(members, fitted)
        self.assertIn("member_type[Active GLP-1 for Weight-loss]", percentage.base_names)
        self.assertNotIn("member_type[Coaching Only]", percentage.base_names)
        self.assertIn("member_type[Active GLP-1 for Weight-loss]", longitudinal.base_names)

    def test_percentage_design_uses_literal_outcome_and_columns(self) -> None:
        """Build percentage loss, base terms, and requested mean candidates.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Protects the signed outcome formula and unpenalized baseline/type
            adjustment while keeping exploratory features penalized.
        """
        members = _make_member_features(4)
        design = build_percentage_penalized_design(members, module_spec="mean")

        expected_outcome = 100.0 * (
            members["first_weight"].to_numpy()
            - members["last_weight"].to_numpy()
        ) / members["first_weight"].to_numpy()
        np.testing.assert_allclose(design.outcome, expected_outcome)
        self.assertEqual(
            design.base_names,
            ("intercept", "first_weight", "member_type[B]"),
        )
        np.testing.assert_allclose(
            design.base_design[:, :2],
            np.column_stack([np.ones(4), members["first_weight"]]),
        )
        np.testing.assert_array_equal(design.base_design[:, 2], [0, 1, 0, 1])
        self.assertEqual(
            design.candidate_names,
            (
                "engagement_volume_repeatable",
                "engagement_volume_repeatable_rate",
                "engagement_breadth",
                "tenure_days",
                "module_mean",
                "sex[M]",
            ),
        )
        np.testing.assert_allclose(
            design.candidate_design[0],
            [2.0, 0.1, 1.0, 30.0, members.loc[0, "module_mean"], 0.0],
        )
        np.testing.assert_array_equal(design.group_ids, members["member_id"])
        np.testing.assert_array_equal(design.strata, members["member_type"])
        with self.assertRaises(ValueError):
            design.candidate_design[0, 0] = -1.0

    def test_module_mean_and_domains_are_separate_specifications(self) -> None:
        """Never place module mean beside its four defining domains.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Avoids the exact module-mean dependency that makes LASSO allocation
            among those five columns non-unique.
        """
        members = _make_member_features(8)
        mean_design = build_percentage_penalized_design(
            members,
            module_spec="mean",
        )
        domain_design = build_percentage_penalized_design(
            members,
            module_spec="domains",
        )

        self.assertIn("module_mean", mean_design.candidate_names)
        self.assertNotIn("module_mean", domain_design.candidate_names)
        self.assertTrue(
            {
                "module_core",
                "module_mindset",
                "module_nutrition",
                "module_physical_activity",
            }.issubset(domain_design.candidate_names)
        )

    def test_longitudinal_design_matches_manual_cholesky_whitening(self) -> None:
        """Whiten each two-row member block and penalize time interactions.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Verifies retrospective two-stage marginal-GLS construction: feature
            main effects enforce hierarchy and time interactions represent change.
        """
        members = _make_member_features(4)
        fitted = fit_longitudinal_gls(
            members,
            outcome_scale="raw",
            covariance_structure="unstructured",
        )
        design = build_longitudinal_penalized_design(
            members,
            fitted,
            module_spec="mean",
        )

        candidate_values = np.array(
            [
                members.loc[0, "engagement_volume_repeatable"],
                members.loc[0, "engagement_volume_repeatable_rate"],
                members.loc[0, "engagement_breadth"],
                members.loc[0, "tenure_days"],
                members.loc[0, "module_mean"],
                0.0,
            ]
        )
        unwhitened_outcome = members.loc[0, ["first_weight", "last_weight"]].to_numpy(
            dtype=float
        )
        unwhitened_base = np.vstack(
            [
                np.concatenate([[1.0, 0.0, 0.0], candidate_values]),
                np.concatenate([[1.0, 0.0, 1.0], candidate_values]),
            ]
        )
        unwhitened_candidates = np.vstack(
            [np.zeros(candidate_values.size), candidate_values]
        )
        cholesky = np.linalg.cholesky(fitted.covariance_matrix)

        np.testing.assert_allclose(
            design.outcome[:2],
            np.linalg.solve(cholesky, unwhitened_outcome),
        )
        np.testing.assert_allclose(
            design.base_design[:2],
            np.linalg.solve(cholesky, unwhitened_base),
        )
        np.testing.assert_allclose(
            design.candidate_design[:2],
            np.linalg.solve(cholesky, unwhitened_candidates),
        )
        self.assertEqual(
            design.base_names[:3],
            ("intercept", "member_type[B]", "time"),
        )
        self.assertTrue(
            all(name.startswith("time:") for name in design.candidate_names)
        )
        np.testing.assert_array_equal(
            design.group_ids,
            np.repeat(members["member_id"].to_numpy(), 2),
        )

    def test_longitudinal_design_requires_matching_training_schema(self) -> None:
        """Reject fitted covariance metadata from a different member set size.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Blocks accidental reuse of a covariance estimate trained on a
            different analysis sample.
        """
        members = _make_member_features(4)
        other_members = members.copy()
        other_members["last_weight"] = other_members["last_weight"] - np.array(
            [0.5, -0.2, 0.3, -0.4]
        )
        fitted_elsewhere = fit_longitudinal_gls(
            other_members,
            outcome_scale="raw",
            covariance_structure="unstructured",
        )
        with self.assertRaisesRegex(ValueError, "same training members"):
            build_longitudinal_penalized_design(
                members,
                fitted_elsewhere,
                module_spec="mean",
            )

    def test_longitudinal_design_rejects_replaced_training_member_ids(self) -> None:
        """Require exact fitted-member provenance even when estimates match.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents a numerically identical covariance fit from being falsely
            attributed to a different set of training members.
        """
        fitted_members = _make_member_features(8)
        fitted = fit_longitudinal_gls(
            fitted_members,
            outcome_scale="raw",
            covariance_structure="unstructured",
        )
        replaced_members = fitted_members.copy()
        replaced_members["member_id"] = [
            f"replacement_{index:02d}" for index in range(len(replaced_members))
        ]

        with self.assertRaisesRegex(ValueError, "same training members"):
            build_longitudinal_penalized_design(
                replaced_members,
                fitted,
                module_spec="mean",
            )


class LambdaSelectionTestCase(unittest.TestCase):
    """Exercise grouped inner-CV tuning and the one-standard-error rule."""

    def test_grouped_cv_finds_signal_and_applies_one_se_rule(self) -> None:
        """Choose the strongest ratio inside the minimum-score one-SE band.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Balances held-out error and parsimony without treating tuning as
            inferential significance.
        """
        design = _make_signal_design()
        ratios = [1.0, 0.5, 0.15, 0.03]
        minimum = select_lambda_by_grouped_cv(
            design,
            ratios,
            n_splits=4,
            n_repeats=2,
            random_state=23,
            selection_rule="minimum",
        )
        one_se = select_lambda_by_grouped_cv(
            design,
            ratios,
            n_splits=4,
            n_repeats=2,
            random_state=23,
            selection_rule="one_standard_error",
        )

        self.assertIsInstance(one_se, LambdaCVResult)
        self.assertEqual(one_se.fold_scores.shape, (8, 4))
        np.testing.assert_allclose(one_se.fold_scores, minimum.fold_scores)
        self.assertLess(minimum.selected_lambda_ratio, 1.0)
        minimum_index = int(np.argmin(one_se.mean_scores))
        threshold = (
            one_se.mean_scores[minimum_index]
            + one_se.standard_errors[minimum_index]
        )
        eligible = one_se.lambda_ratios[one_se.mean_scores <= threshold]
        self.assertEqual(one_se.selected_lambda_ratio, float(np.max(eligible)))

    def test_fold_training_lambda_max_ignores_heldout_candidate_values(self) -> None:
        """Keep each fold's scaling and lambda-max confined to training rows.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Detects candidate-scaling leakage from an inner validation fold.
        """
        design = _make_signal_design(24)
        folds = make_grouped_repeated_folds(
            design.group_ids,
            strata=design.strata,
            n_splits=3,
            n_repeats=1,
            random_state=31,
        )
        original = select_lambda_by_grouped_cv(
            design,
            [0.5, 0.1],
            n_splits=3,
            n_repeats=1,
            random_state=31,
            selection_rule="minimum",
        )

        # Modify only fold zero's held-out candidate values. Its training fit,
        # including scaling and lambda-max, must remain numerically identical.
        changed_candidates = np.array(design.candidate_design, copy=True)
        changed_candidates[folds[0].test_indices, 0] += 500.0
        changed = PenalizedDesign(
            outcome=design.outcome,
            base_design=design.base_design,
            candidate_design=changed_candidates,
            base_names=design.base_names,
            candidate_names=design.candidate_names,
            group_ids=design.group_ids,
            strata=design.strata,
        )
        modified = select_lambda_by_grouped_cv(
            changed,
            [0.5, 0.1],
            n_splits=3,
            n_repeats=1,
            random_state=31,
            selection_rule="minimum",
        )

        self.assertEqual(
            original.fold_lambda_maxima[0],
            modified.fold_lambda_maxima[0],
        )

    def test_lambda_ratios_must_be_unique_and_inside_open_zero_interval(self) -> None:
        """Reject duplicated, zero, or oversized tuning ratios.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Keeps the tuning grid identifiable and bounded by lambda-max.
        """
        design = _make_signal_design(16)
        for invalid in ([0.5, 0.5], [0.0, 0.5], [0.5, 1.1]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    select_lambda_by_grouped_cv(
                        design,
                        invalid,
                        n_splits=2,
                        n_repeats=1,
                        random_state=4,
                    )

    def test_lambda_result_rejects_internally_inconsistent_summaries(self) -> None:
        """Reject manually constructed tuning results that contradict scores.

        Args:
            self: Test case providing constructor-validation assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents reporting a selected penalty alongside fabricated,
            nonfinite, negative, or miscomputed cross-validation evidence.
        """
        valid = {
            "lambda_ratios": np.array([1.0, 0.5]),
            "fold_scores": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "mean_scores": np.array([2.0, 3.0]),
            "standard_errors": np.array([1.0, 1.0]),
            "fold_lambda_maxima": np.array([5.0, 6.0]),
            "selected_lambda_ratio": 1.0,
            "selected_index": 0,
            "selection_rule": "minimum",
        }
        invalid_overrides = (
            {"lambda_ratios": np.array([np.nan, 0.5])},
            {"fold_scores": np.array([[1.0, -2.0], [3.0, 4.0]])},
            {"mean_scores": np.array([99.0, 3.0])},
            {"standard_errors": np.array([-1.0, 1.0])},
            {"fold_lambda_maxima": np.array([-1.0, 6.0])},
            {"selected_lambda_ratio": 0.5},
            {"selected_index": 1},
        )
        for override in invalid_overrides:
            with self.subTest(override=tuple(override)):
                with self.assertRaises((TypeError, ValueError)):
                    LambdaCVResult(**(valid | override))

        # A zero lambda-max is mathematically valid when candidates contain no
        # residual signal after adjustment for the unpenalized base design.
        zero_maximum = LambdaCVResult(
            **(valid | {"fold_lambda_maxima": np.array([0.0, 6.0])})
        )
        self.assertEqual(zero_maximum.fold_lambda_maxima[0], 0.0)


class StabilitySelectionTestCase(unittest.TestCase):
    """Exercise deterministic stratified member-level stability selection."""

    def test_stability_selection_records_frequencies_and_member_samples(self) -> None:
        """Select the recurring signal and expose bounded reproducible frequencies.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Describes resample consistency as exploratory stability rather than
            converting LASSO selection into significance testing.
        """
        design = _make_signal_design()
        first = stability_select_lasso(
            design,
            lambda_ratio=0.2,
            n_resamples=12,
            subsample_fraction=0.7,
            selection_threshold=0.75,
            random_state=12,
        )
        second = stability_select_lasso(
            design,
            lambda_ratio=0.2,
            n_resamples=12,
            subsample_fraction=0.7,
            selection_threshold=0.75,
            random_state=12,
        )

        self.assertEqual(first.coefficient_matrix.shape, (12, 2))
        self.assertEqual(first.selected_matrix.shape, (12, 2))
        self.assertTrue(np.all(first.selection_frequencies >= 0.0))
        self.assertTrue(np.all(first.selection_frequencies <= 1.0))
        self.assertGreaterEqual(first.selection_frequencies[0], 0.75)
        self.assertIn("signal", first.selected_candidate_names)
        np.testing.assert_allclose(
            first.selection_frequencies,
            first.selected_matrix.mean(axis=0),
        )
        np.testing.assert_array_equal(
            first.sampled_group_ids,
            second.sampled_group_ids,
        )
        np.testing.assert_allclose(
            first.coefficient_matrix,
            second.coefficient_matrix,
        )
        for sampled_groups in first.sampled_group_ids:
            self.assertEqual(len(sampled_groups), len(np.unique(sampled_groups)))
        with self.assertRaises(ValueError):
            first.selection_frequencies[0] = 0.0

    def test_stability_result_rejects_invalid_controls_and_frequencies(self) -> None:
        """Reject stability summaries inconsistent with their resample matrix.

        Args:
            self: Test case providing constructor-validation assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Keeps empirical frequencies tied to stored resamples and prevents
            invalid controls from changing the reported stable-variable set.
        """
        valid = {
            "candidate_names": ("a", "b"),
            "lambda_ratio": 0.5,
            "selection_threshold": 0.5,
            "coefficient_matrix": np.array([[1.0, 0.0], [2.0, 3.0]]),
            "selected_matrix": np.array([[True, False], [True, True]]),
            "selection_frequencies": np.array([1.0, 0.5]),
            "selected_candidate_names": ("a", "b"),
            "sampled_group_ids": np.array(
                [["member_1", "member_2"], ["member_1", "member_3"]]
            ),
        }
        invalid_overrides = (
            {"lambda_ratio": 0.0},
            {"lambda_ratio": np.nan},
            {"selection_threshold": 0.0},
            {"selection_threshold": np.inf},
            {"selection_frequencies": np.array([0.5, 0.5])},
        )
        for override in invalid_overrides:
            with self.subTest(override=tuple(override)):
                with self.assertRaises((TypeError, ValueError)):
                    StabilitySelectionResult(**(valid | override))


class BaseModelComparisonTestCase(unittest.TestCase):
    """Exercise paired held-out comparison on raw follow-up weight."""

    def test_base_models_share_test_members_and_report_only_prediction_metrics(
        self,
    ) -> None:
        """Use identical folds and verify RMSE/MAE from literal held-out rows.

        Args:
            self: Test case providing assertion methods.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Compares different outcome families only on their common raw last-
            weight target, never by cross-family likelihood criteria.
        """
        members = _make_member_features(24)
        comparison = compare_base_models(
            members,
            n_splits=3,
            n_repeats=2,
            random_state=55,
        )

        self.assertEqual(
            list(comparison.columns),
            ["repeat", "fold", "model", "n_test", "rmse", "mae"],
        )
        self.assertEqual(len(comparison), 12)
        self.assertNotIn("aic", {name.lower() for name in comparison.columns})
        paired_counts = comparison.pivot(
            index=["repeat", "fold"],
            columns="model",
            values="n_test",
        )
        self.assertTrue(paired_counts.nunique(axis="columns").eq(1).all())

        # Recreate fold zero independently to prove both reported model rows use
        # that exact test-member set and the documented prediction equations.
        folds = make_grouped_repeated_folds(
            members["member_id"].to_numpy(),
            strata=members["member_type"].to_numpy(),
            n_splits=3,
            n_repeats=2,
            random_state=55,
        )
        fold = folds[0]
        training = members.iloc[fold.train_indices]
        testing = members.iloc[fold.test_indices]
        observed = testing["last_weight"].to_numpy(dtype=float)
        longitudinal = fit_longitudinal_gls(
            training,
            outcome_scale="log",
            covariance_structure="compound_symmetry",
        )
        percentage = fit_percentage_loss_ols(training)
        predictions = {
            "log_compound_symmetry_gls": predict_last_weight_longitudinal(
                longitudinal,
                testing,
            ),
            "percentage_loss_ols": predict_last_weight_percentage_loss(
                percentage,
                testing,
            ),
        }
        for model, predicted in predictions.items():
            row = comparison.loc[
                comparison["repeat"].eq(0)
                & comparison["fold"].eq(0)
                & comparison["model"].eq(model)
            ].iloc[0]
            residual = observed - predicted
            self.assertAlmostEqual(row["rmse"], np.sqrt(np.mean(residual**2)))
            self.assertAlmostEqual(row["mae"], np.mean(np.abs(residual)))

        repeated = compare_base_models(
            members,
            n_splits=3,
            n_repeats=2,
            random_state=55,
        )
        pd.testing.assert_frame_equal(comparison, repeated)


if __name__ == "__main__":
    unittest.main()
