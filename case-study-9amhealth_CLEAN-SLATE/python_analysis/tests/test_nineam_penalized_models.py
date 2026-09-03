"""Test partially penalized LASSO fitting and collinearity diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from nineam_health_analysis.nineam_penalized_models import (
    diagnose_collinearity,
    fit_partially_penalized_lasso,
    predict_partially_penalized_lasso,
)


def _make_signal_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Create a deterministic regression problem with one true candidate.

    Args:
        None.

    Returns:
        Outcome, unpenalized design, candidate design, and their column names.

    Side effects:
        None.

    Statistical intent:
        Supplies an intercept and trend as base terms while only the first
        candidate contributes additional outcome variation.
    """
    # This repeating contrast is orthogonal to both candidates, allowing the
    # base coefficient to be checked independently when LASSO terms are zero.
    base_trend = np.tile(np.array([1.0, -1.0, -1.0, 1.0]), 6)
    true_signal = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 6)
    noise_candidate = np.repeat(np.array([-1.0, 0.0, 1.0]), 8)
    base_design = np.column_stack([np.ones(24), base_trend])
    candidate_design = np.column_stack([true_signal, noise_candidate])
    outcome = 5.0 + 1.5 * base_trend + 3.0 * true_signal
    return (
        outcome,
        base_design,
        candidate_design,
        ("intercept", "baseline_trend"),
        ("true_signal", "noise_candidate"),
    )


class PartiallyPenalizedLassoTestCase(unittest.TestCase):
    """Exercise coefficient fitting, prediction, and immutable result state."""

    def test_lambda_max_zeroes_candidates_but_preserves_base(self) -> None:
        """Protect lambda-max screening and the unpenalized base model.

        Args:
            self: Test case providing hand-constructed model inputs.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            At lambda-max every penalized KKT condition must hold at zero,
            while the intercept and baseline trend remain freely estimated.
        """
        outcome, base, candidates, base_names, candidate_names = (
            _make_signal_problem()
        )

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            base_names,
            candidate_names,
            lambda_ratio=1.0,
        )

        np.testing.assert_array_equal(
            result.standardized_candidate_coefficients,
            np.zeros(2),
        )
        np.testing.assert_array_equal(
            result.candidate_coefficients,
            np.zeros(2),
        )
        np.testing.assert_allclose(
            result.base_coefficients,
            np.array([5.0, 1.5]),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(result.lambda_value, result.lambda_max)
        self.assertEqual(result.lambda_ratio, 1.0)
        self.assertLessEqual(result.kkt_violation, 1e-10)

    def test_lower_lambda_selects_the_true_candidate(self) -> None:
        """Protect recovery of a strong signal below lambda-max.

        Args:
            self: Test case providing deterministic synthetic data.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Demonstrates that shrinkage releases the candidate explaining
            residual outcome variation while retaining the base terms.
        """
        outcome, base, candidates, base_names, candidate_names = (
            _make_signal_problem()
        )

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            base_names,
            candidate_names,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )

        self.assertGreater(abs(result.candidate_coefficients[0]), 2.0)
        self.assertAlmostEqual(result.candidate_coefficients[1], 0.0, places=12)
        self.assertLessEqual(result.kkt_violation, 1e-12)
        self.assertGreaterEqual(result.iterations, 1)

    def test_constants_and_base_aliases_are_explicitly_excluded(self) -> None:
        """Protect exclusion of non-estimable penalized columns.

        Args:
            self: Test case providing constant and aliased candidates.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Removes columns with no training variation or no variation beyond
            the unpenalized design before lambda and KKT calculations.
        """
        outcome, base, candidates, base_names, candidate_names = (
            _make_signal_problem()
        )
        expanded_candidates = np.column_stack(
            [candidates, np.full(len(outcome), 7.0), base[:, 1]]
        )
        expanded_names = candidate_names + ("constant", "base_alias")

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            expanded_candidates,
            base_names,
            expanded_names,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )

        self.assertEqual(result.excluded_candidate_names, ("constant", "base_alias"))
        np.testing.assert_array_equal(
            result.candidate_coefficients[2:],
            np.zeros(2),
        )
        self.assertEqual(result.candidate_means[2], 7.0)
        self.assertEqual(result.candidate_scales[2], 0.0)

    def test_penalty_weights_change_relative_selection_pressure(self) -> None:
        """Protect weighted lambda-max and coordinate thresholds.

        Args:
            self: Test case providing equally predictive candidates.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Confirms a larger penalty weight requires a larger correlation for
            a candidate to enter at the same lambda ratio.
        """
        first = np.tile(np.array([-1.0, 1.0]), 10)
        second = np.repeat(np.array([-1.0, 1.0]), 10)
        base = np.ones((20, 1))
        candidates = np.column_stack([first, second])
        outcome = 2.0 + 2.0 * first + 2.0 * second

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            ("intercept",),
            ("light_penalty", "heavy_penalty"),
            lambda_ratio=0.5,
            penalty_weights=np.array([1.0, 4.0]),
            tolerance=1e-12,
        )

        self.assertGreater(abs(result.candidate_coefficients[0]), 0.0)
        self.assertEqual(result.candidate_coefficients[1], 0.0)

    def test_weighted_lambda_objective_and_kkt_match_literal_oracles(
        self,
    ) -> None:
        """Protect weighted optimization with independently derived values.

        Args:
            self: Test case providing orthogonal unit-scale candidates.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Checks lambda-max, the penalized objective, and active-coordinate
            KKT equations against hand-calculated values rather than another
            implementation of the fitted algorithm.
        """
        first = np.array([-1.0, -1.0, 1.0, 1.0])
        second = np.array([-1.0, 1.0, -1.0, 1.0])
        base = np.ones((4, 1))
        candidates = np.column_stack([first, second])
        outcome = 2.0 + 4.0 * first + 2.0 * second

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            ("intercept",),
            ("first", "second"),
            lambda_ratio=0.25,
            penalty_weights=np.array([2.0, 0.5]),
            tolerance=1e-12,
        )

        # Hand calculation: weighted correlation ratios are 4/2=2 and
        # 2/0.5=4, so lambda-max is 4 and the fitted lambda is 1.
        self.assertAlmostEqual(result.lambda_max, 4.0, places=12)
        self.assertAlmostEqual(result.lambda_value, 1.0, places=12)
        np.testing.assert_allclose(
            result.standardized_candidate_coefficients,
            np.array([2.0, 1.5]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.base_coefficients,
            np.array([2.0]),
            rtol=0.0,
            atol=1e-12,
        )

        # Remaining residual coefficients are 2 and 0.5.  Thus loss is
        # (2^2+0.5^2)/2=2.125 and weighted L1 penalty is 4.75.
        self.assertAlmostEqual(result.objective, 6.875, places=12)
        residual = (
            outcome
            - base @ result.base_coefficients
            - candidates @ result.candidate_coefficients
        )
        active_correlations = candidates.T @ residual / len(outcome)
        np.testing.assert_allclose(
            active_correlations,
            np.array([2.0, 0.5]),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertLessEqual(result.kkt_violation, 1e-12)

    def test_qr_residualization_matches_least_squares_projection_oracle(
        self,
    ) -> None:
        """Protect QR residualization against an independent projection route.

        Args:
            self: Test case providing nonorthogonal base and candidates.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Confirms lambda-max uses only candidate and outcome variation left
            after the base design, using least-squares residuals as the oracle.
        """
        trend = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        base = np.column_stack([np.ones(6), trend])
        candidates = np.array(
            [
                [-2.0, 1.0],
                [1.0, -1.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [-1.0, 3.0],
                [3.0, -2.0],
            ]
        )
        outcome = np.array([1.0, 4.0, 2.0, 8.0, 3.0, 10.0])
        weights = np.array([1.0, 2.0])

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            ("intercept", "trend"),
            ("candidate_a", "candidate_b"),
            lambda_ratio=1.0,
            penalty_weights=weights,
        )

        # This oracle uses SVD least squares, not the production QR route, to
        # remove base-fitted values from the response and scaled candidates.
        standardized = (
            candidates - candidates.mean(axis=0)
        ) / candidates.std(axis=0, ddof=0)
        response_residual = outcome - base @ np.linalg.lstsq(
            base,
            outcome,
            rcond=None,
        )[0]
        candidate_residual = standardized - base @ np.linalg.lstsq(
            base,
            standardized,
            rcond=None,
        )[0]
        expected_lambda_max = np.max(
            np.abs(candidate_residual.T @ response_residual)
            / (len(outcome) * weights)
        )

        self.assertAlmostEqual(
            result.lambda_max,
            float(expected_lambda_max),
            places=12,
        )

    def test_correlated_candidate_solution_matches_active_set_oracle(self) -> None:
        """Protect coordinate descent for jointly active correlated predictors.

        Args:
            self: Test case providing two standardized candidates correlated .5.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Checks the converged solution against the closed-form positive
            active-set equations instead of an orthogonal special case.
        """
        first = np.array([-1.0, -1.0, 1.0, 1.0])
        orthogonal = np.array([-1.0, 1.0, -1.0, 1.0])
        second = 0.5 * first + np.sqrt(0.75) * orthogonal
        base = np.ones((4, 1))
        candidates = np.column_stack([first, second])
        outcome = 2.0 + 3.0 * first + 2.0 * second

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            ("intercept",),
            ("first", "second"),
            lambda_ratio=0.25,
            tolerance=1e-12,
        )

        # With Gram matrix [[1,.5],[.5,1]], correlations [4,3.5], and
        # lambda 1, the positive active-set solution is [7/3,4/3].
        np.testing.assert_allclose(
            result.standardized_candidate_coefficients,
            np.array([7.0 / 3.0, 4.0 / 3.0]),
            rtol=0.0,
            atol=2e-12,
        )
        self.assertLessEqual(result.kkt_violation, 1e-12)

    def test_prediction_uses_training_scaling_and_stored_raw_coefficients(
        self,
    ) -> None:
        """Protect prediction from refitting scales on new observations.

        Args:
            self: Test case providing deliberately shifted prediction rows.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Ensures held-out predictions apply the fitted raw-scale equation;
            test-set means and scales must never enter prediction.
        """
        outcome, base, candidates, base_names, candidate_names = (
            _make_signal_problem()
        )
        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            base_names,
            candidate_names,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )
        new_base = np.array([[1.0, -0.5], [1.0, 0.5]])
        new_candidates = np.array([[100.0, -20.0], [104.0, 30.0]])

        prediction = predict_partially_penalized_lasso(
            result,
            new_base,
            new_candidates,
        )
        literal_raw_prediction = (
            new_base @ result.base_coefficients
            + new_candidates @ result.candidate_coefficients
        )

        np.testing.assert_allclose(
            prediction,
            literal_raw_prediction,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.candidate_means,
            np.array([0.0, 0.0]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.candidate_scales,
            np.array([np.sqrt(2.5), np.sqrt(2.0 / 3.0)]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_repeated_fits_are_deterministic_and_arrays_are_read_only(self) -> None:
        """Protect deterministic fitting and immutable numerical state.

        Args:
            self: Test case providing repeated identical fits.

        Returns:
            None.

        Side effects:
            Attempts result mutation and expects NumPy to reject it.

        Statistical intent:
            Fixes coordinate order and prevents post-fit changes from making
            stored objectives or KKT diagnostics inconsistent.
        """
        arguments = _make_signal_problem()
        first = fit_partially_penalized_lasso(
            *arguments,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )
        second = fit_partially_penalized_lasso(
            *arguments,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )

        np.testing.assert_array_equal(
            first.candidate_coefficients,
            second.candidate_coefficients,
        )
        self.assertEqual(first.iterations, second.iterations)
        for stored_array in (
            first.base_coefficients,
            first.candidate_coefficients,
            first.standardized_candidate_coefficients,
            first.candidate_means,
            first.candidate_scales,
        ):
            with self.subTest(shape=stored_array.shape):
                self.assertFalse(stored_array.flags.writeable)
                with self.assertRaises(ValueError):
                    stored_array[...] = 0.0
                with self.assertRaises(ValueError):
                    stored_array.setflags(write=True)

    def test_base_design_must_span_a_constant_for_raw_back_transformation(
        self,
    ) -> None:
        """Protect the intercept requirement behind raw-scale prediction.

        Args:
            self: Test case providing empty and intercept-free base designs.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Ensures candidate centering can be absorbed into unpenalized base
            coefficients when standardized effects are converted to raw units.
        """
        outcome, base, candidates, _, candidate_names = _make_signal_problem()
        invalid_bases = (
            (np.empty((len(outcome), 0)), ()),
            (base[:, [1]], ("trend_only",)),
        )

        for invalid_base, invalid_names in invalid_bases:
            with self.subTest(width=invalid_base.shape[1]):
                with self.assertRaisesRegex(ValueError, "constant|intercept"):
                    fit_partially_penalized_lasso(
                        outcome,
                        invalid_base,
                        candidates,
                        invalid_names,
                        candidate_names,
                        lambda_ratio=0.5,
                    )

    def test_base_design_accepts_a_constant_spanned_by_multiple_columns(
        self,
    ) -> None:
        """Protect span-based intercept validation rather than name matching.

        Args:
            self: Test case providing no literal all-ones base column.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Allows any full-rank parameterization whose column space contains a
            constant, preserving model equivalence under base reparameterization.
        """
        outcome, base, candidates, _, candidate_names = _make_signal_problem()
        transformed_base = np.column_stack(
            [base[:, 0] + base[:, 1], base[:, 0] - base[:, 1]]
        )

        result = fit_partially_penalized_lasso(
            outcome,
            transformed_base,
            candidates,
            ("sum_contrast", "difference_contrast"),
            candidate_names,
            lambda_ratio=1.0,
        )

        np.testing.assert_array_equal(
            result.candidate_coefficients,
            np.zeros(2),
        )

    def test_huge_finite_candidate_is_scaled_without_overflow(self) -> None:
        """Protect stable population scaling for extreme finite magnitudes.

        Args:
            self: Test case providing a symmetric near-limit candidate.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Requires mathematically representable means, scales, coefficients,
            and predictions to remain finite without squaring huge raw values.
        """
        largest = np.finfo(float).max
        candidate = np.array([largest, largest, -largest, -largest])
        base = np.ones((4, 1))
        candidates = candidate[:, None]
        outcome = 2.0 + 3.0 * np.array([1.0, 1.0, -1.0, -1.0])

        result = fit_partially_penalized_lasso(
            outcome,
            base,
            candidates,
            ("intercept",),
            ("huge_candidate",),
            lambda_ratio=0.0,
            tolerance=1e-12,
        )
        prediction = predict_partially_penalized_lasso(
            result,
            base,
            candidates,
        )

        self.assertTrue(np.isfinite(result.candidate_scales).all())
        self.assertEqual(result.candidate_scales[0], largest)
        self.assertTrue(np.isfinite(result.candidate_coefficients).all())
        np.testing.assert_allclose(
            prediction,
            outcome,
            rtol=0.0,
            atol=1e-12,
        )

    def test_nonfinite_derived_quantities_from_finite_inputs_are_rejected(
        self,
    ) -> None:
        """Protect every fitted result from finite-input arithmetic overflow.

        Args:
            self: Test case providing centering and objective overflow fixtures.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Rejects unrepresentable centered ranges and residual objectives
            instead of returning silent exclusions or infinite diagnostics.
        """
        largest = np.finfo(float).max
        base = np.ones((4, 1))
        ordinary_candidate = np.array([-1.0, 1.0, -1.0, 1.0])[:, None]
        overflowing_center = np.array(
            [largest, -largest, -largest, -largest]
        )[:, None]
        huge_outcome = np.array([largest, -largest, largest, -largest])

        invalid_calls = (
            lambda: fit_partially_penalized_lasso(
                np.array([1.0, 2.0, 3.0, 4.0]),
                base,
                overflowing_center,
                ("intercept",),
                ("overflowing_center",),
                lambda_ratio=0.5,
            ),
            lambda: fit_partially_penalized_lasso(
                huge_outcome,
                base,
                ordinary_candidate,
                ("intercept",),
                ("ordinary_candidate",),
                lambda_ratio=0.5,
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaisesRegex(ValueError, "finite|overflow"):
                    invalid_call()

    def test_fit_rejects_invalid_shapes_values_names_and_controls(self) -> None:
        """Protect the numerical and identification input contract.

        Args:
            self: Test case providing malformed-input subtests.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Stops rank-deficient bases, nonfinite values, ambiguous names, and
            invalid penalty settings before optimization.
        """
        outcome, base, candidates, base_names, candidate_names = (
            _make_signal_problem()
        )
        nonfinite_candidates = candidates.copy()
        nonfinite_candidates[0, 0] = np.nan
        invalid_calls = {
            "row mismatch": lambda: fit_partially_penalized_lasso(
                outcome[:-1], base, candidates, base_names, candidate_names, 0.5
            ),
            "nonfinite": lambda: fit_partially_penalized_lasso(
                outcome,
                base,
                nonfinite_candidates,
                base_names,
                candidate_names,
                0.5,
            ),
            "duplicate names": lambda: fit_partially_penalized_lasso(
                outcome,
                base,
                candidates,
                base_names,
                ("same", "same"),
                0.5,
            ),
            "rank deficient base": lambda: fit_partially_penalized_lasso(
                outcome,
                np.column_stack([base[:, 0], base[:, 0]]),
                candidates,
                ("intercept", "duplicate_intercept"),
                candidate_names,
                0.5,
            ),
            "invalid ratio": lambda: fit_partially_penalized_lasso(
                outcome, base, candidates, base_names, candidate_names, 1.1
            ),
            "invalid weights": lambda: fit_partially_penalized_lasso(
                outcome,
                base,
                candidates,
                base_names,
                candidate_names,
                0.5,
                penalty_weights=np.array([1.0, 0.0]),
            ),
        }

        for case_name, invalid_call in invalid_calls.items():
            with self.subTest(case=case_name):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()

    def test_prediction_rejects_invalid_or_nonfinite_designs(self) -> None:
        """Protect fitted-schema and finite-value checks at prediction time.

        Args:
            self: Test case providing invalid prediction matrices.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents silently reordered, truncated, or undefined predictors
            from producing apparently valid model predictions.
        """
        arguments = _make_signal_problem()
        result = fit_partially_penalized_lasso(
            *arguments,
            lambda_ratio=0.1,
            tolerance=1e-12,
        )

        with self.assertRaises(ValueError):
            predict_partially_penalized_lasso(
                result,
                np.ones((2, 1)),
                np.ones((2, 2)),
            )
        with self.assertRaises(ValueError):
            predict_partially_penalized_lasso(
                result,
                np.array([[1.0, np.inf]]),
                np.ones((1, 2)),
            )


class CollinearityDiagnosticTestCase(unittest.TestCase):
    """Exercise rank and null-space reporting without culprit attribution."""

    def test_detects_module_mean_dependency_as_a_shared_null_space(self) -> None:
        """Protect exact module-domain dependency detection.

        Args:
            self: Test case providing four domains and their row mean.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Identifies the five-column dependency as a joint linear relation;
            no individual module variable is labeled as the unique culprit.
        """
        domains = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.2, 0.4, 0.6, 0.8],
            ]
        )
        module_mean = domains.mean(axis=1)
        design = np.column_stack([domains, module_mean])
        names = (
            "module_core",
            "module_mindset",
            "module_nutrition",
            "module_physical_activity",
            "module_mean",
        )

        diagnostic = diagnose_collinearity(design, names)

        self.assertTrue(diagnostic.is_rank_deficient)
        self.assertEqual(diagnostic.rank, 4)
        self.assertEqual(diagnostic.n_rows, 6)
        self.assertEqual(diagnostic.n_columns, 5)
        self.assertEqual(diagnostic.column_names, names)
        self.assertEqual(diagnostic.null_space_vectors.shape, (1, 5))
        np.testing.assert_allclose(
            design @ diagnostic.null_space_vectors.T,
            np.zeros((6, 1)),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertFalse(diagnostic.singular_values.flags.writeable)
        self.assertFalse(diagnostic.null_space_vectors.flags.writeable)
        for stored_array in (
            diagnostic.singular_values,
            diagnostic.null_space_vectors,
        ):
            with self.assertRaises(ValueError):
                stored_array.setflags(write=True)

    def test_collinearity_rejects_nonfinite_svd_from_huge_finite_matrix(
        self,
    ) -> None:
        """Protect diagnostics from singular-value overflow.

        Args:
            self: Test case providing an extreme but finite matrix.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Rejects an unrepresentable singular spectrum instead of reporting
            a misleading rank or null space based on infinite singular values.
        """
        largest = np.finfo(float).max
        design = np.array(
            [[largest, largest], [largest, -largest]],
            dtype=float,
        )

        with self.assertRaisesRegex(ValueError, "SVD|singular|finite"):
            diagnose_collinearity(design, ("first", "second"))

    def test_collinearity_rejects_name_shape_and_finite_violations(self) -> None:
        """Protect diagnostic inputs from ambiguous or undefined matrices.

        Args:
            self: Test case providing malformed diagnostic requests.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Ensures the reported rank and null space describe a named finite
            matrix with one label per predictor.
        """
        valid = np.eye(3)
        invalid_calls = (
            lambda: diagnose_collinearity(valid.ravel(), ("a", "b", "c")),
            lambda: diagnose_collinearity(valid, ("a", "b")),
            lambda: diagnose_collinearity(valid, ("a", "a", "c")),
            lambda: diagnose_collinearity(
                np.array([[1.0, np.nan]]), ("a", "b")
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
