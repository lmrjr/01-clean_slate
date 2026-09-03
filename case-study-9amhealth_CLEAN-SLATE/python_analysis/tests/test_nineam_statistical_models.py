"""Tests for the two base statistical-model families."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from nineam_health_analysis.nineam_statistical_models import (
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
    predict_last_weight_longitudinal,
    predict_last_weight_percentage_loss,
)


def _make_longitudinal_members(
    *,
    group_b_change: float = -10.0,
) -> pd.DataFrame:
    """Create paired raw weights with hand-derived GLS parameters.

    Args:
        group_b_change: Mean follow-up change assigned to member type B.

    Returns:
        Eight member rows whose UN factorization has known coefficients,
        covariance, and model-based standard errors.

    Side effects:
        None; a new dataframe and arrays are created on every call.

    Statistical intent:
        Makes the difference residual orthogonal to each type and the baseline
        residual orthogonal to both the type design and difference residual.
    """
    member_type = np.array(["A"] * 4 + ["B"] * 4)
    group_b = (member_type == "B").astype(float)
    difference_residual = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    )
    baseline_residual = np.array(
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    )

    # Construct baseline weight from an intercept, a type-B contrast, and the
    # conditional UN representation with known kappa equal to -0.25.
    first_weight = (
        100.0
        + 20.0 * group_b
        - 0.25 * difference_residual
        + baseline_residual
    )

    # Give type A a -10 change and optionally give type B a different change;
    # the fixed residual pattern preserves exact group-specific estimates.
    mean_change = -10.0 + (group_b_change + 10.0) * group_b
    last_weight = first_weight + mean_change + difference_residual
    return pd.DataFrame(
        {
            "member_id": [f"member_{index}" for index in range(8)],
            "member_type": member_type,
            "first_weight": first_weight,
            "last_weight": last_weight,
        }
    )


def _make_log_longitudinal_members() -> pd.DataFrame:
    """Create paired weights with known parameters on the logarithmic scale.

    Args:
        None.

    Returns:
        Eight positive raw-weight pairs generated from a known UN model in log
        weight.

    Side effects:
        None; a new dataframe and arrays are created on every call.

    Statistical intent:
        Supports independent tests of the lognormal Jacobian and conditional
        raw-scale mean correction.
    """
    member_type = np.array(["A"] * 4 + ["B"] * 4)
    group_b = (member_type == "B").astype(float)
    difference_residual = 0.01 * np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    )
    baseline_residual = 0.005 * np.array(
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    )

    # Generate log baseline weights with a known group contrast and the same
    # conditional factorization used in the raw-scale fixture.
    first_log_weight = (
        math.log(100.0)
        + math.log(1.2) * group_b
        - 0.25 * difference_residual
        + baseline_residual
    )

    # Exponentiation produces valid raw weights while retaining an exactly
    # known Gaussian likelihood and covariance on the transformed scale.
    last_log_weight = (
        first_log_weight + math.log(0.95) + difference_residual
    )
    return pd.DataFrame(
        {
            "member_id": [f"member_{index}" for index in range(8)],
            "member_type": member_type,
            "first_weight": np.exp(first_log_weight),
            "last_weight": np.exp(last_log_weight),
        }
    )


def _make_percentage_members() -> pd.DataFrame:
    """Create percentage-loss observations with an exact OLS solution.

    Args:
        None.

    Returns:
        Eight member rows generated from a known intercept, baseline-weight
        slope, type-B contrast, and orthogonal residual.

    Side effects:
        None; a new dataframe and arrays are created on every call.

    Statistical intent:
        Separates the ML residual variance used by the likelihood from the
        unbiased residual variance used by conventional OLS standard errors.
    """
    first_weight = np.array(
        [100.0, 120.0, 140.0, 160.0, 110.0, 130.0, 150.0, 170.0]
    )
    member_type = np.array(["A"] * 4 + ["B"] * 4)
    group_b = (member_type == "B").astype(float)
    residual = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0])

    # The residual has zero sums and zero baseline-weight cross-product within
    # the factor design, so the literal generating coefficients are OLS exact.
    percentage_loss = 2.0 + 0.1 * first_weight + 3.0 * group_b + residual

    # Invert the confirmed percentage-loss formula to supply raw last weights
    # rather than passing a precomputed outcome into the production function.
    last_weight = first_weight * (1.0 - percentage_loss / 100.0)
    return pd.DataFrame(
        {
            "member_id": [f"member_{index}" for index in range(8)],
            "member_type": member_type,
            "first_weight": first_weight,
            "last_weight": last_weight,
        }
    )


def _make_diagonal_boundary_members(
    zero_variance_occasion: str,
) -> pd.DataFrame:
    """Create data whose diagonal covariance MLE lies on a boundary.

    Args:
        zero_variance_occasion: ``"baseline"`` or ``"follow_up"`` indicating
            which occasion is fitted without residual variation.

    Returns:
        Eight paired member rows with one exact zero-variance occasion.

    Side effects:
        None; new arrays and a dataframe are created on every call.

    Statistical intent:
        Represents the exact counterexample where the open diagonal parameter
        space has a likelihood supremum but no finite interior Gaussian MLE.

    Raises:
        ValueError: If the requested zero-variance occasion is unknown.
    """
    member_type = np.array(["A"] * 4 + ["B"] * 4)
    group_b = (member_type == "B").astype(float)
    residual = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    )
    mean_baseline = 100.0 + 20.0 * group_b

    # Put all residual variation on only one occasion so the other fitted
    # diagonal variance is exactly zero at the likelihood boundary.
    if zero_variance_occasion == "baseline":
        first_weight = mean_baseline
        last_weight = mean_baseline - 10.0 + residual
    elif zero_variance_occasion == "follow_up":
        first_weight = mean_baseline + residual
        last_weight = mean_baseline - 10.0
    else:
        raise ValueError("zero_variance_occasion must name a fitted occasion")

    return pd.DataFrame(
        {
            "member_id": [f"boundary_{index}" for index in range(8)],
            "member_type": member_type,
            "first_weight": first_weight,
            "last_weight": last_weight,
        }
    )


def _make_near_exact_percentage_members(
    residual_scale: float,
) -> pd.DataFrame:
    """Create an OLS outcome with exact or near-exact residual variation.

    Args:
        residual_scale: Multiplier for a design-orthogonal residual pattern.

    Returns:
        Eight paired rows whose percentage loss is nearly deterministic from
        baseline weight and member type.

    Side effects:
        None; new arrays and a dataframe are created on every call.

    Statistical intent:
        Exposes likelihoods whose residual variance is numerically on the
        zero-variance boundary despite positive residual degrees of freedom.
    """
    first_weight = np.array(
        [100.0, 120.0, 140.0, 160.0, 110.0, 130.0, 150.0, 170.0]
    )
    member_type = np.array(["A"] * 4 + ["B"] * 4)
    group_b = (member_type == "B").astype(float)
    residual_pattern = np.array(
        [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
    )

    # The residual pattern is orthogonal to the complete design; scaling it to
    # zero or machine-negligible size creates the intended variance boundary.
    percentage_loss = (
        2.0
        + 0.1 * first_weight
        + 3.0 * group_b
        + residual_scale * residual_pattern
    )
    last_weight = first_weight * (1.0 - percentage_loss / 100.0)
    return pd.DataFrame(
        {
            "member_id": [f"near_exact_{index}" for index in range(8)],
            "member_type": member_type,
            "first_weight": first_weight,
            "last_weight": last_weight,
        }
    )


class LongitudinalGLSTestCase(unittest.TestCase):
    """Exercise ML GLS behavior with independently constructed paired data."""

    def test_confirmed_member_types_default_to_coaching_reference(self) -> None:
        """The confirmed comparison omits Coaching rather than sorted Active GLP-1."""
        members = _make_longitudinal_members().replace(
            {"A": "Coaching Only", "B": "Active GLP-1 for Weight-loss"}
        )
        result = fit_longitudinal_gls(members, covariance_structure="unstructured")
        self.assertEqual(result.schema.reference_member_type, "Coaching Only")
        self.assertIn("member_type[Active GLP-1 for Weight-loss]", result.coefficient_names)

    def test_unstructured_fit_recovers_coefficients_covariance_and_ses(
        self,
    ) -> None:
        """Protect factor coding and the exact two-occasion UN factorization.

        Args:
            self: Test case providing numerical assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Detects changes to reference coding, ML covariance divisors, or
            model-based fixed-effect standard errors.
        """
        result = fit_longitudinal_gls(
            _make_longitudinal_members(),
            outcome_scale="raw",
            covariance_structure="unstructured",
        )

        self.assertEqual(
            result.coefficient_names,
            ("intercept", "member_type[B]", "time"),
        )
        np.testing.assert_allclose(
            result.coefficients,
            np.array([100.0, 20.0, -10.0]),
            rtol=0.0,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result.covariance_matrix,
            np.array([[1.46875, -0.40625], [-0.40625, 5.21875]]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.standard_errors,
            np.array([0.5555121510822243, 0.7071067811865476, 0.9682458365518543]),
            rtol=0.0,
            atol=1e-11,
        )
        self.assertEqual(result.schema.reference_member_type, "A")
        self.assertEqual(result.schema.member_type_levels, ("A", "B"))
        self.assertEqual(result.design_rank, 3)

    def test_covariance_structures_match_derived_ml_fits_and_counts(
        self,
    ) -> None:
        """Protect derived IID, diagonal, CS, and UN maximum-likelihood fits.

        Args:
            self: Test case providing subtests and numerical assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Ensures every covariance label returns its independently derived
            coefficients, covariance, likelihood, and parameter count.
        """
        members = _make_longitudinal_members()
        covariance_parameter_counts = {
            "iid": 1,
            "diagonal": 2,
            "compound_symmetry": 2,
            "unstructured": 3,
        }
        expected_covariances = {
            "iid": np.array([[3.34375, 0.0], [0.0, 3.34375]]),
            "diagonal": np.array([[1.46875, 0.0], [0.0, 5.21875]]),
            "compound_symmetry": np.array(
                [[3.34375, -0.40625], [-0.40625, 3.34375]]
            ),
            "unstructured": np.array(
                [[1.46875, -0.40625], [-0.40625, 5.21875]]
            ),
        }
        expected_log_likelihoods = {
            "iid": -32.3597599845722,
            "diagonal": -30.849694965384206,
            "compound_symmetry": -32.30027540908515,
            "unstructured": -30.762628613443823,
        }

        for structure, covariance_count in covariance_parameter_counts.items():
            with self.subTest(covariance_structure=structure):
                result = fit_longitudinal_gls(
                    members,
                    outcome_scale="raw",
                    covariance_structure=structure,
                )
                covariance = result.covariance_matrix

                # Literal values were derived from orthogonal residual sums of
                # squares, independently of the production parameterizations.
                np.testing.assert_allclose(
                    result.coefficients,
                    np.array([100.0, 20.0, -10.0]),
                    rtol=0.0,
                    atol=1e-11,
                )
                np.testing.assert_allclose(
                    covariance,
                    expected_covariances[structure],
                    rtol=0.0,
                    atol=1e-7,
                )
                self.assertAlmostEqual(
                    result.transformed_log_likelihood,
                    expected_log_likelihoods[structure],
                    places=11,
                )
                self.assertEqual(result.parameter_count, 3 + covariance_count)

                if structure == "iid":
                    self.assertAlmostEqual(covariance[0, 1], 0.0, places=12)
                    self.assertAlmostEqual(
                        covariance[0, 0], covariance[1, 1], places=12
                    )
                elif structure == "diagonal":
                    self.assertAlmostEqual(covariance[0, 1], 0.0, places=12)
                    self.assertNotAlmostEqual(
                        covariance[0, 0], covariance[1, 1], places=6
                    )
                elif structure == "compound_symmetry":
                    self.assertAlmostEqual(
                        covariance[0, 0], covariance[1, 1], places=12
                    )
                else:
                    self.assertNotAlmostEqual(covariance[0, 1], 0.0, places=6)
                    self.assertNotAlmostEqual(
                        covariance[0, 0], covariance[1, 1], places=6
                    )

    def test_diagonal_fit_rejects_zero_variance_boundaries(self) -> None:
        """Protect against reporting a diagonal boundary as a finite MLE.

        Args:
            self: Test case providing exact boundary subtests.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Requires both occasion variances to be meaningfully inside the
            positive-definite parameter space, not optimizer guard values.
        """
        for zero_variance_occasion in ("baseline", "follow_up"):
            with self.subTest(occasion=zero_variance_occasion):
                members = _make_diagonal_boundary_members(
                    zero_variance_occasion
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "boundary|positive definite|interior",
                ):
                    fit_longitudinal_gls(
                        members,
                        covariance_structure="diagonal",
                    )

    def test_interaction_fit_estimates_type_specific_time_change(self) -> None:
        """Protect time-by-member-type design construction.

        Args:
            self: Test case providing numerical assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Confirms the sensitivity model estimates a reference time effect
            and an additional type-B time contrast without changing baseline
            factor coding.
        """
        result = fit_longitudinal_gls(
            _make_longitudinal_members(group_b_change=-5.0),
            outcome_scale="raw",
            covariance_structure="unstructured",
            include_time_by_member_type=True,
        )

        self.assertEqual(
            result.coefficient_names,
            (
                "intercept",
                "member_type[B]",
                "time",
                "time:member_type[B]",
            ),
        )
        np.testing.assert_allclose(
            result.coefficients,
            np.array([100.0, 20.0, -10.0, 5.0]),
            rtol=0.0,
            atol=1e-11,
        )
        self.assertTrue(result.include_time_by_member_type)
        self.assertEqual(result.design_rank, 4)
        self.assertEqual(result.parameter_count, 7)

    def test_unstructured_likelihood_includes_constants_and_member_bic(
        self,
    ) -> None:
        """Protect Gaussian likelihood constants and member-level BIC size.

        Args:
            self: Test case providing literal likelihood assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents omission of the normalizing constant, use of REML, or use
            of 16 observations instead of eight independent members in BIC.
        """
        result = fit_longitudinal_gls(
            _make_longitudinal_members(),
            outcome_scale="raw",
            covariance_structure="unstructured",
        )

        self.assertAlmostEqual(
            result.transformed_log_likelihood,
            -30.762628613443823,
            places=11,
        )
        self.assertAlmostEqual(
            result.raw_scale_log_likelihood,
            -30.762628613443823,
            places=11,
        )
        self.assertAlmostEqual(
            result.negative_two_log_likelihood,
            61.525257226887646,
            places=11,
        )
        self.assertAlmostEqual(result.aic, 73.52525722688765, places=11)
        self.assertAlmostEqual(result.bic, 74.00190647696667, places=11)
        self.assertEqual(result.n_members, 8)
        self.assertEqual(result.n_observations, 16)

    def test_log_likelihood_applies_raw_scale_jacobian(self) -> None:
        """Protect the lognormal change-of-variables adjustment.

        Args:
            self: Test case providing literal transformed-scale assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Keeps likelihood criteria for log-weight models expressed on the
            original weight density by subtracting both endpoint log weights.
        """
        result = fit_longitudinal_gls(
            _make_log_longitudinal_members(),
            outcome_scale="log",
            covariance_structure="unstructured",
        )

        self.assertAlmostEqual(
            result.transformed_log_likelihood,
            48.4652718068452,
            places=10,
        )
        self.assertAlmostEqual(
            result.raw_scale_log_likelihood,
            -26.265677268215505,
            places=10,
        )
        self.assertAlmostEqual(
            result.transformed_log_likelihood
            - result.raw_scale_log_likelihood,
            74.7309490750607,
            places=10,
        )
        self.assertAlmostEqual(result.aic, 64.53135453643101, places=10)
        self.assertEqual(result.outcome_scale, "log")

    def test_raw_prediction_conditions_on_observed_baseline(self) -> None:
        """Protect conditional raw-scale follow-up prediction.

        Args:
            self: Test case providing a literal conditional-mean assertion.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Ensures prediction uses the fitted within-member covariance rather
            than returning only the marginal follow-up mean.
        """
        result = fit_longitudinal_gls(
            _make_longitudinal_members(),
            outcome_scale="raw",
            covariance_structure="unstructured",
        )
        new_members = pd.DataFrame(
            {
                "member_id": ["new_a"],
                "member_type": ["A"],
                "first_weight": [102.0],
            }
        )

        prediction = predict_last_weight_longitudinal(result, new_members)

        np.testing.assert_allclose(
            prediction,
            np.array([89.44680851063829]),
            rtol=0.0,
            atol=1e-11,
        )

    def test_log_prediction_returns_conditional_raw_scale_mean(self) -> None:
        """Protect lognormal conditional-mean bias correction.

        Args:
            self: Test case providing a literal conditional-mean assertion.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Requires exponentiation of the conditional log mean plus half its
            conditional variance, rather than exponentiating the mean alone.
        """
        result = fit_longitudinal_gls(
            _make_log_longitudinal_members(),
            outcome_scale="log",
            covariance_structure="unstructured",
        )
        new_members = pd.DataFrame(
            {
                "member_id": ["new_a"],
                "member_type": ["A"],
                "first_weight": [102.02013400267558],
            }
        )

        prediction = predict_last_weight_longitudinal(result, new_members)

        np.testing.assert_allclose(
            prediction,
            np.array([92.00412503121295]),
            rtol=0.0,
            atol=1e-11,
        )


class PercentageLossOLSTestCase(unittest.TestCase):
    """Exercise percentage-loss OLS with hand-derived regression results."""

    def test_ols_recovers_coefficients_variances_metrics_and_ses(self) -> None:
        """Protect the percentage outcome and conventional OLS calculations.

        Args:
            self: Test case providing literal regression assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Separates RSS/n likelihood variance from RSS/(n-p) standard-error
            variance while preserving factor and baseline-weight coefficients.
        """
        result = fit_percentage_loss_ols(_make_percentage_members())

        self.assertEqual(
            result.coefficient_names,
            ("intercept", "first_weight", "member_type[B]"),
        )
        np.testing.assert_allclose(
            result.coefficients,
            np.array([2.0, 0.1, 3.0]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.standard_errors,
            np.array([2.6758176320519302, 0.02, 0.9165151389911679]),
            rtol=0.0,
            atol=1e-11,
        )
        self.assertAlmostEqual(result.ml_residual_variance, 1.0, places=12)
        self.assertAlmostEqual(
            result.unbiased_residual_variance, 1.6, places=12
        )
        self.assertAlmostEqual(result.log_likelihood, -11.351508265637381, places=11)
        self.assertAlmostEqual(
            result.negative_two_log_likelihood,
            22.703016531274763,
            places=11,
        )
        self.assertAlmostEqual(result.aic, 30.703016531274763, places=11)
        self.assertAlmostEqual(result.bic, 31.020782697994107, places=11)
        self.assertAlmostEqual(result.r_squared, 0.9, places=12)
        self.assertAlmostEqual(result.adjusted_r_squared, 0.86, places=12)
        self.assertAlmostEqual(result.rmse, 1.0, places=12)
        self.assertEqual(result.parameter_count, 4)
        self.assertEqual(result.design_rank, 3)
        self.assertEqual(result.schema.reference_member_type, "A")

    def test_ols_prediction_converts_percentage_to_unclipped_last_weight(
        self,
    ) -> None:
        """Protect conversion from predicted percentage loss to last weight.

        Args:
            self: Test case providing literal feed-forward predictions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Applies the algebraic inverse of percentage loss without clipping
            valid model extrapolations.
        """
        result = fit_percentage_loss_ols(_make_percentage_members())
        new_members = pd.DataFrame(
            {
                "member_id": ["new_a", "new_b"],
                "member_type": ["A", "B"],
                "first_weight": [125.0, 125.0],
            }
        )

        prediction = predict_last_weight_percentage_loss(result, new_members)

        np.testing.assert_allclose(
            prediction,
            np.array([106.875, 103.125]),
            rtol=0.0,
            atol=1e-11,
        )

    def test_ols_standard_errors_remain_stable_with_large_baseline_offset(
        self,
    ) -> None:
        """Protect QR/SVD covariance calculation from normal-equation loss.

        Args:
            self: Test case providing literal standard-error assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Uses a full-rank but ill-conditioned raw predictor scale where
            forming X-prime-X materially corrupts the covariance calculation.
        """
        members = _make_percentage_members().copy()
        baseline_shift = 100_000_000.0 - 100.0
        members["first_weight"] += baseline_shift

        # Preserve each fixture member's original percentage loss after shifting
        # baseline scale so the analytic residual variance remains 1.6.
        original = _make_percentage_members()
        original_percentage_loss = 100.0 * (
            original["first_weight"] - original["last_weight"]
        ) / original["first_weight"]
        members["last_weight"] = members["first_weight"] * (
            1.0 - original_percentage_loss / 100.0
        )

        result = fit_percentage_loss_ols(members)

        np.testing.assert_allclose(
            result.standard_errors,
            np.array([2_000_000.6000001, 0.02, 0.9165151389911679]),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_ols_rejects_exact_and_near_exact_residual_variance(self) -> None:
        """Protect against a spurious finite likelihood at zero variance.

        Args:
            self: Test case providing exact and numerical-boundary subtests.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Rejects exact and scale-negligible residual norms before logging an
            effectively zero ML variance or reporting meaningless tiny SEs.
        """
        for residual_scale in (0.0, 1e-8):
            with self.subTest(residual_scale=residual_scale):
                with self.assertRaisesRegex(ValueError, "residual variance"):
                    fit_percentage_loss_ols(
                        _make_near_exact_percentage_members(residual_scale)
                    )


class StatisticalModelValidationTestCase(unittest.TestCase):
    """Exercise failures that prevent invalid or unidentified model fits."""

    def test_fit_rejects_invalid_member_rows_and_weights(self) -> None:
        """Protect the one-row, two-positive-finite-weights contract.

        Args:
            self: Test case providing invalid-input subtests.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents duplicated members or undefined transformations from
            entering either model family.
        """
        valid = _make_longitudinal_members()
        invalid_cases = {
            "duplicate member": pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
            "zero weight": valid.assign(
                first_weight=lambda table: table["first_weight"].mask(
                    table.index == 0, 0.0
                )
            ),
            "nonfinite weight": valid.assign(
                last_weight=lambda table: table["last_weight"].mask(
                    table.index == 0, np.inf
                )
            ),
            "blank type": valid.assign(
                member_type=lambda table: table["member_type"].mask(
                    table.index == 0, "   "
                )
            ),
        }

        for case_name, members in invalid_cases.items():
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    fit_longitudinal_gls(members)
                with self.assertRaises(ValueError):
                    fit_percentage_loss_ols(members)

    def test_fit_rejects_rank_deficiency_and_degenerate_covariance(self) -> None:
        """Protect estimability and positive-definite covariance checks.

        Args:
            self: Test case providing invalid model fixtures.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Stops singular fixed-effect systems and zero residual-variance
            likelihoods instead of returning unstable coefficients.
        """
        rank_deficient = _make_percentage_members().assign(first_weight=100.0)
        rank_deficient["last_weight"] = 90.0
        with self.assertRaisesRegex(ValueError, "rank"):
            fit_percentage_loss_ols(rank_deficient)

        degenerate = _make_longitudinal_members()
        degenerate["last_weight"] = degenerate["first_weight"] - 10.0
        with self.assertRaisesRegex(ValueError, "positive definite|variance"):
            fit_longitudinal_gls(
                degenerate,
                covariance_structure="unstructured",
            )

    def test_fit_rejects_unknown_model_options(self) -> None:
        """Protect the explicit scale and covariance option vocabulary.

        Args:
            self: Test case providing invalid-option assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Prevents a misspelled model definition from silently selecting a
            different likelihood.
        """
        members = _make_longitudinal_members()
        with self.assertRaisesRegex(ValueError, "outcome_scale"):
            fit_longitudinal_gls(members, outcome_scale="square_root")
        with self.assertRaisesRegex(ValueError, "covariance_structure"):
            fit_longitudinal_gls(members, covariance_structure="exchangeable")

    def test_predictions_reject_unseen_member_type_levels(self) -> None:
        """Protect training-schema enforcement during prediction.

        Args:
            self: Test case providing unseen-level assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Avoids silently treating an unseen category as the fitted reference
            group in either prediction family.
        """
        longitudinal = fit_longitudinal_gls(_make_longitudinal_members())
        percentage = fit_percentage_loss_ols(_make_percentage_members())
        unseen = pd.DataFrame(
            {
                "member_id": ["new_c"],
                "member_type": ["C"],
                "first_weight": [125.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "unseen"):
            predict_last_weight_longitudinal(longitudinal, unseen)
        with self.assertRaisesRegex(ValueError, "unseen"):
            predict_last_weight_percentage_loss(percentage, unseen)

    def test_frozen_results_store_only_read_only_arrays(self) -> None:
        """Protect fitted result values from mutation through ndarray fields.

        Args:
            self: Test case providing write-protection assertions.

        Returns:
            None.

        Side effects:
            Attempts mutation and expects NumPy to reject it.

        Statistical intent:
            Keeps coefficients, standard errors, and covariance synchronized
            with immutable metadata and likelihood diagnostics after fitting.
        """
        longitudinal = fit_longitudinal_gls(_make_longitudinal_members())
        percentage = fit_percentage_loss_ols(_make_percentage_members())
        stored_arrays = (
            longitudinal.coefficients,
            longitudinal.standard_errors,
            longitudinal.covariance_matrix,
            percentage.coefficients,
            percentage.standard_errors,
        )

        for stored_array in stored_arrays:
            with self.subTest(shape=stored_array.shape):
                self.assertFalse(stored_array.flags.writeable)
                with self.assertRaises(ValueError):
                    stored_array[...] = 0.0
                # A view backed by ordinary writable memory can re-enable its
                # write flag. Requiring this failure proves result arrays use
                # immutable backing storage rather than a cosmetic flag only.
                with self.assertRaises(ValueError):
                    stored_array.setflags(write=True)

    def test_longitudinal_result_records_immutable_training_member_ids(self) -> None:
        """Retain exact training-sample provenance for two-stage whitening.

        Args:
            self: Test case providing provenance assertions.

        Returns:
            None.

        Side effects:
            None.

        Statistical intent:
            Allows downstream penalized GLS construction to reject a covariance
            estimate fitted on a different set of members, even when their
            numeric covariates happen to reproduce the same estimates.
        """
        members = _make_longitudinal_members()
        result = fit_longitudinal_gls(members)

        self.assertIsInstance(result.training_member_ids, tuple)
        self.assertEqual(
            result.training_member_ids,
            tuple(members["member_id"].astype(str)),
        )


if __name__ == "__main__":
    unittest.main()
