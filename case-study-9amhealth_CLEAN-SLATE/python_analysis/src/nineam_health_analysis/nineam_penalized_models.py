"""Fit partially penalized LASSO models and diagnose linear dependencies.

The LASSO implementation leaves a supplied base design unpenalized, scales
candidate predictors on training data, and solves the residualized problem by
deterministic cyclic coordinate descent.  Only NumPy and the Python standard
library are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_read_only_float_array(values: ArrayLike) -> FloatArray:
    """Return an isolated, immutable float array.

    Args:
        values: Numeric values to store in a fitted result.

    Returns:
        A float copy with NumPy's write flag disabled.

    Side effects:
        None; caller-owned values are not modified.

    Statistical intent:
        Prevents fitted coefficients or diagnostics from being changed after
        their objective and convergence measures have been calculated.
    """
    contiguous_values = np.ascontiguousarray(values, dtype=float)

    # A NumPy-owned buffer can later have write access re-enabled.  Rebuilding
    # the array over immutable ``bytes`` makes that reversal impossible while
    # still isolating the result from every caller-owned array.
    immutable_buffer = contiguous_values.tobytes(order="C")
    immutable_values = np.frombuffer(
        immutable_buffer,
        dtype=np.float64,
    ).reshape(contiguous_values.shape)
    return immutable_values


@dataclass(frozen=True, slots=True)
class PartiallyPenalizedLassoResult:
    """Store one partially penalized LASSO fit on raw and scaled coordinates."""

    base_names: tuple[str, ...]
    candidate_names: tuple[str, ...]
    base_coefficients: FloatArray
    candidate_coefficients: FloatArray
    standardized_candidate_coefficients: FloatArray
    candidate_means: FloatArray
    candidate_scales: FloatArray
    lambda_value: float
    lambda_max: float
    lambda_ratio: float
    objective: float
    kkt_violation: float
    iterations: int
    excluded_candidate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Copy every numerical vector into immutable result-owned storage.

        Args:
            self: Newly constructed fitted result.

        Returns:
            None.

        Side effects:
            Replaces array fields on this frozen instance with read-only copies.

        Statistical intent:
            Keeps coefficients, scaling metadata, the objective, and KKT
            diagnostics internally consistent after fitting.
        """
        array_fields = (
            "base_coefficients",
            "candidate_coefficients",
            "standardized_candidate_coefficients",
            "candidate_means",
            "candidate_scales",
        )
        for field_name in array_fields:
            # Frozen dataclasses permit controlled assignment only during
            # post-init; copying also breaks every caller-held array alias.
            object.__setattr__(
                self,
                field_name,
                _as_read_only_float_array(getattr(self, field_name)),
            )


@dataclass(frozen=True, slots=True)
class CollinearityDiagnostic:
    """Describe matrix rank and every estimated null-space relationship."""

    column_names: tuple[str, ...]
    rank: int
    n_rows: int
    n_columns: int
    singular_values: FloatArray
    null_space_vectors: FloatArray
    is_rank_deficient: bool

    def __post_init__(self) -> None:
        """Copy singular values and null-space vectors into immutable storage.

        Args:
            self: Newly constructed collinearity diagnostic.

        Returns:
            None.

        Side effects:
            Replaces numerical fields with read-only copies.

        Statistical intent:
            Prevents post-diagnostic mutation from invalidating the reported
            rank or shared linear dependencies.
        """
        object.__setattr__(
            self,
            "singular_values",
            _as_read_only_float_array(self.singular_values),
        )
        object.__setattr__(
            self,
            "null_space_vectors",
            _as_read_only_float_array(self.null_space_vectors),
        )


def _validate_names(
    names: tuple[str, ...] | list[str],
    expected_count: int,
    label: str,
) -> tuple[str, ...]:
    """Validate one unambiguous predictor-name sequence.

    Args:
        names: Labels supplied in design-matrix column order.
        expected_count: Number of labels required by the matrix width.
        label: Human-readable sequence name used in errors.

    Returns:
        The validated labels as an immutable tuple.

    Side effects:
        None.

    Statistical intent:
        Preserves a one-to-one mapping between coefficients and predictors.

    Raises:
        TypeError: If names are not a sequence of strings.
        ValueError: If counts, blank labels, or uniqueness are invalid.
    """
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings")
    try:
        validated = tuple(names)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence of strings") from error

    if len(validated) != expected_count:
        raise ValueError(
            f"{label} must contain exactly {expected_count} column names"
        )
    if any(not isinstance(name, str) for name in validated):
        raise TypeError(f"{label} must contain only strings")
    if any(not name.strip() for name in validated):
        raise ValueError(f"{label} cannot contain blank names")
    if len(set(validated)) != len(validated):
        raise ValueError(f"{label} must contain unique names")
    return validated


def _as_finite_vector(values: ArrayLike, label: str) -> FloatArray:
    """Convert and validate a finite one-dimensional numeric vector.

    Args:
        values: Array-like values to use as an outcome or weight vector.
        label: Human-readable value name used in errors.

    Returns:
        A float vector view or copy suitable for numerical calculations.

    Side effects:
        None.

    Statistical intent:
        Prevents missing or infinite quantities from entering optimization.

    Raises:
        TypeError: If values cannot be converted to numeric data.
        ValueError: If values are not a nonempty finite vector.
    """
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain numeric values") from error
    if vector.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{label} must contain at least one observation")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain only finite values")
    return vector


def _as_finite_matrix(values: ArrayLike, label: str) -> FloatArray:
    """Convert and validate a finite two-dimensional numeric matrix.

    Args:
        values: Array-like design or diagnostic matrix.
        label: Human-readable matrix name used in errors.

    Returns:
        A float matrix view or copy suitable for numerical calculations.

    Side effects:
        None.

    Statistical intent:
        Enforces an explicit observation-by-predictor matrix contract.

    Raises:
        TypeError: If values cannot be converted to numeric data.
        ValueError: If values are not a finite two-dimensional matrix.
    """
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain numeric values") from error
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must contain only finite values")
    return matrix


def _require_finite_derived(values: ArrayLike, label: str) -> None:
    """Reject a derived numerical quantity containing NaN or infinity.

    Args:
        values: Scalar or array produced by an analytical calculation.
        label: Human-readable calculation name used in errors.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Stops finite inputs that overflow during transformation from producing
        misleading exclusions, convergence claims, or fit diagnostics.

    Raises:
        ValueError: If any derived value is nonfinite.
    """
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"{label} must remain finite; numerical overflow occurred")


def _stable_column_means(matrix: FloatArray, label: str) -> FloatArray:
    """Calculate column means after scaling away extreme magnitudes.

    Args:
        matrix: Finite observation-by-variable matrix.
        label: Human-readable calculation name used in errors.

    Returns:
        One finite arithmetic mean per matrix column.

    Side effects:
        None.

    Statistical intent:
        Avoids overflow in a direct column sum when the true mean remains
        representable, including balanced values near float limits.
    """
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)

    column_maxima = np.max(np.abs(matrix), axis=0)
    scaled = np.zeros_like(matrix, dtype=float)
    nonzero = column_maxima > 0.0
    scaled[:, nonzero] = matrix[:, nonzero] / column_maxima[nonzero]
    scaled_means = np.clip(scaled.mean(axis=0), -1.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        means = column_maxima * scaled_means
    _require_finite_derived(means, label)
    return means


def _stable_column_root_mean_squares(
    matrix: FloatArray,
    label: str,
) -> FloatArray:
    """Calculate column population RMS values without squaring raw extremes.

    Args:
        matrix: Finite observation-by-variable matrix, commonly centered.
        label: Human-readable calculation name used in errors.

    Returns:
        One finite root-mean-square value per matrix column.

    Side effects:
        None.

    Statistical intent:
        Produces population standard deviations for centered candidates while
        avoiding overflow from directly forming their raw squares.
    """
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)

    column_maxima = np.max(np.abs(matrix), axis=0)
    scaled = np.zeros_like(matrix, dtype=float)
    nonzero = column_maxima > 0.0
    scaled[:, nonzero] = matrix[:, nonzero] / column_maxima[nonzero]
    scaled_mean_squares = np.mean(scaled * scaled, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        root_mean_squares = column_maxima * np.sqrt(scaled_mean_squares)
    _require_finite_derived(root_mean_squares, label)
    return root_mean_squares


def _stable_column_norms(matrix: FloatArray, label: str) -> FloatArray:
    """Calculate Euclidean column norms with scale-first arithmetic.

    Args:
        matrix: Finite observation-by-variable matrix.
        label: Human-readable calculation name used in errors.

    Returns:
        One finite Euclidean norm per matrix column.

    Side effects:
        None.

    Statistical intent:
        Supports scale-aware alias detection without overflowing on squared
        components that are individually finite.
    """
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)

    column_maxima = np.max(np.abs(matrix), axis=0)
    scaled = np.zeros_like(matrix, dtype=float)
    nonzero = column_maxima > 0.0
    scaled[:, nonzero] = matrix[:, nonzero] / column_maxima[nonzero]
    with np.errstate(over="ignore", invalid="ignore"):
        norms = column_maxima * np.sqrt(np.sum(scaled * scaled, axis=0))
    _require_finite_derived(norms, label)
    return norms


def _stable_mean_cross_products(
    matrix: FloatArray,
    vector: FloatArray,
    label: str,
) -> FloatArray:
    """Calculate columnwise mean cross-products using scaled operands.

    Args:
        matrix: Finite observation-by-variable matrix.
        vector: Finite outcome or residual vector with matching rows.
        label: Human-readable calculation name used in errors.

    Returns:
        One finite mean cross-product per matrix column.

    Side effects:
        None.

    Statistical intent:
        Stabilizes lambda and KKT correlations when finite outcomes are very
        large, while rejecting correlations whose true magnitude overflows.
    """
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)

    matrix_maxima = np.max(np.abs(matrix), axis=0)
    vector_maximum = float(np.max(np.abs(vector), initial=0.0))
    scaled_matrix = np.zeros_like(matrix, dtype=float)
    nonzero_columns = matrix_maxima > 0.0
    scaled_matrix[:, nonzero_columns] = (
        matrix[:, nonzero_columns] / matrix_maxima[nonzero_columns]
    )
    if vector_maximum > 0.0:
        scaled_vector = vector / vector_maximum
    else:
        scaled_vector = np.zeros_like(vector, dtype=float)

    scaled_products = np.mean(scaled_matrix * scaled_vector[:, None], axis=0)
    # Multiply by the smaller magnitude first so a small normalized correlation
    # can keep an otherwise representable final product away from overflow.
    smaller_scales = np.minimum(matrix_maxima, vector_maximum)
    larger_scales = np.maximum(matrix_maxima, vector_maximum)
    with np.errstate(over="ignore", invalid="ignore"):
        products = (scaled_products * smaller_scales) * larger_scales
    _require_finite_derived(products, label)
    return products


def _stable_mean_square(vector: FloatArray, label: str) -> float:
    """Calculate a vector mean square with scaled overflow checks.

    Args:
        vector: Finite residual values.
        label: Human-readable calculation name used in errors.

    Returns:
        The finite arithmetic mean of squared values.

    Side effects:
        None.

    Statistical intent:
        Computes the loss portion of the objective when representable and
        rejects a residual scale whose mean square exceeds float capacity.
    """
    maximum = float(np.max(np.abs(vector), initial=0.0))
    if maximum == 0.0:
        return 0.0
    scaled = vector / maximum
    scaled_mean_square = float(np.mean(scaled * scaled))
    with np.errstate(over="ignore", invalid="ignore"):
        mean_square = (scaled_mean_square * maximum) * maximum
    _require_finite_derived(mean_square, label)
    return float(mean_square)


def _stable_nonnegative_sum(values: FloatArray, label: str) -> float:
    """Sum finite nonnegative values with scale-first arithmetic.

    Args:
        values: Finite nonnegative terms to aggregate.
        label: Human-readable calculation name used in errors.

    Returns:
        The finite arithmetic sum of all terms.

    Side effects:
        None.

    Statistical intent:
        Accumulates weighted absolute coefficients without an intermediate
        overflow when the final L1 penalty remains representable.
    """
    if values.size == 0:
        return 0.0
    maximum = float(np.max(values, initial=0.0))
    if maximum == 0.0:
        return 0.0
    scaled_sum = float(np.sum(values / maximum))
    with np.errstate(over="ignore", invalid="ignore"):
        result = scaled_sum * maximum
    _require_finite_derived(result, label)
    return float(result)


def _validated_singular_values(matrix: FloatArray, label: str) -> FloatArray:
    """Calculate and validate singular values used for numerical rank.

    Args:
        matrix: Finite matrix whose rank is required.
        label: Human-readable matrix name used in errors.

    Returns:
        A finite vector of singular values in descending order.

    Side effects:
        None.

    Statistical intent:
        Prevents an overflowed SVD spectrum from producing a false rank claim.

    Raises:
        ValueError: If SVD fails or returns a nonfinite singular value.
    """
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{label} SVD did not converge") from error
    _require_finite_derived(singular_values, f"{label} SVD singular values")
    return singular_values


def _rank_from_singular_values(
    singular_values: FloatArray,
    shape: tuple[int, int],
    label: str,
) -> int:
    """Derive numerical rank from a validated singular spectrum.

    Args:
        singular_values: Finite descending singular values.
        shape: Row and column dimensions of the diagnosed matrix.
        label: Human-readable rank name used in errors.

    Returns:
        The standard floating-point numerical rank.

    Side effects:
        None.

    Statistical intent:
        Applies NumPy's conventional dimension-scaled machine-epsilon boundary
        while explicitly rejecting a nonfinite tolerance.
    """
    if singular_values.size == 0:
        return 0
    rank_tolerance = (
        max(shape) * np.finfo(float).eps * singular_values[0]
    )
    _require_finite_derived(rank_tolerance, f"{label} rank tolerance")
    return int(np.sum(singular_values > rank_tolerance))


def _validate_positive_control(value: Real, label: str) -> float:
    """Validate a strictly positive finite floating-point control.

    Args:
        value: Optimization tolerance or another positive control.
        label: Control name used in validation errors.

    Returns:
        The validated control as a Python float.

    Side effects:
        None.

    Statistical intent:
        Avoids undefined stopping rules and silent nonconvergence.

    Raises:
        TypeError: If the value is not a real scalar.
        ValueError: If the value is nonfinite or nonpositive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    validated = float(value)
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return validated


def _kkt_max_violation(
    residualized_candidates: FloatArray,
    residual: FloatArray,
    coefficients: FloatArray,
    lambda_value: float,
    penalty_weights: FloatArray,
    n_observations: int,
) -> float:
    """Calculate the largest coordinatewise LASSO KKT violation.

    Args:
        residualized_candidates: Scaled candidates orthogonal to base terms.
        residual: Current residualized outcome residual.
        coefficients: Current standardized candidate coefficients.
        lambda_value: Absolute LASSO penalty applied at this fit.
        penalty_weights: Positive coordinate-specific penalty multipliers.
        n_observations: Objective-function sample-size denominator.

    Returns:
        The maximum absolute active or positive inactive KKT violation.

    Side effects:
        None.

    Statistical intent:
        Verifies first-order optimality for both selected and zeroed candidates.
    """
    if coefficients.size == 0:
        return 0.0

    # Correlation equals the negative loss gradient with sign reversed; active
    # terms equal lambda times their signed weight, while inactive terms stay
    # inside the corresponding threshold interval.
    correlations = _stable_mean_cross_products(
        residualized_candidates,
        residual,
        "KKT correlations",
    )
    active = coefficients != 0.0
    violations = np.empty_like(coefficients)
    with np.errstate(over="ignore", invalid="ignore"):
        kkt_thresholds = lambda_value * penalty_weights
        violations[active] = np.abs(
            correlations[active]
            - kkt_thresholds[active] * np.sign(coefficients[active])
        )
        violations[~active] = np.maximum(
            np.abs(correlations[~active]) - kkt_thresholds[~active],
            0.0,
        )
    _require_finite_derived(kkt_thresholds, "KKT thresholds")
    _require_finite_derived(violations, "KKT violations")
    return float(np.max(violations, initial=0.0))


def _soft_threshold(value: float, threshold: float) -> float:
    """Apply the scalar LASSO soft-thresholding operator.

    Args:
        value: Unpenalized coordinate correlation.
        threshold: Nonnegative coordinate penalty.

    Returns:
        The signed correlation remaining beyond the penalty threshold.

    Side effects:
        None.

    Statistical intent:
        Produces exact zero coefficients when correlation does not justify a
        candidate's weighted L1 penalty.
    """
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_partially_penalized_lasso(
    outcome: ArrayLike,
    base_design: ArrayLike,
    candidate_design: ArrayLike,
    base_names: tuple[str, ...] | list[str],
    candidate_names: tuple[str, ...] | list[str],
    lambda_ratio: Real,
    penalty_weights: ArrayLike | None = None,
    tolerance: Real = 1e-10,
    max_iterations: int = 10_000,
) -> PartiallyPenalizedLassoResult:
    """Fit LASSO candidates while leaving supplied base terms unpenalized.

    Args:
        outcome: Finite training outcome with one value per observation.
        base_design: Unpenalized training design whose column span must contain
            a constant, commonly an intercept and established predictors.
        candidate_design: Candidate predictors eligible for L1 shrinkage.
        base_names: Names aligned with ``base_design`` columns.
        candidate_names: Names aligned with ``candidate_design`` columns.
        lambda_ratio: Penalty fraction in the closed interval zero to one,
            multiplied by the data-derived lambda-max.
        penalty_weights: Optional positive multiplier for each candidate;
            defaults to equal weights.
        tolerance: Maximum KKT violation accepted as convergence.
        max_iterations: Maximum deterministic coordinate-descent sweeps.

    Returns:
        An immutable fit containing raw coefficients, standardized candidate
        coefficients, training scales, exclusions, and convergence measures.

    Side effects:
        None; all inputs are inspected but never modified.

    Statistical intent:
        Solves ``RSS/(2N) + lambda * sum(weight * abs(theta))`` after QR-
        residualizing training-standardized candidates against an unpenalized
        full-rank base that spans a constant.  The constant is required because
        candidate centering must be absorbed when scaled coefficients are
        converted back to the raw prediction equation.  Selection is
        exploratory and is not a significance test.

    Raises:
        TypeError: If numerical inputs, names, or controls have invalid types.
        ValueError: If shapes, finite values, rank, names, or controls are
            invalid.
        RuntimeError: If coordinate descent does not satisfy the KKT tolerance.
    """
    response = _as_finite_vector(outcome, "outcome")
    base = _as_finite_matrix(base_design, "base_design")
    candidates = _as_finite_matrix(candidate_design, "candidate_design")
    n_observations = response.size

    if base.shape[0] != n_observations:
        raise ValueError("base_design rows must match outcome length")
    if candidates.shape[0] != n_observations:
        raise ValueError("candidate_design rows must match outcome length")

    validated_base_names = _validate_names(
        base_names,
        base.shape[1],
        "base_names",
    )
    validated_candidate_names = _validate_names(
        candidate_names,
        candidates.shape[1],
        "candidate_names",
    )
    if set(validated_base_names).intersection(validated_candidate_names):
        raise ValueError("base and candidate names must be unique together")

    if isinstance(lambda_ratio, (bool, np.bool_)) or not isinstance(
        lambda_ratio, Real
    ):
        raise TypeError("lambda_ratio must be a real scalar")
    validated_ratio = float(lambda_ratio)
    if not np.isfinite(validated_ratio) or not 0.0 <= validated_ratio <= 1.0:
        raise ValueError("lambda_ratio must be finite and between 0 and 1")

    validated_tolerance = _validate_positive_control(tolerance, "tolerance")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
    ):
        raise TypeError("max_iterations must be an integer")
    validated_max_iterations = int(max_iterations)
    if validated_max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    n_candidates = candidates.shape[1]
    if penalty_weights is None:
        weights = np.ones(n_candidates, dtype=float)
    else:
        weights = _as_finite_vector(penalty_weights, "penalty_weights")
        if weights.size != n_candidates:
            raise ValueError(
                "penalty_weights must contain one value per candidate"
            )
        if not (weights > 0.0).all():
            raise ValueError("penalty_weights must be strictly positive")

    if base.shape[1] == 0:
        raise ValueError("base_design must span a constant or intercept")

    base_singular_values = _validated_singular_values(base, "base_design")
    base_rank = _rank_from_singular_values(
        base_singular_values,
        base.shape,
        "base_design",
    )
    if base_rank != base.shape[1]:
        raise ValueError("base_design must have full column rank")

    # Compute the reduced QR basis once for both intercept-span validation and
    # later residualization.  Every factor is checked because a finite but
    # extreme base matrix can still overflow inside a factorization.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            base_q, base_r = np.linalg.qr(base, mode="reduced")
    except np.linalg.LinAlgError as error:
        raise ValueError("base_design QR factorization did not converge") from error
    _require_finite_derived(base_q, "base_design QR basis")
    _require_finite_derived(base_r, "base_design QR factor")

    constant = np.ones(n_observations, dtype=float)
    constant_residual = constant - base_q @ (base_q.T @ constant)
    _require_finite_derived(constant_residual, "base constant-span residual")
    constant_residual_norm = _stable_column_norms(
        constant_residual[:, None],
        "base constant-span norm",
    )[0]
    constant_norm = np.sqrt(float(n_observations))
    constant_span_boundary = (
        np.finfo(float).eps
        * 100.0
        * max(n_observations, base.shape[1])
        * constant_norm
    )
    _require_finite_derived(
        constant_span_boundary,
        "base constant-span boundary",
    )
    if constant_residual_norm > constant_span_boundary:
        raise ValueError("base_design must span a constant or intercept")

    # Candidate centering and population scaling use training rows only.  The
    # stored means/scales later prevent held-out data from redefining the fit.
    candidate_means = _stable_column_means(candidates, "candidate means")
    with np.errstate(over="ignore", invalid="ignore"):
        centered_candidates = candidates - candidate_means
    _require_finite_derived(centered_candidates, "centered candidates")
    candidate_scales = _stable_column_root_mean_squares(
        centered_candidates,
        "candidate population scales",
    )

    # A scale-aware numerical boundary treats exact and effectively constant
    # columns as unidentified before division by their population scale.
    maximum_absolute_values = np.max(np.abs(candidates), axis=0, initial=0.0)
    constant_boundaries = (
        np.finfo(float).eps
        * 100.0
        * np.maximum(1.0, maximum_absolute_values)
    )
    _require_finite_derived(constant_boundaries, "constant-column boundaries")
    is_constant = candidate_scales <= constant_boundaries
    standardized_candidates = np.zeros_like(candidates, dtype=float)
    varying_indices = np.flatnonzero(~is_constant)
    if varying_indices.size:
        with np.errstate(over="ignore", invalid="ignore"):
            standardized_candidates[:, varying_indices] = (
                centered_candidates[:, varying_indices]
                / candidate_scales[varying_indices]
            )
    _require_finite_derived(
        standardized_candidates,
        "standardized candidates",
    )

    # Reduced QR residualization removes all unpenalized base variation without
    # forming a projection matrix or inverse, which is both stable and compact.
    response_coordinate_means = _stable_mean_cross_products(
        base_q,
        response,
        "base-outcome QR correlations",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        response_coordinates = n_observations * response_coordinate_means
        residualized_outcome = response - base_q @ response_coordinates
        candidate_coordinates = base_q.T @ standardized_candidates
        residualized_candidates = standardized_candidates - base_q @ (
            candidate_coordinates
        )
    _require_finite_derived(response_coordinates, "base-outcome QR coordinates")
    _require_finite_derived(
        candidate_coordinates,
        "base-candidate QR coordinates",
    )
    _require_finite_derived(residualized_outcome, "residualized outcome")
    _require_finite_derived(
        residualized_candidates,
        "residualized candidates",
    )

    # A nonconstant candidate may still lie wholly in the base span.  Its tiny
    # QR residual identifies a base alias that has no separately estimable LASSO
    # effect and therefore must be excluded explicitly.
    residual_norms = _stable_column_norms(
        residualized_candidates,
        "residualized candidate norms",
    )
    standardized_norms = _stable_column_norms(
        standardized_candidates,
        "standardized candidate norms",
    )
    alias_boundaries = (
        np.finfo(float).eps
        * 100.0
        * max(n_observations, base.shape[1] + n_candidates, 1)
        * np.maximum(1.0, standardized_norms)
    )
    _require_finite_derived(alias_boundaries, "base-alias boundaries")
    is_base_alias = (~is_constant) & (residual_norms <= alias_boundaries)
    is_excluded = is_constant | is_base_alias
    retained_indices = np.flatnonzero(~is_excluded)
    excluded_names = tuple(
        validated_candidate_names[index]
        for index in np.flatnonzero(is_excluded)
    )

    retained_design = residualized_candidates[:, retained_indices]
    retained_weights = weights[retained_indices]
    retained_coefficients = np.zeros(retained_indices.size, dtype=float)

    # Lambda-max is the smallest weighted penalty at which every retained
    # candidate satisfies its inactive KKT condition at an exact zero.
    if retained_indices.size:
        initial_correlations = np.abs(
            _stable_mean_cross_products(
                retained_design,
                residualized_outcome,
                "lambda-max correlations",
            )
        )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_correlations = initial_correlations / retained_weights
        _require_finite_derived(
            weighted_correlations,
            "weighted lambda-max correlations",
        )
        lambda_max = float(np.max(weighted_correlations))
    else:
        lambda_max = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        lambda_value = validated_ratio * lambda_max
    _require_finite_derived(lambda_max, "lambda_max")
    _require_finite_derived(lambda_value, "lambda value")

    residual = residualized_outcome.copy()
    kkt_violation = _kkt_max_violation(
        retained_design,
        residual,
        retained_coefficients,
        lambda_value,
        retained_weights,
        n_observations,
    )
    iterations = 0

    # Cyclic updates use a fixed column order for reproducibility.  The residual
    # is updated in place after each soft-thresholded coordinate solution.
    while (
        kkt_violation > validated_tolerance
        and iterations < validated_max_iterations
    ):
        iterations += 1
        for coordinate in range(retained_coefficients.size):
            candidate_column = retained_design[:, coordinate]
            old_coefficient = retained_coefficients[coordinate]
            with np.errstate(over="ignore", invalid="ignore"):
                partial_residual = (
                    residual + candidate_column * old_coefficient
                )
            _require_finite_derived(
                partial_residual,
                "coordinate partial residual",
            )
            correlation = float(
                _stable_mean_cross_products(
                    candidate_column[:, None],
                    partial_residual,
                    "coordinate correlation",
                )[0]
            )
            curvature = _stable_mean_square(
                candidate_column,
                "coordinate curvature",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                coordinate_penalty = (
                    lambda_value * retained_weights[coordinate]
                )
            _require_finite_derived(
                coordinate_penalty,
                "coordinate penalty",
            )
            thresholded = _soft_threshold(
                correlation,
                coordinate_penalty,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                new_coefficient = thresholded / curvature
            _require_finite_derived(
                new_coefficient,
                "coordinate coefficient",
            )
            retained_coefficients[coordinate] = new_coefficient
            with np.errstate(over="ignore", invalid="ignore"):
                residual = (
                    partial_residual - candidate_column * new_coefficient
                )
            _require_finite_derived(residual, "coordinate residual")

        # Recompute rather than incrementally reuse the residual for a clean KKT
        # audit that is not inflated by many small floating-point updates.
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (
                residualized_outcome
                - retained_design @ retained_coefficients
            )
        _require_finite_derived(residual, "recomputed LASSO residual")
        kkt_violation = _kkt_max_violation(
            retained_design,
            residual,
            retained_coefficients,
            lambda_value,
            retained_weights,
            n_observations,
        )

    if kkt_violation > validated_tolerance:
        raise RuntimeError(
            "Coordinate descent did not converge within max_iterations; "
            f"KKT violation is {kkt_violation:.6g}"
        )

    standardized_coefficients = np.zeros(n_candidates, dtype=float)
    standardized_coefficients[retained_indices] = retained_coefficients

    # Back-transform standardized coefficients to original candidate units, then
    # refit the unpenalized base by stable least squares.  This stores the exact
    # raw-scale equation used for all future predictions.
    raw_candidate_coefficients = np.zeros(n_candidates, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_candidate_coefficients[retained_indices] = (
            retained_coefficients / candidate_scales[retained_indices]
        )
        raw_candidate_fitted = candidates @ raw_candidate_coefficients
        base_target = response - raw_candidate_fitted
    _require_finite_derived(
        raw_candidate_coefficients,
        "raw candidate coefficients",
    )
    _require_finite_derived(
        raw_candidate_fitted,
        "raw candidate fitted values",
    )
    _require_finite_derived(base_target, "base refit outcome")
    try:
        base_coefficients, _, _, _ = np.linalg.lstsq(
            base,
            base_target,
            rcond=None,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("base coefficient refit did not converge") from error
    _require_finite_derived(base_coefficients, "base coefficients")

    with np.errstate(over="ignore", invalid="ignore"):
        fitted_values = base @ base_coefficients + raw_candidate_fitted
        final_residual = response - fitted_values
        weighted_absolute_coefficients = (
            weights * np.abs(standardized_coefficients)
        )
    _require_finite_derived(fitted_values, "training fitted values")
    _require_finite_derived(final_residual, "training residuals")
    _require_finite_derived(
        weighted_absolute_coefficients,
        "weighted absolute coefficients",
    )
    loss_component = 0.5 * _stable_mean_square(
        final_residual,
        "objective residual mean square",
    )
    l1_sum = _stable_nonnegative_sum(
        weighted_absolute_coefficients,
        "weighted L1 sum",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        penalty_component = lambda_value * l1_sum
        objective = loss_component + penalty_component
    _require_finite_derived(penalty_component, "objective penalty")
    _require_finite_derived(objective, "objective")

    return PartiallyPenalizedLassoResult(
        base_names=validated_base_names,
        candidate_names=validated_candidate_names,
        base_coefficients=base_coefficients,
        candidate_coefficients=raw_candidate_coefficients,
        standardized_candidate_coefficients=standardized_coefficients,
        candidate_means=candidate_means,
        candidate_scales=candidate_scales,
        lambda_value=lambda_value,
        lambda_max=lambda_max,
        lambda_ratio=validated_ratio,
        objective=objective,
        kkt_violation=kkt_violation,
        iterations=iterations,
        excluded_candidate_names=excluded_names,
    )


def predict_partially_penalized_lasso(
    result: PartiallyPenalizedLassoResult,
    base_design: ArrayLike,
    candidate_design: ArrayLike,
) -> FloatArray:
    """Predict outcomes from a fitted raw-scale partially penalized equation.

    Args:
        result: Previously fitted partially penalized LASSO result.
        base_design: New unpenalized rows in fitted base-name order.
        candidate_design: New candidate rows in fitted candidate-name order.

    Returns:
        One finite prediction per supplied row.

    Side effects:
        None; prediction matrices and fitted state remain unchanged.

    Statistical intent:
        Uses stored raw coefficients so held-out rows cannot refit or leak their
        own candidate means and scales into prediction.

    Raises:
        TypeError: If ``result`` or matrix values have invalid types.
        ValueError: If prediction shapes or finite-value checks fail.
    """
    if not isinstance(result, PartiallyPenalizedLassoResult):
        raise TypeError("result must be a PartiallyPenalizedLassoResult")
    base = _as_finite_matrix(base_design, "base_design")
    candidates = _as_finite_matrix(candidate_design, "candidate_design")

    if base.shape[0] != candidates.shape[0]:
        raise ValueError("Prediction designs must contain the same row count")
    if base.shape[1] != len(result.base_names):
        raise ValueError("base_design columns must match fitted base_names")
    if candidates.shape[1] != len(result.candidate_names):
        raise ValueError(
            "candidate_design columns must match fitted candidate_names"
        )

    # Raw coefficients already include the training-scale back-transformation;
    # direct matrix multiplication is therefore the leakage-free prediction.
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = (
            base @ result.base_coefficients
            + candidates @ result.candidate_coefficients
        )
    _require_finite_derived(prediction, "prediction")
    return np.asarray(prediction, dtype=float)


def diagnose_collinearity(
    candidate_design: ArrayLike,
    names: tuple[str, ...] | list[str],
) -> CollinearityDiagnostic:
    """Report rank and shared null-space relations for named candidate columns.

    Args:
        candidate_design: Finite observation-by-candidate matrix to diagnose.
        names: Unique labels aligned with matrix columns.

    Returns:
        An immutable diagnostic with dimensions, rank, singular values, and one
        null-space row vector per estimated linear dependency.

    Side effects:
        None.

    Statistical intent:
        Identifies joint dependencies such as a module mean equaling the mean
        of four domain columns without assigning blame to a unique predictor.

    Raises:
        TypeError: If values or names have invalid types.
        ValueError: If the matrix is empty, nonfinite, mislabeled, or malformed.
    """
    design = _as_finite_matrix(candidate_design, "candidate_design")
    if design.shape[0] == 0:
        raise ValueError("candidate_design must contain at least one row")
    validated_names = _validate_names(names, design.shape[1], "names")

    # Full right singular vectors include null directions when columns exceed
    # rows; the same numerical threshold used by matrix-rank logic determines
    # which directions carry estimable variation.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            _, singular_values, right_vectors = np.linalg.svd(
                design,
                full_matrices=True,
            )
    except np.linalg.LinAlgError as error:
        raise ValueError("candidate_design SVD did not converge") from error
    _require_finite_derived(
        singular_values,
        "candidate_design SVD singular values",
    )
    _require_finite_derived(
        right_vectors,
        "candidate_design SVD right vectors",
    )
    rank = _rank_from_singular_values(
        singular_values,
        design.shape,
        "candidate_design",
    )
    null_space_vectors = right_vectors[rank:, :]
    is_rank_deficient = rank < design.shape[1]

    return CollinearityDiagnostic(
        column_names=validated_names,
        rank=rank,
        n_rows=design.shape[0],
        n_columns=design.shape[1],
        singular_values=singular_values,
        null_space_vectors=null_space_vectors,
        is_rank_deficient=is_rank_deficient,
    )
