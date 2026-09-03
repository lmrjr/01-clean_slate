"""Fit and predict from the two confirmed base statistical-model families.

The longitudinal implementation uses exact maximum-likelihood factorizations
for two measurement occasions.  The percentage-loss implementation uses
ordinary least squares with baseline weight and member type as predictors.
Only NumPy, pandas, and the Python standard library are required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

OutcomeScale = Literal["raw", "log"]
CovarianceStructure = Literal[
    "iid",
    "diagonal",
    "compound_symmetry",
    "unstructured",
]
FloatArray = NDArray[np.float64]

_TRAINING_COLUMNS = (
    "member_id",
    "member_type",
    "first_weight",
    "last_weight",
)
_PREDICTION_COLUMNS = ("member_type", "first_weight")
_COVARIANCE_PARAMETER_COUNTS = {
    "iid": 1,
    "diagonal": 2,
    "compound_symmetry": 2,
    "unstructured": 3,
}
_CONFIRMED_REFERENCE_MEMBER_TYPE = "Coaching Only"


def _as_read_only_float_array(values: FloatArray) -> FloatArray:
    """Copy numeric result values into an immutable NumPy array.

    Args:
        values: Numeric array produced during model fitting.

    Returns:
        A float copy whose NumPy write flag is disabled.

    Side effects:
        None; the supplied array remains unchanged.

    Statistical intent:
        Prevents fitted coefficients, standard errors, or covariance values
        from changing after their associated diagnostics have been calculated.
    """
    contiguous_values = np.ascontiguousarray(values, dtype=float)

    # A write-disabled owning ndarray can make itself writable again. Building
    # the result over immutable bytes makes the protection irreversible while
    # retaining the original shape and contiguous numerical representation.
    immutable_buffer = contiguous_values.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=float).reshape(
        contiguous_values.shape
    )


@dataclass(frozen=True, slots=True)
class ModelSchema:
    """Store the columns and factor levels required for valid prediction."""

    member_id_column: str
    member_type_column: str
    baseline_weight_column: str
    follow_up_weight_column: str
    member_type_levels: tuple[str, ...]
    reference_member_type: str


@dataclass(frozen=True, slots=True)
class LongitudinalGLSResult:
    """Store one fitted two-occasion marginal GLS model and its fit indices."""

    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    standard_errors: FloatArray
    covariance_matrix: FloatArray
    outcome_scale: OutcomeScale
    covariance_structure: CovarianceStructure
    include_time_by_member_type: bool
    schema: ModelSchema
    training_member_ids: tuple[str, ...]
    transformed_log_likelihood: float
    raw_scale_log_likelihood: float
    negative_two_log_likelihood: float
    aic: float
    bic: float
    n_members: int
    n_observations: int
    parameter_count: int
    design_rank: int

    def __post_init__(self) -> None:
        """Make numerical arrays and training provenance immutable.

        Args:
            self: Newly constructed longitudinal result.

        Returns:
            None.

        Side effects:
            Replaces array and member-ID fields with immutable result-owned
            values.

        Statistical intent:
            Keeps coefficients, covariance, fit indices, and the exact training
            sample internally consistent for downstream two-stage analyses.
        """
        training_member_ids = tuple(
            str(member_id).strip() for member_id in self.training_member_ids
        )
        if len(training_member_ids) != self.n_members:
            raise ValueError(
                "training_member_ids must contain one ID per fitted member"
            )
        if any(member_id == "" for member_id in training_member_ids):
            raise ValueError("training_member_ids cannot contain blank IDs")
        if len(set(training_member_ids)) != len(training_member_ids):
            raise ValueError("training_member_ids must be unique")

        # Frozen dataclasses require object-level assignment during post-init;
        # each replacement is an isolated copy so external views cannot mutate it.
        object.__setattr__(self, "training_member_ids", training_member_ids)
        object.__setattr__(
            self,
            "coefficients",
            _as_read_only_float_array(self.coefficients),
        )
        object.__setattr__(
            self,
            "standard_errors",
            _as_read_only_float_array(self.standard_errors),
        )
        object.__setattr__(
            self,
            "covariance_matrix",
            _as_read_only_float_array(self.covariance_matrix),
        )


@dataclass(frozen=True, slots=True)
class PercentageLossOLSResult:
    """Store one percentage-loss OLS fit and conventional diagnostics."""

    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    standard_errors: FloatArray
    schema: ModelSchema
    ml_residual_variance: float
    unbiased_residual_variance: float
    log_likelihood: float
    negative_two_log_likelihood: float
    aic: float
    bic: float
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    n_members: int
    parameter_count: int
    design_rank: int

    def __post_init__(self) -> None:
        """Make every stored numerical array immutable after construction.

        Args:
            self: Newly constructed percentage-loss result.

        Returns:
            None.

        Side effects:
            Replaces array fields on this frozen instance with read-only copies.

        Statistical intent:
            Prevents coefficient or SE mutation from invalidating stored OLS
            likelihoods, diagnostics, and predictions.
        """
        # Copy before disabling writes so neither caller-held arrays nor result
        # arrays can modify one another after the frozen result is constructed.
        object.__setattr__(
            self,
            "coefficients",
            _as_read_only_float_array(self.coefficients),
        )
        object.__setattr__(
            self,
            "standard_errors",
            _as_read_only_float_array(self.standard_errors),
        )


def _require_columns(
    table: pd.DataFrame,
    required_columns: tuple[str, ...],
    table_name: str,
) -> None:
    """Verify that a dataframe exposes every required analytical field.

    Args:
        table: Dataframe whose column contract is being checked.
        required_columns: Exact fields required by the calling operation.
        table_name: Human-readable label used in validation errors.

    Returns:
        None.

    Side effects:
        None; the dataframe is inspected but not mutated.

    Statistical intent:
        Prevents a missing predictor or outcome from silently changing a model.

    Raises:
        TypeError: If ``table`` is not a pandas dataframe.
        ValueError: If one or more required columns are absent.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame")

    missing_columns = [
        column for column in required_columns if column not in table.columns
    ]
    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(
            f"{table_name} is missing required columns: {missing_display}"
        )


def _prepare_training_members(members: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy one member row containing two endpoint weights.

    Args:
        members: Member-level rows with identifiers, type, and paired weights.

    Returns:
        A copied dataframe with normalized identifiers, types, and float weights.

    Side effects:
        None; the caller's dataframe is never modified.

    Statistical intent:
        Enforces exactly one independent pair of positive finite weights per
        member before likelihood or percentage-loss calculations begin.

    Raises:
        TypeError: If ``members`` is not a dataframe.
        ValueError: If required fields, identifiers, types, or weights are
            missing, duplicated, nonpositive, or nonfinite.
    """
    _require_columns(members, _TRAINING_COLUMNS, "Members")
    if members.empty:
        raise ValueError("Members must contain at least one weight pair")

    prepared = members.loc[:, list(_TRAINING_COLUMNS)].copy()

    # Normalize identity fields before uniqueness checks so whitespace cannot
    # create false members or factor levels.
    for column in ("member_id", "member_type"):
        normalized = prepared[column].astype("string").str.strip()
        invalid = normalized.isna() | normalized.eq("")
        if invalid.any():
            raise ValueError(f"Members {column} contains null or blank values")
        prepared[column] = normalized

    # Each row already represents the two required occasions; duplicate IDs
    # would therefore contribute more than one endpoint pair for a member.
    if prepared["member_id"].duplicated().any():
        raise ValueError(
            "Each member must contribute exactly two weights in one unique row"
        )

    # Convert both outcomes together and fail loudly on strings or objects that
    # cannot be interpreted as numeric weight measurements.
    try:
        prepared[["first_weight", "last_weight"]] = prepared[
            ["first_weight", "last_weight"]
        ].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Weights must be numeric") from error

    weights = prepared[["first_weight", "last_weight"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(weights).all() or not (weights > 0.0).all():
        raise ValueError(
            "Each member must have exactly two positive finite weights"
        )

    return prepared.reset_index(drop=True)


def _prepare_prediction_members(members: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the baseline fields required for prediction.

    Args:
        members: New member rows containing member type and baseline weight.

    Returns:
        A copied dataframe with normalized member types and float weights.

    Side effects:
        None; the caller's dataframe is never modified.

    Statistical intent:
        Keeps conditional predictions defined on positive finite baseline
        weights and a valid categorical predictor.

    Raises:
        TypeError: If ``members`` is not a dataframe.
        ValueError: If a required field or usable prediction value is absent.
    """
    _require_columns(members, _PREDICTION_COLUMNS, "Prediction members")
    prepared = members.loc[:, list(_PREDICTION_COLUMNS)].copy()

    # Blank categories have no fitted contrast and cannot be safely assigned to
    # the reference level during prediction.
    member_type = prepared["member_type"].astype("string").str.strip()
    invalid_type = member_type.isna() | member_type.eq("")
    if invalid_type.any():
        raise ValueError("Prediction member_type contains null or blank values")
    prepared["member_type"] = member_type

    # Conditional and percentage predictions both require a valid denominator
    # and, for log models, a defined logarithm.
    try:
        prepared["first_weight"] = pd.to_numeric(
            prepared["first_weight"], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Prediction first_weight must be numeric") from error
    baseline_weight = prepared["first_weight"].to_numpy(dtype=float)
    if (
        not np.isfinite(baseline_weight).all()
        or not (baseline_weight > 0.0).all()
    ):
        raise ValueError("Prediction first_weight must be positive and finite")

    return prepared.reset_index(drop=True)


def _factor_schema(
    member_types: pd.Series,
    reference_member_type: str | None,
) -> ModelSchema:
    """Define deterministic treatment coding for the member-type factor.

    Args:
        member_types: Normalized training categories.
        reference_member_type: Requested reference. When omitted, use
            ``Coaching Only`` if observed, otherwise the first sorted level.

    Returns:
        A schema containing sorted observed levels and the chosen reference.

    Side effects:
        None.

    Statistical intent:
        Makes coefficient interpretation reproducible across row orderings.

    Raises:
        ValueError: If an explicitly requested reference was not observed.
    """
    levels = tuple(sorted(str(value) for value in member_types.unique()))
    if reference_member_type is None:
        # The confirmed analysis has a clinical comparison reference. Generic
        # fixtures retain deterministic sorted treatment coding.
        reference = (
            _CONFIRMED_REFERENCE_MEMBER_TYPE
            if _CONFIRMED_REFERENCE_MEMBER_TYPE in levels
            else levels[0]
        )
    else:
        reference = str(reference_member_type).strip()
        if reference not in levels:
            raise ValueError(
                "reference_member_type must be an observed member type"
            )

    return ModelSchema(
        member_id_column="member_id",
        member_type_column="member_type",
        baseline_weight_column="first_weight",
        follow_up_weight_column="last_weight",
        member_type_levels=levels,
        reference_member_type=reference,
    )


def _factor_design(
    member_types: pd.Series,
    schema: ModelSchema,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Build an intercept-plus-treatment-contrast member-type matrix.

    Args:
        member_types: Normalized categories to encode.
        schema: Training levels and deterministic reference category.

    Returns:
        The numeric design matrix and names for its columns.

    Side effects:
        None.

    Statistical intent:
        Uses one omitted reference level to produce a full-rank categorical
        design whose coefficients have stable meanings.

    Raises:
        ValueError: If prediction data contains a level absent during fitting.
    """
    observed_levels = set(str(value) for value in member_types.unique())
    fitted_levels = set(schema.member_type_levels)
    unseen_levels = sorted(observed_levels.difference(fitted_levels))
    if unseen_levels:
        unseen_display = ", ".join(unseen_levels)
        raise ValueError(f"Prediction data contains unseen levels: {unseen_display}")

    contrast_levels = tuple(
        level
        for level in schema.member_type_levels
        if level != schema.reference_member_type
    )

    # Treatment coding places the intercept first and adds one indicator for
    # every sorted nonreference level, fixing coefficient order deterministically.
    member_type_values = member_types.astype(str).to_numpy()
    columns = [np.ones(len(member_types), dtype=float)]
    columns.extend(
        (member_type_values == level).astype(float) for level in contrast_levels
    )
    design = np.column_stack(columns).astype(float, copy=False)
    names = ("intercept",) + tuple(
        f"member_type[{level}]" for level in contrast_levels
    )
    return design, names


def _solve_least_squares(
    design: FloatArray,
    response: FloatArray,
    context: str,
) -> tuple[FloatArray, int]:
    """Solve a full-rank least-squares system and report its rank.

    Args:
        design: Two-dimensional numeric design matrix.
        response: One-dimensional numeric response vector.
        context: Model component named in an estimability error.

    Returns:
        The least-squares coefficient vector and verified design rank.

    Side effects:
        None.

    Statistical intent:
        Rejects unidentified fixed effects before they can contaminate
        covariance estimates, likelihoods, or predictions.

    Raises:
        ValueError: If inputs are nonfinite or the design is rank deficient.
    """
    if not np.isfinite(design).all() or not np.isfinite(response).all():
        raise ValueError(f"{context} contains nonfinite model values")

    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != design.shape[1]:
        raise ValueError(f"{context} design matrix is rank deficient")

    # NumPy's SVD-based solver avoids an explicit matrix inverse and is more
    # stable than solving normal equations for differently scaled predictors.
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients.astype(float, copy=False), design_rank


def _residual_variance(
    residual: FloatArray,
    divisor: int,
    context: str,
) -> float:
    """Calculate and validate a strictly positive residual variance.

    Args:
        residual: Residual vector from a fitted mean component.
        divisor: ML or degrees-of-freedom denominator specified by the caller.
        context: Variance component named in a validation error.

    Returns:
        The residual sum of squares divided by ``divisor``.

    Side effects:
        None.

    Statistical intent:
        Prevents a singular Gaussian likelihood or covariance estimate.

    Raises:
        ValueError: If the divisor or resulting variance is not positive and
            finite.
    """
    if divisor <= 0:
        raise ValueError(f"{context} variance has no positive residual degrees")
    numerical_zero = (
        np.finfo(float).eps * 100.0 * max(1.0, np.sqrt(len(residual)))
    )
    if np.linalg.norm(residual) <= numerical_zero:
        raise ValueError(f"{context} variance must be positive and finite")
    variance = float(np.dot(residual, residual) / divisor)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"{context} variance must be positive and finite")
    return variance


def _fit_unstructured_covariance(
    baseline: FloatArray,
    difference_residual: FloatArray,
    baseline_design: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit the exact two-occasion unstructured ML covariance factorization.

    Args:
        baseline: Transformed baseline outcomes.
        difference_residual: Residuals from the time-change model.
        baseline_design: Baseline member-type design matrix.

    Returns:
        Baseline fixed-effect coefficients and a 2-by-2 covariance matrix.

    Side effects:
        None.

    Statistical intent:
        Regresses baseline on the change residual to estimate its conditional
        coefficient, yielding all three unrestricted covariance parameters.
    """
    n_members = len(baseline)

    # Validate the change variance before adding its residual to the conditional
    # design so a degenerate covariance is reported as a variance failure rather
    # than as an incidental collinearity failure.
    difference_variance = _residual_variance(
        difference_residual,
        n_members,
        "Difference",
    )

    # Adding the change residual to the baseline design estimates kappa, the
    # conditional association that reconstructs covariance between occasions.
    conditional_design = np.column_stack(
        [baseline_design, difference_residual]
    )
    conditional_coefficients, _ = _solve_least_squares(
        conditional_design,
        baseline,
        "Unstructured conditional baseline",
    )
    baseline_coefficients = conditional_coefficients[:-1]
    kappa = float(conditional_coefficients[-1])
    baseline_residual = baseline - conditional_design @ conditional_coefficients

    # ML uses the independent-member count, not residual degrees of freedom,
    # for both factorized variance components.
    conditional_variance = _residual_variance(
        baseline_residual,
        n_members,
        "Conditional baseline",
    )

    # The triangular conditional representation maps kappa and the two
    # independent variances back to the unrestricted marginal covariance.
    variance_00 = conditional_variance + kappa**2 * difference_variance
    covariance_01 = variance_00 + kappa * difference_variance
    variance_11 = (
        variance_00 + 2.0 * kappa * difference_variance + difference_variance
    )
    covariance = np.array(
        [[variance_00, covariance_01], [covariance_01, variance_11]],
        dtype=float,
    )
    return baseline_coefficients, covariance


def _fit_compound_symmetry_covariance(
    baseline: FloatArray,
    follow_up: FloatArray,
    difference_mean: FloatArray,
    difference_residual: FloatArray,
    baseline_design: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit equal-variance compound symmetry from midpoint and difference.

    Args:
        baseline: Transformed baseline outcomes.
        follow_up: Transformed follow-up outcomes.
        difference_mean: Fitted mean change for every member.
        difference_residual: Residuals from the change model.
        baseline_design: Baseline member-type design matrix.

    Returns:
        Baseline fixed-effect coefficients and a 2-by-2 CS covariance matrix.

    Side effects:
        None.

    Statistical intent:
        Uses independent midpoint and difference components, whose variances
        exactly identify the common marginal variance and covariance.
    """
    n_members = len(baseline)

    # Under compound symmetry, midpoint and difference are independent. Their
    # separate OLS fits therefore give the exact ML mean and variance estimates.
    midpoint = 0.5 * (baseline + follow_up)
    midpoint_coefficients, _ = _solve_least_squares(
        baseline_design,
        midpoint,
        "Compound-symmetry midpoint",
    )
    midpoint_residual = midpoint - baseline_design @ midpoint_coefficients
    difference_variance = _residual_variance(
        difference_residual,
        n_members,
        "Difference",
    )
    midpoint_variance = _residual_variance(
        midpoint_residual,
        n_members,
        "Midpoint",
    )

    # Recover the baseline mean by subtracting half the fitted change from the
    # midpoint mean in the shared baseline-design column space.
    change_in_baseline_space, _ = _solve_least_squares(
        baseline_design,
        difference_mean,
        "Compound-symmetry mean reconstruction",
    )
    baseline_coefficients = (
        midpoint_coefficients - 0.5 * change_in_baseline_space
    )

    # Var(midpoint) and Var(difference) map directly to equal marginal
    # variances and a freely estimated common off-diagonal covariance.
    marginal_variance = midpoint_variance + 0.25 * difference_variance
    marginal_covariance = midpoint_variance - 0.25 * difference_variance
    covariance = np.array(
        [
            [marginal_variance, marginal_covariance],
            [marginal_covariance, marginal_variance],
        ],
        dtype=float,
    )
    return baseline_coefficients, covariance


def _profile_diagonal_at_kappa(
    kappa: float,
    baseline: FloatArray,
    difference_residual: FloatArray,
    baseline_design: FloatArray,
) -> tuple[float, FloatArray, float]:
    """Evaluate the diagonal-covariance profile likelihood at one kappa.

    Args:
        kappa: Conditional baseline-on-difference coefficient in ``(-1, 0)``.
        baseline: Transformed baseline outcomes.
        difference_residual: Residuals from the change model.
        baseline_design: Baseline member-type design matrix.

    Returns:
        Profile objective, baseline coefficients, and total difference variance.

    Side effects:
        None.

    Statistical intent:
        Profiles the two diagonal variances through kappa and their sum, using
        the exact conditional likelihood for two independent occasions.
    """
    n_members = len(baseline)
    q_value = -kappa * (1.0 + kappa)
    if q_value <= 0.0:
        raise ValueError("Diagonal covariance kappa must be between -1 and 0")

    # For fixed kappa, subtracting kappa times the difference residual converts
    # the conditional baseline equation into an ordinary baseline regression.
    conditional_target = baseline - kappa * difference_residual
    baseline_coefficients, _ = _solve_least_squares(
        baseline_design,
        conditional_target,
        "Diagonal conditional baseline",
    )
    conditional_residual = (
        baseline
        - baseline_design @ baseline_coefficients
        - kappa * difference_residual
    )

    # Profiling the common scale V=sigma_00+sigma_11 leaves a one-dimensional
    # objective in kappa; q is the conditional-variance fraction.
    total_variance = float(
        (
            np.dot(difference_residual, difference_residual)
            + np.dot(conditional_residual, conditional_residual) / q_value
        )
        / (2.0 * n_members)
    )
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("Diagonal residual variance must be positive and finite")
    objective = float(
        2.0 * n_members * np.log(total_variance)
        + n_members * np.log(q_value)
    )
    return objective, baseline_coefficients, total_variance


def _fit_diagonal_covariance(
    baseline: FloatArray,
    difference_residual: FloatArray,
    baseline_design: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit unequal independent occasion variances by deterministic profiling.

    Args:
        baseline: Transformed baseline outcomes.
        difference_residual: Residuals from the change model.
        baseline_design: Baseline member-type design matrix.

    Returns:
        Baseline fixed-effect coefficients and a diagonal covariance matrix.

    Side effects:
        None.

    Statistical intent:
        Uses golden-section minimization over kappa to estimate two positive
        marginal variances without SciPy or a nondeterministic optimizer.
    """
    # Keep the search strictly inside (-1, 0), where both diagonal variances
    # and the conditional-variance fraction are positive.
    boundary_offset = 1e-10
    left = -1.0 + boundary_offset
    right = -boundary_offset
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left_probe = right - golden_ratio * (right - left)
    right_probe = left + golden_ratio * (right - left)
    left_value = _profile_diagonal_at_kappa(
        left_probe,
        baseline,
        difference_residual,
        baseline_design,
    )[0]
    right_value = _profile_diagonal_at_kappa(
        right_probe,
        baseline,
        difference_residual,
        baseline_design,
    )[0]

    # A fixed tolerance and iteration ceiling make optimization reproducible
    # across runs while resolving covariance estimates beyond reporting precision.
    for _ in range(200):
        if right - left <= 1e-12:
            break
        if left_value < right_value:
            right = right_probe
            right_probe = left_probe
            right_value = left_value
            left_probe = right - golden_ratio * (right - left)
            left_value = _profile_diagonal_at_kappa(
                left_probe,
                baseline,
                difference_residual,
                baseline_design,
            )[0]
        else:
            left = left_probe
            left_probe = right_probe
            left_value = right_value
            right_probe = left + golden_ratio * (right - left)
            right_value = _profile_diagonal_at_kappa(
                right_probe,
                baseline,
                difference_residual,
                baseline_design,
            )[0]

    kappa = 0.5 * (left + right)
    interior_tolerance = np.sqrt(np.finfo(float).eps)
    if min(-kappa, 1.0 + kappa) <= interior_tolerance:
        raise ValueError(
            "Diagonal covariance profile reached a variance boundary; "
            "no finite interior MLE exists"
        )
    _, baseline_coefficients, total_variance = _profile_diagonal_at_kappa(
        kappa,
        baseline,
        difference_residual,
        baseline_design,
    )

    # The profile parameterization guarantees positive baseline and follow-up
    # variances and fixes their covariance at zero.
    covariance = np.diag(
        [-kappa * total_variance, (1.0 + kappa) * total_variance]
    ).astype(float)
    return baseline_coefficients, covariance


def _fit_iid_covariance(
    baseline: FloatArray,
    difference_residual: FloatArray,
    baseline_design: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit one common independent residual variance for both occasions.

    Args:
        baseline: Transformed baseline outcomes.
        difference_residual: Residuals from the change model.
        baseline_design: Baseline member-type design matrix.

    Returns:
        Baseline fixed-effect coefficients and an IID covariance matrix.

    Side effects:
        None.

    Statistical intent:
        Fixes kappa at -0.5, the value implied by equal independent marginal
        variances, then profiles their shared ML scale.
    """
    # Equal independent variances imply kappa=-sigma^2/(2*sigma^2)=-0.5.
    _, baseline_coefficients, total_variance = _profile_diagonal_at_kappa(
        -0.5,
        baseline,
        difference_residual,
        baseline_design,
    )
    common_variance = 0.5 * total_variance
    covariance = np.array(
        [[common_variance, 0.0], [0.0, common_variance]],
        dtype=float,
    )
    return baseline_coefficients, covariance


def _validate_covariance(covariance: FloatArray) -> None:
    """Verify a fitted two-occasion covariance is symmetric positive definite.

    Args:
        covariance: Candidate 2-by-2 residual covariance matrix.

    Returns:
        None.

    Side effects:
        None.

    Statistical intent:
        Ensures the Gaussian density, conditional prediction, and fixed-effect
        information matrix are all well defined.

    Raises:
        ValueError: If the matrix is malformed, nonfinite, nonsymmetric, or not
            positive definite.
    """
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("Residual covariance must be a finite 2-by-2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("Residual covariance must be symmetric")

    # Compare eigenvalues on the matrix's own spectral scale. Merely checking
    # positivity would accept optimizer guard values at a zero-variance boundary.
    eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_scale = float(np.max(eigenvalues))
    relative_tolerance = np.sqrt(np.finfo(float).eps)
    if (
        not np.isfinite(eigenvalues).all()
        or covariance_scale <= 0.0
        or float(np.min(eigenvalues))
        <= relative_tolerance * covariance_scale
    ):
        raise ValueError(
            "Residual covariance must be positive definite away from a "
            "variance boundary"
        )

    # Cholesky provides a second structural check without computing a matrix
    # inverse or relying on eigenvalue signs alone.
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Residual covariance must be positive definite"
        ) from error


def _longitudinal_standard_errors(
    baseline_design: FloatArray,
    difference_design: FloatArray,
    covariance: FloatArray,
) -> FloatArray:
    """Calculate model-based GLS standard errors for all fixed effects.

    Args:
        baseline_design: Baseline member-type design matrix.
        difference_design: Time or time-by-type change design matrix.
        covariance: Estimated 2-by-2 within-member covariance.

    Returns:
        Standard errors in baseline-then-time coefficient order.

    Side effects:
        None.

    Statistical intent:
        Uses the fitted marginal covariance in the GLS information matrix;
        these are model-based rather than sandwich standard errors.
    """
    n_members = len(baseline_design)
    n_baseline_coefficients = baseline_design.shape[1]
    n_difference_coefficients = difference_design.shape[1]
    n_coefficients = n_baseline_coefficients + n_difference_coefficients
    information = np.zeros((n_coefficients, n_coefficients), dtype=float)

    # Each independent member contributes a two-row fixed-effect block. Solving
    # Sigma times the block avoids forming either a large block matrix or an
    # explicit covariance inverse.
    for member_index in range(n_members):
        baseline_row = np.concatenate(
            [
                baseline_design[member_index],
                np.zeros(n_difference_coefficients, dtype=float),
            ]
        )
        follow_up_row = np.concatenate(
            [
                baseline_design[member_index],
                difference_design[member_index],
            ]
        )
        member_design = np.vstack([baseline_row, follow_up_row])
        weighted_design = np.linalg.solve(covariance, member_design)
        information += member_design.T @ weighted_design

    # Solving the information system yields its covariance action without
    # calling a matrix-inverse routine; diagonal square roots are the SEs.
    try:
        coefficient_covariance = np.linalg.solve(
            information,
            np.eye(n_coefficients, dtype=float),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("Longitudinal information matrix is rank deficient") from error
    diagonal = np.diag(coefficient_covariance)
    if not np.isfinite(diagonal).all() or (diagonal <= 0.0).any():
        raise ValueError("Longitudinal standard errors are not positive and finite")
    return np.sqrt(diagonal)


def _longitudinal_log_likelihood(
    residuals: FloatArray,
    covariance: FloatArray,
) -> float:
    """Evaluate the full bivariate Gaussian ML log likelihood.

    Args:
        residuals: One two-occasion residual row per independent member.
        covariance: Fitted 2-by-2 within-member covariance.

    Returns:
        Log likelihood on the modeled outcome scale, including constants.

    Side effects:
        None.

    Statistical intent:
        Uses the independent member as the likelihood unit and retains the
        normalizing constant required by AIC, BIC, and negative twice log L.
    """
    n_members = residuals.shape[0]
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("Residual covariance must have a positive log determinant")

    # Solve all member residual pairs in one operation, avoiding an explicit
    # inverse while computing the summed Mahalanobis quadratic form.
    weighted_residuals = np.linalg.solve(covariance, residuals.T).T
    quadratic_form = float(np.sum(residuals * weighted_residuals))
    return float(
        -n_members * np.log(2.0 * np.pi)
        - 0.5 * n_members * log_determinant
        - 0.5 * quadratic_form
    )


def fit_longitudinal_gls(
    members: pd.DataFrame,
    *,
    outcome_scale: OutcomeScale = "raw",
    covariance_structure: CovarianceStructure = "unstructured",
    include_time_by_member_type: bool = False,
    reference_member_type: str | None = None,
) -> LongitudinalGLSResult:
    """Fit a two-occasion marginal GLS model by maximum likelihood.

    Args:
        members: One row per member with ID, type, first weight, and last weight.
        outcome_scale: ``"raw"`` for weight or ``"log"`` for log weight.
        covariance_structure: One of IID, diagonal, compound symmetry, or
            unstructured residual covariance.
        include_time_by_member_type: Whether change has member-type contrasts;
            ``False`` fits the additive base model.
        reference_member_type: Optional observed reference category. Omission
            uses ``Coaching Only`` when observed, with sorted-level fallback.

    Returns:
        Coefficients, model-based standard errors, covariance, likelihood fit
        indices, sample sizes, parameter counts, rank, and prediction schema.

    Side effects:
        None; input data is copied and no files are written.

    Statistical intent:
        Fits the confirmed two-time-point marginal model with ML residual
        covariance. A log fit reports both Gaussian log-weight likelihood and
        the Jacobian-adjusted likelihood on the original weight scale.

    Raises:
        ValueError: If data, factor coding, options, design rank, variances, or
            covariance are invalid.
    """
    if outcome_scale not in ("raw", "log"):
        raise ValueError("outcome_scale must be 'raw' or 'log'")
    if covariance_structure not in _COVARIANCE_PARAMETER_COUNTS:
        raise ValueError(
            "covariance_structure must be 'iid', 'diagonal', "
            "'compound_symmetry', or 'unstructured'"
        )

    prepared = _prepare_training_members(members)
    schema = _factor_schema(
        prepared["member_type"],
        reference_member_type,
    )
    baseline_design, baseline_names = _factor_design(
        prepared["member_type"],
        schema,
    )

    # The base model has one common time coefficient. The sensitivity model
    # reuses the factor design to add one time contrast for each nonreference type.
    if include_time_by_member_type:
        difference_design = baseline_design.copy()
        difference_names = ("time",) + tuple(
            f"time:{name}" for name in baseline_names[1:]
        )
    else:
        difference_design = np.ones((len(prepared), 1), dtype=float)
        difference_names = ("time",)

    # Stack baseline and follow-up fixed-effect rows to verify estimability of
    # the complete mean model before estimating residual covariance.
    zero_difference = np.zeros_like(difference_design)
    baseline_rows = np.column_stack([baseline_design, zero_difference])
    follow_up_rows = np.column_stack([baseline_design, difference_design])
    full_design = np.vstack([baseline_rows, follow_up_rows])
    design_rank = int(np.linalg.matrix_rank(full_design))
    n_fixed_effects = full_design.shape[1]
    if design_rank != n_fixed_effects:
        raise ValueError("Longitudinal design matrix is rank deficient")

    raw_outcomes = prepared[["first_weight", "last_weight"]].to_numpy(
        dtype=float
    )

    # Log models transform both positive endpoints before fitting the Gaussian
    # mean and covariance; raw weights remain available for the Jacobian.
    if outcome_scale == "log":
        modeled_outcomes = np.log(raw_outcomes)
    else:
        modeled_outcomes = raw_outcomes.copy()
    baseline = modeled_outcomes[:, 0]
    follow_up = modeled_outcomes[:, 1]

    # The exact two-occasion likelihood first models the paired difference;
    # its residual drives every covariance-specific factorization below.
    difference = follow_up - baseline
    difference_coefficients, _ = _solve_least_squares(
        difference_design,
        difference,
        "Longitudinal difference",
    )
    difference_mean = difference_design @ difference_coefficients
    difference_residual = difference - difference_mean

    if covariance_structure == "unstructured":
        baseline_coefficients, covariance = _fit_unstructured_covariance(
            baseline,
            difference_residual,
            baseline_design,
        )
    elif covariance_structure == "compound_symmetry":
        baseline_coefficients, covariance = (
            _fit_compound_symmetry_covariance(
                baseline,
                follow_up,
                difference_mean,
                difference_residual,
                baseline_design,
            )
        )
    elif covariance_structure == "diagonal":
        baseline_coefficients, covariance = _fit_diagonal_covariance(
            baseline,
            difference_residual,
            baseline_design,
        )
    else:
        baseline_coefficients, covariance = _fit_iid_covariance(
            baseline,
            difference_residual,
            baseline_design,
        )

    _validate_covariance(covariance)

    # Combine baseline and change means to evaluate every residual under the
    # same joint bivariate density, regardless of covariance parameterization.
    baseline_mean = baseline_design @ baseline_coefficients
    follow_up_mean = baseline_mean + difference_mean
    residuals = np.column_stack(
        [baseline - baseline_mean, follow_up - follow_up_mean]
    )
    transformed_log_likelihood = _longitudinal_log_likelihood(
        residuals,
        covariance,
    )

    # A lognormal density on raw weight includes one 1/weight Jacobian for each
    # endpoint. Raw models need no change of variables.
    if outcome_scale == "log":
        raw_scale_log_likelihood = float(
            transformed_log_likelihood - np.log(raw_outcomes).sum()
        )
    else:
        raw_scale_log_likelihood = transformed_log_likelihood

    coefficient_names = baseline_names + difference_names
    coefficients = np.concatenate(
        [baseline_coefficients, difference_coefficients]
    ).astype(float, copy=False)
    standard_errors = _longitudinal_standard_errors(
        baseline_design,
        difference_design,
        covariance,
    )

    # Likelihood parameter counts include fixed effects and the covariance
    # family; BIC uses independent members rather than the two stacked rows.
    n_members = len(prepared)
    covariance_parameters = _COVARIANCE_PARAMETER_COUNTS[covariance_structure]
    parameter_count = n_fixed_effects + covariance_parameters
    negative_two_log_likelihood = -2.0 * raw_scale_log_likelihood
    aic = negative_two_log_likelihood + 2.0 * parameter_count
    bic = (
        negative_two_log_likelihood
        + np.log(n_members) * parameter_count
    )

    return LongitudinalGLSResult(
        coefficient_names=coefficient_names,
        coefficients=coefficients,
        standard_errors=standard_errors,
        covariance_matrix=covariance,
        outcome_scale=outcome_scale,
        covariance_structure=covariance_structure,
        include_time_by_member_type=include_time_by_member_type,
        schema=schema,
        training_member_ids=tuple(prepared["member_id"].astype(str)),
        transformed_log_likelihood=float(transformed_log_likelihood),
        raw_scale_log_likelihood=float(raw_scale_log_likelihood),
        negative_two_log_likelihood=float(negative_two_log_likelihood),
        aic=float(aic),
        bic=float(bic),
        n_members=n_members,
        n_observations=2 * n_members,
        parameter_count=parameter_count,
        design_rank=design_rank,
    )


def predict_last_weight_longitudinal(
    result: LongitudinalGLSResult,
    members: pd.DataFrame,
) -> FloatArray:
    """Predict conditional mean last weight from observed baseline weight.

    Args:
        result: Fitted result returned by ``fit_longitudinal_gls``.
        members: New rows with member type and positive baseline weight.

    Returns:
        One conditional expected raw last weight per input row.

    Side effects:
        None; input data and the fitted result are not mutated.

    Statistical intent:
        Conditions the joint two-occasion Gaussian model on baseline. Log fits
        return the lognormal conditional mean using the variance correction.

    Raises:
        TypeError: If ``result`` is not a longitudinal result.
        ValueError: If prediction data is invalid or contains unseen levels.
    """
    if not isinstance(result, LongitudinalGLSResult):
        raise TypeError("result must be a LongitudinalGLSResult")
    prepared = _prepare_prediction_members(members)
    baseline_design, _ = _factor_design(
        prepared["member_type"],
        result.schema,
    )

    n_baseline_coefficients = baseline_design.shape[1]
    baseline_coefficients = result.coefficients[:n_baseline_coefficients]
    difference_coefficients = result.coefficients[n_baseline_coefficients:]

    # Recreate the fitted additive or interaction change design using the exact
    # training schema so predictions align with coefficient ordering.
    if result.include_time_by_member_type:
        difference_design = baseline_design
    else:
        difference_design = np.ones((len(prepared), 1), dtype=float)
    baseline_mean = baseline_design @ baseline_coefficients
    follow_up_mean = (
        baseline_mean + difference_design @ difference_coefficients
    )

    # The bivariate-normal conditional mean adjusts the marginal follow-up mean
    # by the observed baseline deviation and fitted within-member covariance.
    covariance = result.covariance_matrix
    conditional_slope = covariance[1, 0] / covariance[0, 0]
    conditional_variance = (
        covariance[1, 1]
        - covariance[1, 0] * covariance[0, 1] / covariance[0, 0]
    )
    baseline_raw = prepared["first_weight"].to_numpy(dtype=float)
    if result.outcome_scale == "log":
        baseline_modeled = np.log(baseline_raw)
    else:
        baseline_modeled = baseline_raw
    conditional_mean = (
        follow_up_mean
        + conditional_slope * (baseline_modeled - baseline_mean)
    )

    # For log models, E[weight | baseline] is exp(mu + variance/2), not the
    # median exp(mu). Raw models are already on the requested prediction scale.
    if result.outcome_scale == "log":
        return np.exp(conditional_mean + 0.5 * conditional_variance)
    return conditional_mean.astype(float, copy=False)


def fit_percentage_loss_ols(
    members: pd.DataFrame,
    *,
    reference_member_type: str | None = None,
) -> PercentageLossOLSResult:
    """Fit percentage loss on baseline weight and member type using OLS.

    Args:
        members: One row per member with ID, type, first weight, and last weight.
        reference_member_type: Optional observed reference category. Omission
            uses ``Coaching Only`` when observed, with sorted-level fallback.

    Returns:
        Coefficients, conventional standard errors, ML likelihood variance,
        unbiased SE variance, fit metrics, rank, and prediction schema.

    Side effects:
        None; input data is copied and no files are written.

    Statistical intent:
        Models ``100 * (first-last) / first`` while keeping baseline weight and
        member type as unpenalized base predictors for later extensions.

    Raises:
        ValueError: If input data, design rank, outcome variation, or residual
            degrees of freedom are invalid.
    """
    prepared = _prepare_training_members(members)
    schema = _factor_schema(
        prepared["member_type"],
        reference_member_type,
    )
    factor_design, factor_names = _factor_design(
        prepared["member_type"],
        schema,
    )
    first_weight = prepared["first_weight"].to_numpy(dtype=float)
    last_weight = prepared["last_weight"].to_numpy(dtype=float)

    # Preserve the confirmed signed percentage-loss definition: positive values
    # mean weight reduction and negative values mean weight gain.
    percentage_loss = 100.0 * (first_weight - last_weight) / first_weight

    # Place baseline weight immediately after the intercept, followed by sorted
    # treatment contrasts, to give a stable and interpretable coefficient schema.
    design = np.column_stack(
        [factor_design[:, 0], first_weight, factor_design[:, 1:]]
    )
    coefficient_names = (
        "intercept",
        "first_weight",
        *factor_names[1:],
    )
    coefficients, design_rank = _solve_least_squares(
        design,
        percentage_loss,
        "Percentage-loss OLS",
    )
    fitted = design @ coefficients
    residual = percentage_loss - fitted

    n_members = len(prepared)
    n_fixed_effects = design.shape[1]
    residual_degrees = n_members - n_fixed_effects
    if residual_degrees <= 0:
        raise ValueError(
            "Percentage-loss OLS requires positive residual degrees of freedom"
        )
    residual_sum_squares = float(np.dot(residual, residual))
    residual_norm = float(np.linalg.norm(residual))
    outcome_norm = float(np.linalg.norm(percentage_loss))
    relative_zero_tolerance = np.sqrt(np.finfo(float).eps)
    if (
        not np.isfinite(residual_norm)
        or not np.isfinite(outcome_norm)
        or residual_norm <= relative_zero_tolerance * outcome_norm
    ):
        raise ValueError(
            "Percentage-loss residual variance is at a numerical boundary"
        )
    if not np.isfinite(residual_sum_squares) or residual_sum_squares <= 0.0:
        raise ValueError("Percentage-loss residual variance must be positive")

    # ML variance powers the Gaussian likelihood, whereas conventional OLS
    # coefficient standard errors use the unbiased residual variance.
    ml_residual_variance = residual_sum_squares / n_members
    unbiased_residual_variance = residual_sum_squares / residual_degrees
    # A reduced QR factorization gives X'X = R'R without explicitly forming the
    # ill-conditioned normal equations. Two solves recover R^-1 R^-T.
    _, triangular_factor = np.linalg.qr(design, mode="reduced")
    try:
        inverse_transpose = np.linalg.solve(
            triangular_factor.T,
            np.eye(n_fixed_effects, dtype=float),
        )
        coefficient_covariance = unbiased_residual_variance * np.linalg.solve(
            triangular_factor,
            inverse_transpose,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Percentage-loss OLS design matrix is rank deficient"
        ) from error
    covariance_diagonal = np.diag(coefficient_covariance)
    if (
        not np.isfinite(covariance_diagonal).all()
        or (covariance_diagonal <= 0.0).any()
    ):
        raise ValueError("Percentage-loss standard errors are not positive and finite")
    standard_errors = np.sqrt(covariance_diagonal)

    # At the ML variance estimate, the Gaussian residual quadratic equals n;
    # retaining the full constant makes information criteria reproducible.
    log_likelihood = float(
        -0.5
        * n_members
        * (
            np.log(2.0 * np.pi)
            + 1.0
            + np.log(ml_residual_variance)
        )
    )
    parameter_count = n_fixed_effects + 1
    negative_two_log_likelihood = -2.0 * log_likelihood
    aic = negative_two_log_likelihood + 2.0 * parameter_count
    bic = (
        negative_two_log_likelihood
        + np.log(n_members) * parameter_count
    )

    # R-squared uses variability around the observed outcome mean; adjusted R2
    # applies the conventional fixed-effect degrees-of-freedom correction.
    centered_outcome = percentage_loss - percentage_loss.mean()
    total_sum_squares = float(np.dot(centered_outcome, centered_outcome))
    if not np.isfinite(total_sum_squares) or total_sum_squares <= 0.0:
        raise ValueError("Percentage-loss outcome must have positive variation")
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = 1.0 - (
        (1.0 - r_squared)
        * (n_members - 1)
        / residual_degrees
    )
    rmse = float(np.sqrt(ml_residual_variance))

    return PercentageLossOLSResult(
        coefficient_names=coefficient_names,
        coefficients=coefficients,
        standard_errors=standard_errors,
        schema=schema,
        ml_residual_variance=float(ml_residual_variance),
        unbiased_residual_variance=float(unbiased_residual_variance),
        log_likelihood=log_likelihood,
        negative_two_log_likelihood=float(negative_two_log_likelihood),
        aic=float(aic),
        bic=float(bic),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        rmse=rmse,
        n_members=n_members,
        parameter_count=parameter_count,
        design_rank=design_rank,
    )


def predict_last_weight_percentage_loss(
    result: PercentageLossOLSResult,
    members: pd.DataFrame,
) -> FloatArray:
    """Predict last weight by converting fitted percentage loss to weight.

    Args:
        result: Fitted result returned by ``fit_percentage_loss_ols``.
        members: New rows with member type and positive baseline weight.

    Returns:
        One unbounded predicted last weight per input row.

    Side effects:
        None; input data and the fitted result are not mutated.

    Statistical intent:
        Applies the algebraic inverse of percentage loss without clipping,
        preserving the ordinary regression's predictions and extrapolations.

    Raises:
        TypeError: If ``result`` is not a percentage-loss OLS result.
        ValueError: If prediction data is invalid or contains unseen levels.
    """
    if not isinstance(result, PercentageLossOLSResult):
        raise TypeError("result must be a PercentageLossOLSResult")
    prepared = _prepare_prediction_members(members)
    factor_design, _ = _factor_design(
        prepared["member_type"],
        result.schema,
    )
    first_weight = prepared["first_weight"].to_numpy(dtype=float)

    # Recreate the fitted OLS column order before multiplying by stored
    # coefficients: intercept, baseline weight, then treatment contrasts.
    design = np.column_stack(
        [factor_design[:, 0], first_weight, factor_design[:, 1:]]
    )
    predicted_percentage_loss = design @ result.coefficients

    # This is the exact inverse of 100*(first-last)/first. No clipping is
    # applied because it would silently change the fitted model's estimand.
    return first_weight * (1.0 - predicted_percentage_loss / 100.0)
