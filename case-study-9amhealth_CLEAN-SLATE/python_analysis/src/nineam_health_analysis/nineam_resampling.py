"""Build penalized designs and run leakage-safe member-level resampling.

This module keeps each member intact in every split, constructs separate
module-mean and module-domain candidate sets, tunes partially penalized LASSO
models inside grouped folds, and compares the two base model families on a
common raw follow-up-weight target. Only NumPy, pandas, and the standard
library are required.

For a prewhitened longitudinal design, grouped lambda CV re-estimates LASSO
scaling and lambda-max inside each fold but cannot re-estimate the covariance
from matrix inputs alone. Such a run is retrospective two-stage exploration,
not an unbiased nested-CV performance assessment. Final predictive evaluation
must use an untouched outer split with every preprocessing fit on training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from nineam_health_analysis.nineam_penalized_models import (
    fit_partially_penalized_lasso,
    predict_partially_penalized_lasso,
)
from nineam_health_analysis.nineam_statistical_models import (
    LongitudinalGLSResult,
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
    predict_last_weight_longitudinal,
    predict_last_weight_percentage_loss,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]
ModuleSpecification = Literal["mean", "domains"]
SelectionRule = Literal["minimum", "one_standard_error"]

_ENGAGEMENT_CANDIDATES = (
    "engagement_volume_repeatable",
    "engagement_volume_repeatable_rate",
    "engagement_breadth",
    "tenure_days",
)
_MODULE_MEAN_CANDIDATES = ("module_mean",)
_MODULE_DOMAIN_CANDIDATES = (
    "module_core",
    "module_mindset",
    "module_nutrition",
    "module_physical_activity",
)
_MEMBER_COLUMNS = (
    "member_id",
    "member_type",
    "first_weight",
    "last_weight",
    "sex",
)
_CONFIRMED_REFERENCE_MEMBER_TYPE = "Coaching Only"


def _immutable_array(values: ArrayLike, dtype: np.dtype | type) -> np.ndarray:
    """Copy an array into storage whose write flag cannot be re-enabled.

    Args:
        values: Values to isolate from caller-owned storage.
        dtype: NumPy dtype required by the result field.

    Returns:
        A contiguous array backed by immutable bytes.

    Side effects:
        None; the supplied values remain unchanged.

    Statistical intent:
        Prevents resampling assignments, designs, or diagnostics from drifting
        after related scores and selection summaries have been calculated.
    """
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    immutable_buffer = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _immutable_strings(values: ArrayLike) -> StringArray:
    """Normalize labels and return them in strictly immutable Unicode storage.

    Args:
        values: One-dimensional identifiers or stratum labels.

    Returns:
        A read-only Unicode array containing stripped string values.

    Side effects:
        None.

    Statistical intent:
        Gives grouping and stratification stable labels independent of mutable
        pandas or object-array storage.
    """
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional")
    if pd.isna(raw).any():
        raise ValueError("Label arrays cannot contain missing values")
    normalized = [str(value).strip() for value in raw]
    if any(value == "" for value in normalized):
        raise ValueError("Label arrays cannot contain blank values")
    maximum_length = max((len(value) for value in normalized), default=1)
    return _immutable_array(normalized, np.dtype(f"<U{maximum_length}"))


def _validate_names(
    names: tuple[str, ...] | list[str],
    expected_count: int,
    label: str,
) -> tuple[str, ...]:
    """Validate a unique predictor-name sequence against a matrix width.

    Args:
        names: Predictor labels in design-column order.
        expected_count: Number of matrix columns requiring labels.
        label: Human-readable field name for validation errors.

    Returns:
        A normalized immutable tuple of names.

    Side effects:
        None.

    Statistical intent:
        Keeps coefficients and selected-variable summaries unambiguously
        aligned with the columns used for estimation.
    """
    if not isinstance(names, (tuple, list)):
        raise TypeError(f"{label} must be a tuple or list")
    normalized = tuple(str(name).strip() for name in names)
    if len(normalized) != expected_count:
        raise ValueError(f"{label} must contain one name per design column")
    if any(name == "" for name in normalized):
        raise ValueError(f"{label} cannot contain blank names")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain unique names")
    return normalized


def _validate_positive_integer(value: object, label: str) -> int:
    """Return a positive Python integer or raise a focused validation error.

    Args:
        value: Candidate count or seed-like control.
        label: Human-readable control name.

    Returns:
        The validated positive integer.

    Side effects:
        None.

    Statistical intent:
        Prevents degenerate fold, repeat, and resampling counts.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    validated = int(value)
    if validated <= 0:
        raise ValueError(f"{label} must be positive")
    return validated


def _validate_random_state(random_state: object) -> int:
    """Validate an explicit nonnegative seed for deterministic randomization.

    Args:
        random_state: Seed supplied to a NumPy random generator.

    Returns:
        A nonnegative Python integer.

    Side effects:
        None.

    Statistical intent:
        Makes every fold and stability sample exactly reproducible.
    """
    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
    ):
        raise TypeError("random_state must be an integer")
    validated = int(random_state)
    if validated < 0:
        raise ValueError("random_state must be nonnegative")
    return validated


@dataclass(frozen=True, slots=True)
class GroupedFold:
    """Store one immutable train/test row split for a repeated grouped fold."""

    repeat: int
    fold: int
    train_indices: IntArray
    test_indices: IntArray

    def __post_init__(self) -> None:
        """Validate fold metadata and isolate both row-index arrays.

        Args:
            self: Newly constructed grouped fold.

        Returns:
            None.

        Side effects:
            Replaces index fields with immutable result-owned copies.

        Statistical intent:
            Protects the exact sample membership underlying held-out scores.
        """
        for field_name in ("repeat", "fold"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                Integral,
            ):
                raise TypeError(f"{field_name} must be an integer")
            if int(value) < 0:
                raise ValueError("repeat and fold must be nonnegative")
            object.__setattr__(self, field_name, int(value))
        train = _validated_fold_indices(self.train_indices, "train_indices")
        test = _validated_fold_indices(self.test_indices, "test_indices")
        if train.size == 0 or test.size == 0:
            raise ValueError("Fold train and test indices cannot be empty")
        if np.intersect1d(train, test).size:
            raise ValueError("Fold train and test indices must be disjoint")
        object.__setattr__(self, "train_indices", _immutable_array(train, np.int64))
        object.__setattr__(self, "test_indices", _immutable_array(test, np.int64))


def _validated_fold_indices(values: ArrayLike, label: str) -> IntArray:
    """Validate row indices without lossy or coercive integer conversion.

    Args:
        values: Candidate one-dimensional row-index sequence.
        label: Field name used in focused validation errors.

    Returns:
        A unique nonnegative signed-64-bit integer array.

    Side effects:
        None.

    Statistical intent:
        Prevents fractional truncation, unsafe overflow, or duplicate row
        weighting from changing a grouped train/test sample.
    """
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("Fold indices must be one-dimensional")
    maximum = np.iinfo(np.int64).max
    normalized: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            Integral,
        ):
            raise TypeError(f"{label} must contain only integer values")
        integer = int(value)
        if integer < 0:
            raise ValueError("Fold indices must be nonnegative")
        if integer > maximum:
            raise ValueError(f"{label} contains an index outside int64 range")
        normalized.append(integer)
    indices = np.asarray(normalized, dtype=np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{label} cannot contain duplicate indices")
    return indices


@dataclass(frozen=True, slots=True)
class PenalizedDesign:
    """Store aligned outcome, base, candidate, group, and stratum arrays."""

    outcome: FloatArray
    base_design: FloatArray
    candidate_design: FloatArray
    base_names: tuple[str, ...]
    candidate_names: tuple[str, ...]
    group_ids: StringArray
    strata: StringArray

    def __post_init__(self) -> None:
        """Validate dimensions and copy every array into immutable storage.

        Args:
            self: Newly constructed penalized design.

        Returns:
            None.

        Side effects:
            Replaces all array fields and normalized name tuples on this frozen
            instance with isolated immutable values.

        Statistical intent:
            Locks outcome/predictor alignment and member grouping before any
            fold-specific scaling or penalization begins.
        """
        outcome = np.asarray(self.outcome, dtype=float)
        base = np.asarray(self.base_design, dtype=float)
        candidates = np.asarray(self.candidate_design, dtype=float)
        if outcome.ndim != 1:
            raise ValueError("outcome must be one-dimensional")
        if base.ndim != 2 or candidates.ndim != 2:
            raise ValueError("Design matrices must be two-dimensional")
        if outcome.size == 0:
            raise ValueError("Penalized designs require at least one row")
        if base.shape[0] != outcome.size or candidates.shape[0] != outcome.size:
            raise ValueError("Outcome and design matrices must have equal rows")
        if base.shape[1] == 0 or candidates.shape[1] == 0:
            raise ValueError("Base and candidate designs require columns")
        if not (
            np.isfinite(outcome).all()
            and np.isfinite(base).all()
            and np.isfinite(candidates).all()
        ):
            raise ValueError("Penalized design values must be finite")
        base_names = _validate_names(self.base_names, base.shape[1], "base_names")
        candidate_names = _validate_names(
            self.candidate_names,
            candidates.shape[1],
            "candidate_names",
        )
        if set(base_names).intersection(candidate_names):
            raise ValueError("Base and candidate names must be unique together")
        group_ids = _immutable_strings(self.group_ids)
        strata = _immutable_strings(self.strata)
        if group_ids.size != outcome.size or strata.size != outcome.size:
            raise ValueError("group_ids and strata must align with design rows")

        # A member must carry one stratum across all repeated rows; otherwise a
        # grouped stratified split has no coherent assignment for that member.
        grouping = pd.DataFrame({"group": group_ids, "stratum": strata})
        if grouping.groupby("group", sort=False)["stratum"].nunique().gt(1).any():
            raise ValueError("strata cannot vary within group")

        object.__setattr__(self, "outcome", _immutable_array(outcome, float))
        object.__setattr__(self, "base_design", _immutable_array(base, float))
        object.__setattr__(
            self,
            "candidate_design",
            _immutable_array(candidates, float),
        )
        object.__setattr__(self, "base_names", base_names)
        object.__setattr__(self, "candidate_names", candidate_names)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "strata", strata)


@dataclass(frozen=True, slots=True)
class LambdaCVResult:
    """Store tuning-only grouped-CV scores and one selected lambda ratio."""

    lambda_ratios: FloatArray
    fold_scores: FloatArray
    mean_scores: FloatArray
    standard_errors: FloatArray
    fold_lambda_maxima: FloatArray
    selected_lambda_ratio: float
    selected_index: int
    selection_rule: SelectionRule

    def __post_init__(self) -> None:
        """Copy all tuning arrays into immutable result-owned storage.

        Args:
            self: Newly constructed lambda cross-validation result.

        Returns:
            None.

        Side effects:
            Replaces numerical fields with strictly immutable copies.

        Statistical intent:
            Preserves the complete held-out evidence supporting a tuning choice
            without attaching naive inferential p-values.
        """
        ratios = np.asarray(self.lambda_ratios, dtype=float)
        scores = np.asarray(self.fold_scores, dtype=float)
        means = np.asarray(self.mean_scores, dtype=float)
        standard_errors = np.asarray(self.standard_errors, dtype=float)
        maxima = np.asarray(self.fold_lambda_maxima, dtype=float)
        if ratios.ndim != 1 or scores.ndim != 2:
            raise ValueError("Lambda ratios and fold scores have invalid dimensions")
        if ratios.size == 0 or scores.shape[0] == 0:
            raise ValueError("Lambda results require ratios and held-out folds")
        if scores.shape[1] != ratios.size:
            raise ValueError("Fold-score columns must match lambda ratios")
        if means.shape != ratios.shape or standard_errors.shape != ratios.shape:
            raise ValueError("Lambda score summaries must match lambda ratios")
        if maxima.shape != (scores.shape[0],):
            raise ValueError("fold_lambda_maxima must contain one value per fold")
        if (
            not np.isfinite(ratios).all()
            or ((ratios <= 0.0) | (ratios > 1.0)).any()
            or np.unique(ratios).size != ratios.size
        ):
            raise ValueError("lambda_ratios must be unique finite values in (0, 1]")
        if not np.isfinite(scores).all() or (scores < 0.0).any():
            raise ValueError("fold_scores must be finite nonnegative losses")
        if not np.isfinite(maxima).all() or (maxima < 0.0).any():
            raise ValueError("fold_lambda_maxima must be nonnegative and finite")
        if (
            not np.isfinite(standard_errors).all()
            or (standard_errors < 0.0).any()
        ):
            raise ValueError("standard_errors must be finite and nonnegative")
        if (
            isinstance(self.selected_index, (bool, np.bool_))
            or not isinstance(self.selected_index, Integral)
        ):
            raise TypeError("selected_index must be an integer")
        selected_index = int(self.selected_index)
        if not (0 <= selected_index < ratios.size):
            raise ValueError("selected_index is outside the lambda grid")
        if self.selection_rule not in ("minimum", "one_standard_error"):
            raise ValueError("selection_rule is invalid")

        # Recompute every summary from the immutable fold evidence so a public
        # result cannot pair a tuning decision with contradictory score arrays.
        expected_means = scores.mean(axis=0)
        if scores.shape[0] > 1:
            expected_errors = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
        else:
            expected_errors = np.zeros(ratios.size, dtype=float)
        if not np.allclose(means, expected_means, rtol=1e-12, atol=1e-15):
            raise ValueError("mean_scores must equal the fold-score means")
        if not np.allclose(
            standard_errors,
            expected_errors,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "standard_errors must equal the fold-score standard errors"
            )

        minimum_score = float(np.min(expected_means))
        tied_minima = np.flatnonzero(
            np.isclose(expected_means, minimum_score, rtol=1e-12, atol=1e-15)
        )
        minimum_index = int(tied_minima[np.argmax(ratios[tied_minima])])
        if self.selection_rule == "minimum":
            expected_index = minimum_index
        else:
            threshold = expected_means[minimum_index] + expected_errors[minimum_index]
            eligible = np.flatnonzero(expected_means <= threshold)
            expected_index = int(eligible[np.argmax(ratios[eligible])])
        if selected_index != expected_index:
            raise ValueError("selected_index does not follow selection_rule")
        if (
            isinstance(self.selected_lambda_ratio, (bool, np.bool_))
            or not isinstance(self.selected_lambda_ratio, Real)
        ):
            raise TypeError("selected_lambda_ratio must be a real scalar")
        selected_ratio = float(self.selected_lambda_ratio)
        if not np.isfinite(selected_ratio) or not np.isclose(
            selected_ratio,
            ratios[selected_index],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("selected_lambda_ratio must match selected_index")

        object.__setattr__(self, "selected_index", selected_index)
        object.__setattr__(self, "selected_lambda_ratio", selected_ratio)
        for field_name, values in (
            ("lambda_ratios", ratios),
            ("fold_scores", scores),
            ("mean_scores", means),
            ("standard_errors", standard_errors),
            ("fold_lambda_maxima", maxima),
        ):
            object.__setattr__(self, field_name, _immutable_array(values, float))


@dataclass(frozen=True, slots=True)
class StabilitySelectionResult:
    """Store exploratory LASSO selection consistency across member subsamples."""

    candidate_names: tuple[str, ...]
    lambda_ratio: float
    selection_threshold: float
    coefficient_matrix: FloatArray
    selected_matrix: BoolArray
    selection_frequencies: FloatArray
    selected_candidate_names: tuple[str, ...]
    sampled_group_ids: StringArray

    def __post_init__(self) -> None:
        """Validate summaries and isolate all resample-level arrays.

        Args:
            self: Newly constructed stability-selection result.

        Returns:
            None.

        Side effects:
            Replaces matrices and frequencies with immutable copies.

        Statistical intent:
            Keeps empirical selection frequencies tied to the exact coefficient
            paths and member subsamples that generated them.
        """
        coefficients = np.asarray(self.coefficient_matrix, dtype=float)
        selected = np.asarray(self.selected_matrix, dtype=bool)
        frequencies = np.asarray(self.selection_frequencies, dtype=float)
        sampled_groups = np.asarray(self.sampled_group_ids)
        names = _validate_names(
            self.candidate_names,
            coefficients.shape[1] if coefficients.ndim == 2 else -1,
            "candidate_names",
        )
        if (
            isinstance(self.lambda_ratio, (bool, np.bool_))
            or not isinstance(self.lambda_ratio, Real)
        ):
            raise TypeError("lambda_ratio must be a real scalar")
        if (
            isinstance(self.selection_threshold, (bool, np.bool_))
            or not isinstance(self.selection_threshold, Real)
        ):
            raise TypeError("selection_threshold must be a real scalar")
        lambda_ratio = float(self.lambda_ratio)
        selection_threshold = float(self.selection_threshold)
        if not np.isfinite(lambda_ratio) or not 0.0 < lambda_ratio <= 1.0:
            raise ValueError("lambda_ratio must lie in (0, 1]")
        if (
            not np.isfinite(selection_threshold)
            or not 0.0 < selection_threshold <= 1.0
        ):
            raise ValueError("selection_threshold must lie in (0, 1]")
        if coefficients.ndim != 2 or selected.shape != coefficients.shape:
            raise ValueError("Coefficient and selected matrices must align")
        if coefficients.shape[0] == 0 or coefficients.shape[1] == 0:
            raise ValueError("Stability results require resamples and candidates")
        if frequencies.shape != (coefficients.shape[1],):
            raise ValueError("Selection frequencies must align with candidates")
        if sampled_groups.ndim != 2 or sampled_groups.shape[0] != coefficients.shape[0]:
            raise ValueError("sampled_group_ids must have one row per resample")
        if not np.isfinite(coefficients).all() or not np.isfinite(frequencies).all():
            raise ValueError("Stability-selection values must be finite")
        if ((frequencies < 0.0) | (frequencies > 1.0)).any():
            raise ValueError("Selection frequencies must lie between zero and one")
        expected_frequencies = selected.mean(axis=0)
        if not np.allclose(
            frequencies,
            expected_frequencies,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Selection frequencies must equal selected-matrix means"
            )

        # Each row represents sampling without replacement; duplicate group IDs
        # would contradict the stability procedure's independent member unit.
        for sampled_row in sampled_groups:
            normalized_row = [str(value).strip() for value in sampled_row]
            if len(set(normalized_row)) != len(normalized_row):
                raise ValueError(
                    "sampled_group_ids cannot repeat a group within a resample"
                )
        expected_selected = tuple(
            name
            for name, frequency in zip(names, frequencies, strict=True)
            if frequency >= selection_threshold
        )
        if tuple(self.selected_candidate_names) != expected_selected:
            raise ValueError("Selected names must follow the frequency threshold")
        object.__setattr__(self, "candidate_names", names)
        object.__setattr__(self, "lambda_ratio", lambda_ratio)
        object.__setattr__(self, "selection_threshold", selection_threshold)
        object.__setattr__(
            self,
            "coefficient_matrix",
            _immutable_array(coefficients, float),
        )
        object.__setattr__(
            self,
            "selected_matrix",
            _immutable_array(selected, bool),
        )
        object.__setattr__(
            self,
            "selection_frequencies",
            _immutable_array(frequencies, float),
        )
        object.__setattr__(
            self,
            "sampled_group_ids",
            _immutable_strings(sampled_groups.ravel()).reshape(sampled_groups.shape),
        )


def _prepare_group_labels(
    group_ids: ArrayLike,
    strata: ArrayLike | None,
) -> tuple[StringArray, StringArray, StringArray, StringArray]:
    """Validate row labels and reduce them to one stratum per unique group.

    Args:
        group_ids: One member or cluster identifier per row.
        strata: Optional row-aligned stratification labels; omission uses one
            common stratum.

    Returns:
        Row group IDs, row strata, unique groups, and aligned group strata.

    Side effects:
        None.

    Statistical intent:
        Establishes the independent resampling unit before fold assignment.
    """
    row_groups = _immutable_strings(group_ids)
    if row_groups.size == 0:
        raise ValueError("group_ids cannot be empty")
    if strata is None:
        row_strata = _immutable_strings(np.repeat("all", row_groups.size))
    else:
        row_strata = _immutable_strings(strata)
        if row_strata.size != row_groups.size:
            raise ValueError("strata must contain one value per row")

    # Drop duplicate row labels only after confirming a group never changes
    # stratum; the first-occurrence order becomes the reproducible group order.
    group_table = pd.DataFrame({"group": row_groups, "stratum": row_strata})
    if group_table.groupby("group", sort=False)["stratum"].nunique().gt(1).any():
        raise ValueError("strata cannot vary within group")
    unique_table = group_table.drop_duplicates("group", keep="first")
    unique_groups = _immutable_strings(unique_table["group"].to_numpy())
    group_strata = _immutable_strings(unique_table["stratum"].to_numpy())
    return row_groups, row_strata, unique_groups, group_strata


def make_grouped_repeated_folds(
    group_ids: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> tuple[GroupedFold, ...]:
    """Create deterministic stratified repeated folds without splitting groups.

    Args:
        group_ids: One member identifier per observation row.
        strata: Optional row-aligned labels that must be constant per member.
        n_splits: Number of nonempty test folds in each repeat.
        n_repeats: Number of independent randomized fold assignments.
        random_state: Explicit nonnegative random seed.

    Returns:
        Immutable folds ordered by repeat and then fold number.

    Side effects:
        None; randomization uses a local generator.

    Statistical intent:
        Keeps correlated rows from one member together, gives every member one
        test appearance per repeat, and approximately preserves stratum mix by
        shuffled round-robin assignment.
    """
    validated_splits = _validate_positive_integer(n_splits, "n_splits")
    validated_repeats = _validate_positive_integer(n_repeats, "n_repeats")
    seed = _validate_random_state(random_state)
    row_groups, _, unique_groups, group_strata = _prepare_group_labels(
        group_ids,
        strata,
    )
    if validated_splits < 2:
        raise ValueError("n_splits must be at least two")
    if validated_splits > unique_groups.size:
        raise ValueError("n_splits cannot exceed the number of groups")

    generator = np.random.default_rng(seed)
    folds: list[GroupedFold] = []
    stratum_levels = tuple(dict.fromkeys(group_strata.tolist()))
    for repeat in range(validated_repeats):
        fold_groups: list[list[str]] = [[] for _ in range(validated_splits)]
        round_robin_offset = 0

        # Shuffle only within strata, then rotate a global round-robin cursor.
        # This balances each stratum while also preventing empty folds when
        # several strata individually contain fewer groups than n_splits.
        for stratum in stratum_levels:
            stratum_groups = np.array(
                unique_groups[group_strata == stratum],
                copy=True,
            )
            generator.shuffle(stratum_groups)
            for position, group in enumerate(stratum_groups):
                fold_number = (round_robin_offset + position) % validated_splits
                fold_groups[fold_number].append(str(group))
            round_robin_offset = (
                round_robin_offset + stratum_groups.size
            ) % validated_splits

        for fold_number, test_groups in enumerate(fold_groups):
            test_mask = np.isin(row_groups, test_groups)
            test_indices = np.flatnonzero(test_mask)
            train_indices = np.flatnonzero(~test_mask)
            folds.append(
                GroupedFold(
                    repeat=repeat,
                    fold=fold_number,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )
    return tuple(folds)


def _prepare_member_features(
    member_features: pd.DataFrame,
    module_spec: ModuleSpecification,
) -> pd.DataFrame:
    """Validate and normalize one-row-per-member modeling features.

    Args:
        member_features: Endpoint, factor, engagement, tenure, and module data.
        module_spec: Whether candidates use module mean or four domain values.

    Returns:
        A copied dataframe with normalized factors and finite numeric columns.

    Side effects:
        None; the caller's dataframe is not modified.

    Statistical intent:
        Ensures missingness or malformed columns cannot silently change the
        analysis sample or candidate specification.
    """
    if not isinstance(member_features, pd.DataFrame):
        raise TypeError("member_features must be a pandas DataFrame")
    if module_spec not in ("mean", "domains"):
        raise ValueError("module_spec must be 'mean' or 'domains'")
    module_columns = (
        _MODULE_MEAN_CANDIDATES
        if module_spec == "mean"
        else _MODULE_DOMAIN_CANDIDATES
    )
    required = set(_MEMBER_COLUMNS + _ENGAGEMENT_CANDIDATES + module_columns)
    missing = sorted(required.difference(member_features.columns))
    if missing:
        raise ValueError(
            "member_features is missing required columns: " + ", ".join(missing)
        )
    if member_features.empty:
        raise ValueError("member_features cannot be empty")
    prepared = member_features.loc[:, list(required)].copy()

    # Normalize member and factor labels before duplicate and treatment-coding
    # checks so whitespace cannot create false categories or identities.
    for column in ("member_id", "member_type", "sex"):
        normalized = pd.Series(
            _immutable_strings(prepared[column].to_numpy()),
            index=prepared.index,
        )
        prepared[column] = normalized
    if prepared["member_id"].duplicated().any():
        raise ValueError("member_features must contain one row per member_id")

    numeric_columns = (
        "first_weight",
        "last_weight",
        *_ENGAGEMENT_CANDIDATES,
        *module_columns,
    )
    try:
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Member feature values must be numeric") from error
    numeric = prepared.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Member feature values must be finite")
    if not (prepared[["first_weight", "last_weight"]] > 0.0).all(axis=None):
        raise ValueError("Endpoint weights must be strictly positive")
    return prepared.reset_index(drop=True)


def _treatment_code(
    values: pd.Series,
    prefix: str,
    *,
    levels: tuple[str, ...] | None = None,
    reference: str | None = None,
) -> tuple[FloatArray, tuple[str, ...], tuple[str, ...], str]:
    """Create deterministic reference-coded factor columns without an intercept.

    Args:
        values: Normalized categorical values.
        prefix: Predictor name used in generated labels.
        levels: Optional fitted level order; omission sorts observed levels.
        reference: Optional fitted reference. For ``member_type``, omission
            uses ``Coaching Only`` when observed; otherwise the first sorted
            level is used.

    Returns:
        Dummy matrix, names, full levels, and selected reference level.

    Side effects:
        None.

    Statistical intent:
        Produces reproducible full-rank treatment contrasts aligned across
        construction, fitting, and held-out prediction.
    """
    observed = tuple(sorted(str(value) for value in values.unique()))
    fitted_levels = observed if levels is None else tuple(levels)
    if set(observed) != set(fitted_levels):
        raise ValueError(f"Observed {prefix} levels do not match fitted schema")
    selected_reference = (
        _CONFIRMED_REFERENCE_MEMBER_TYPE
        if reference is None and prefix == "member_type"
        and _CONFIRMED_REFERENCE_MEMBER_TYPE in fitted_levels
        else fitted_levels[0] if reference is None else reference
    )
    if selected_reference not in fitted_levels:
        raise ValueError(f"Reference {prefix} level is not observed")
    contrast_levels = tuple(
        level for level in fitted_levels if level != selected_reference
    )
    factor_values = values.astype(str).to_numpy()
    if contrast_levels:
        design = np.column_stack(
            [(factor_values == level).astype(float) for level in contrast_levels]
        )
    else:
        design = np.empty((len(values), 0), dtype=float)
    names = tuple(f"{prefix}[{level}]" for level in contrast_levels)
    return design, names, fitted_levels, selected_reference


def _candidate_matrix(
    prepared: pd.DataFrame,
    module_spec: ModuleSpecification,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Assemble engagement, tenure, module, and sex candidate columns.

    Args:
        prepared: Validated member-level feature rows.
        module_spec: Module mean or separate domain specification.

    Returns:
        Candidate matrix and exact column names.

    Side effects:
        None.

    Statistical intent:
        Keeps collinear module-mean and module-domain alternatives in separate
        penalized analyses while treating sex as a categorical candidate.
    """
    module_columns = (
        _MODULE_MEAN_CANDIDATES
        if module_spec == "mean"
        else _MODULE_DOMAIN_CANDIDATES
    )
    continuous_names = _ENGAGEMENT_CANDIDATES + module_columns
    continuous = prepared.loc[:, list(continuous_names)].to_numpy(dtype=float)
    sex_design, sex_names, _, _ = _treatment_code(prepared["sex"], "sex")
    return (
        np.column_stack([continuous, sex_design]),
        continuous_names + sex_names,
    )


def build_percentage_penalized_design(
    member_features: pd.DataFrame,
    module_spec: ModuleSpecification = "mean",
) -> PenalizedDesign:
    """Build the partially penalized percentage-loss regression design.

    Args:
        member_features: One row per member with outcomes and candidate features.
        module_spec: ``"mean"`` or ``"domains"`` module candidate family.

    Returns:
        An immutable design with percentage loss, an unpenalized baseline/type
        base, penalized exploratory candidates, member IDs, and type strata.

    Side effects:
        None; input rows are copied and no files are written.

    Statistical intent:
        Models ``100*(first-last)/first`` while always adjusting for baseline
        weight and member type before L1 selection among added covariates.
    """
    prepared = _prepare_member_features(member_features, module_spec)
    first_weight = prepared["first_weight"].to_numpy(dtype=float)
    last_weight = prepared["last_weight"].to_numpy(dtype=float)

    # Positive percentage loss represents weight reduction, matching the
    # confirmed feed-forward outcome definition exactly.
    outcome = 100.0 * (first_weight - last_weight) / first_weight
    member_type_design, member_type_names, _, _ = _treatment_code(
        prepared["member_type"],
        "member_type",
    )
    base_design = np.column_stack(
        [np.ones(len(prepared)), first_weight, member_type_design]
    )
    base_names = ("intercept", "first_weight", *member_type_names)
    candidate_design, candidate_names = _candidate_matrix(prepared, module_spec)
    return PenalizedDesign(
        outcome=outcome,
        base_design=base_design,
        candidate_design=candidate_design,
        base_names=base_names,
        candidate_names=candidate_names,
        group_ids=prepared["member_id"].to_numpy(),
        strata=prepared["member_type"].to_numpy(),
    )


def _longitudinal_fixed_design(
    prepared: pd.DataFrame,
    fitted: LongitudinalGLSResult,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Recreate the fitted longitudinal mean-model rows in member blocks.

    Args:
        prepared: Validated one-row-per-member features.
        fitted: Training-fitted longitudinal result and prediction schema.

    Returns:
        An ``N x 2 x P`` fixed-effect array and aligned coefficient names.

    Side effects:
        None.

    Statistical intent:
        Preserves the exact additive or type-specific time structure used to
        estimate the covariance later used for whitening.
    """
    schema = fitted.schema
    type_design, type_names, _, _ = _treatment_code(
        prepared["member_type"],
        "member_type",
        levels=schema.member_type_levels,
        reference=schema.reference_member_type,
    )
    factor_design = np.column_stack([np.ones(len(prepared)), type_design])
    factor_names = ("intercept", *type_names)
    if fitted.include_time_by_member_type:
        difference_design = factor_design
        difference_names = ("time",) + tuple(
            f"time:{name}" for name in type_names
        )
    else:
        difference_design = np.ones((len(prepared), 1), dtype=float)
        difference_names = ("time",)

    # Baseline rows carry no change terms; follow-up rows carry the fitted time
    # design. Member-major blocking keeps each 2xP system ready for whitening.
    zero_difference = np.zeros_like(difference_design)
    baseline_rows = np.column_stack([factor_design, zero_difference])
    follow_up_rows = np.column_stack([factor_design, difference_design])
    blocks = np.stack([baseline_rows, follow_up_rows], axis=1)
    names = factor_names + difference_names
    if names != fitted.coefficient_names:
        raise ValueError("Fitted longitudinal coefficient schema is inconsistent")
    return blocks, names


def _require_same_longitudinal_training_fit(
    prepared: pd.DataFrame,
    fitted: LongitudinalGLSResult,
) -> None:
    """Verify that a longitudinal result is the deterministic fit of these rows.

    Args:
        prepared: Validated member rows proposed for penalized design building.
        fitted: Longitudinal result claimed to come from the same training rows.

    Returns:
        None.

    Side effects:
        Refits in memory for provenance validation; inputs remain unchanged.

    Statistical intent:
        Prevents covariance leakage or schema reuse from another training sample
        before retrospective two-stage GLS whitening.
    """
    if not isinstance(fitted, LongitudinalGLSResult):
        raise TypeError("fitted_longitudinal must be a LongitudinalGLSResult")
    if fitted.n_members != len(prepared) or fitted.n_observations != 2 * len(prepared):
        raise ValueError(
            "fitted_longitudinal must come from the same training members"
        )

    # Exact normalized IDs provide sample provenance that numerical estimates
    # alone cannot establish: replacing every identifier can leave coefficients
    # and covariance unchanged while representing a different training sample.
    supplied_member_ids = tuple(
        sorted(prepared["member_id"].astype(str).str.strip())
    )
    fitted_member_ids = tuple(sorted(fitted.training_member_ids))
    if supplied_member_ids != fitted_member_ids:
        raise ValueError(
            "fitted_longitudinal must come from the same training members"
        )

    # The estimator is deterministic, so recreating it supplies the strongest
    # available provenance check without changing the base-result data contract.
    recreated = fit_longitudinal_gls(
        prepared,
        outcome_scale=fitted.outcome_scale,
        covariance_structure=fitted.covariance_structure,
        include_time_by_member_type=fitted.include_time_by_member_type,
        reference_member_type=fitted.schema.reference_member_type,
    )
    matching = (
        recreated.coefficient_names == fitted.coefficient_names
        and recreated.schema == fitted.schema
        and np.allclose(
            recreated.coefficients,
            fitted.coefficients,
            rtol=1e-10,
            atol=1e-10,
        )
        and np.allclose(
            recreated.covariance_matrix,
            fitted.covariance_matrix,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    if not matching:
        raise ValueError(
            "fitted_longitudinal must come from the same training members"
        )


def build_longitudinal_penalized_design(
    member_features: pd.DataFrame,
    fitted_longitudinal: LongitudinalGLSResult,
    module_spec: ModuleSpecification = "mean",
) -> PenalizedDesign:
    """Build a whitened retrospective two-stage penalized marginal-GLS design.

    Args:
        member_features: Same training members used for the supplied GLS fit,
            with endpoint outcomes and candidate features.
        fitted_longitudinal: Training-only raw- or log-scale longitudinal fit
            whose two-by-two residual covariance is used for whitening.
        module_spec: ``"mean"`` or ``"domains"`` module candidate family.

    Returns:
        A two-row-per-member whitened design. Longitudinal base terms and all
        candidate main effects are unpenalized; time-by-candidate interactions
        are penalized.

    Side effects:
        None; a deterministic in-memory refit verifies covariance provenance.

    Statistical intent:
        Performs exploratory retrospective two-stage penalized marginal GLS,
        not joint penalized mixed-effects ML. Hierarchical main effects remain
        whenever LASSO considers covariate associations with change.
    """
    if not isinstance(fitted_longitudinal, LongitudinalGLSResult):
        raise TypeError("fitted_longitudinal must be a LongitudinalGLSResult")
    prepared = _prepare_member_features(member_features, module_spec)
    _require_same_longitudinal_training_fit(prepared, fitted_longitudinal)
    fixed_blocks, fixed_names = _longitudinal_fixed_design(
        prepared,
        fitted_longitudinal,
    )
    candidate_values, candidate_names = _candidate_matrix(prepared, module_spec)
    n_members, n_candidates = candidate_values.shape

    # Candidate main effects repeat at both occasions and stay unpenalized for
    # hierarchy. Only follow-up rows carry time-by-candidate interactions.
    main_effect_blocks = np.repeat(candidate_values[:, None, :], 2, axis=1)
    unwhitened_base = np.concatenate(
        [fixed_blocks, main_effect_blocks],
        axis=2,
    )
    interaction_blocks = np.zeros((n_members, 2, n_candidates), dtype=float)
    interaction_blocks[:, 1, :] = candidate_values
    base_names = fixed_names + candidate_names
    interaction_names = tuple(f"time:{name}" for name in candidate_names)

    raw_outcomes = prepared[["first_weight", "last_weight"]].to_numpy(
        dtype=float
    )
    if fitted_longitudinal.outcome_scale == "log":
        outcome_blocks = np.log(raw_outcomes)
    else:
        outcome_blocks = raw_outcomes
    try:
        cholesky = np.linalg.cholesky(fitted_longitudinal.covariance_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Fitted longitudinal covariance is not positive definite"
        ) from error

    whitened_outcome = np.empty_like(outcome_blocks)
    whitened_base = np.empty_like(unwhitened_base)
    whitened_candidates = np.empty_like(interaction_blocks)
    for member_index in range(n_members):
        # Solving L*z=block applies Sigma^(-1/2) without explicitly inverting
        # covariance, preserving each member's paired residual geometry.
        whitened_outcome[member_index] = np.linalg.solve(
            cholesky,
            outcome_blocks[member_index],
        )
        whitened_base[member_index] = np.linalg.solve(
            cholesky,
            unwhitened_base[member_index],
        )
        whitened_candidates[member_index] = np.linalg.solve(
            cholesky,
            interaction_blocks[member_index],
        )

    return PenalizedDesign(
        outcome=whitened_outcome.reshape(-1),
        base_design=whitened_base.reshape(2 * n_members, -1),
        candidate_design=whitened_candidates.reshape(2 * n_members, -1),
        base_names=base_names,
        candidate_names=interaction_names,
        group_ids=np.repeat(prepared["member_id"].to_numpy(), 2),
        strata=np.repeat(prepared["member_type"].to_numpy(), 2),
    )


def _validate_lambda_ratios(lambda_ratios: ArrayLike) -> FloatArray:
    """Validate a unique finite tuning grid within ``(0, 1]``.

    Args:
        lambda_ratios: Fractions of each training fold's lambda-max.

    Returns:
        A one-dimensional float array preserving supplied order.

    Side effects:
        None.

    Statistical intent:
        Keeps every candidate penalty interpretable relative to a fold-local
        lambda-max and prevents duplicate score columns.
    """
    try:
        ratios = np.asarray(lambda_ratios, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("lambda_ratios must be numeric") from error
    if ratios.ndim != 1 or ratios.size == 0:
        raise ValueError("lambda_ratios must be a nonempty one-dimensional grid")
    if not np.isfinite(ratios).all() or ((ratios <= 0.0) | (ratios > 1.0)).any():
        raise ValueError("lambda_ratios must be unique finite values in (0, 1]")
    if np.unique(ratios).size != ratios.size:
        raise ValueError("lambda_ratios must be unique finite values in (0, 1]")
    return ratios


def select_lambda_by_grouped_cv(
    design: PenalizedDesign,
    lambda_ratios: ArrayLike,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    selection_rule: SelectionRule = "one_standard_error",
) -> LambdaCVResult:
    """Tune a partially penalized LASSO by repeated grouped inner CV.

    Args:
        design: Immutable base/candidate design for the current training sample.
        lambda_ratios: Unique penalty fractions in ``(0, 1]``.
        n_splits: Member-level inner folds per repeat.
        n_repeats: Number of inner-CV repeats.
        random_state: Explicit seed for fold randomization.
        selection_rule: Minimum held-out MSE or strongest ratio within one
            standard error of the minimum.

    Returns:
        Fold score matrix, summaries, fold-local lambda maxima, and selected
        ratio. The reported fold-score standard error is a one-SE tuning
        heuristic; repeated folds are correlated, and the result intentionally
        contains no inferential p-values.

    Side effects:
        None; every LASSO fit and scale estimate is fold-local and in memory.

    Statistical intent:
        Tunes predictive shrinkage without splitting member rows or allowing
        held-out candidates to influence training means, scales, or lambda-max.
        These inner-CV scores select a penalty only; they are not an unbiased
        estimate of final model performance, which requires an untouched outer
        assessment split. For a prewhitened longitudinal design, covariance is
        fixed at design construction and is not re-estimated inside these folds;
        that path is retrospective two-stage exploration only. Use CV-based
        performance claims from this interface only for the percentage design.
    """
    if not isinstance(design, PenalizedDesign):
        raise TypeError("design must be a PenalizedDesign")
    ratios = _validate_lambda_ratios(lambda_ratios)
    if selection_rule not in ("minimum", "one_standard_error"):
        raise ValueError("selection_rule must be 'minimum' or 'one_standard_error'")
    folds = make_grouped_repeated_folds(
        design.group_ids,
        strata=design.strata,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    fold_scores = np.empty((len(folds), ratios.size), dtype=float)
    fold_lambda_maxima = np.empty(len(folds), dtype=float)

    for fold_index, fold in enumerate(folds):
        for ratio_index, ratio in enumerate(ratios):
            # fit_partially_penalized_lasso derives centering, population
            # scaling, and lambda-max exclusively from these training rows.
            fitted = fit_partially_penalized_lasso(
                design.outcome[fold.train_indices],
                design.base_design[fold.train_indices],
                design.candidate_design[fold.train_indices],
                design.base_names,
                design.candidate_names,
                float(ratio),
            )
            predicted = predict_partially_penalized_lasso(
                fitted,
                design.base_design[fold.test_indices],
                design.candidate_design[fold.test_indices],
            )
            residual = design.outcome[fold.test_indices] - predicted
            fold_scores[fold_index, ratio_index] = float(np.mean(residual**2))
            if ratio_index == 0:
                fold_lambda_maxima[fold_index] = fitted.lambda_max

    mean_scores = fold_scores.mean(axis=0)
    if len(folds) > 1:
        # This descriptive SE powers only the conventional one-SE tuning rule.
        # Repeated fold scores share observations and are not independent data
        # for inferential confidence intervals or final performance claims.
        standard_errors = fold_scores.std(axis=0, ddof=1) / np.sqrt(len(folds))
    else:
        standard_errors = np.zeros(ratios.size, dtype=float)
    minimum_score = float(np.min(mean_scores))
    minimum_candidates = np.flatnonzero(
        np.isclose(mean_scores, minimum_score, rtol=1e-12, atol=1e-15)
    )
    # Resolve exact score ties toward the stronger penalty for deterministic
    # parsimony, then optionally apply the wider one-standard-error threshold.
    minimum_index = int(
        minimum_candidates[np.argmax(ratios[minimum_candidates])]
    )
    if selection_rule == "minimum":
        selected_index = minimum_index
    else:
        threshold = mean_scores[minimum_index] + standard_errors[minimum_index]
        eligible = np.flatnonzero(mean_scores <= threshold)
        selected_index = int(eligible[np.argmax(ratios[eligible])])

    return LambdaCVResult(
        lambda_ratios=ratios,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        standard_errors=standard_errors,
        fold_lambda_maxima=fold_lambda_maxima,
        selected_lambda_ratio=float(ratios[selected_index]),
        selected_index=selected_index,
        selection_rule=selection_rule,
    )


def _stratified_group_subsample(
    unique_groups: StringArray,
    group_strata: StringArray,
    sample_size: int,
    generator: np.random.Generator,
) -> StringArray:
    """Draw one proportional stratified group sample without replacement.

    Args:
        unique_groups: Unique group IDs in stable source order.
        group_strata: One aligned stratum per unique group.
        sample_size: Exact number of groups to retain.
        generator: Local seeded NumPy random generator.

    Returns:
        Immutable sampled group IDs sorted by original group order.

    Side effects:
        Advances only the supplied local generator state.

    Statistical intent:
        Preserves rare strata as far as sample size permits while keeping the
        member, not the row, as the independent stability-selection unit.
    """
    levels = tuple(dict.fromkeys(group_strata.tolist()))
    if sample_size < len(levels):
        raise ValueError(
            "subsample_fraction is too small to retain every stratum"
        )
    counts = np.array(
        [np.count_nonzero(group_strata == level) for level in levels],
        dtype=int,
    )
    ideal = sample_size * counts / unique_groups.size
    allocation = np.floor(ideal).astype(int)
    allocation = np.maximum(allocation, 1)

    # Largest-remainder corrections hit the requested total while respecting
    # each stratum's available groups and one-member minimum.
    while allocation.sum() < sample_size:
        capacity = counts - allocation
        eligible = np.flatnonzero(capacity > 0)
        priorities = ideal[eligible] - allocation[eligible]
        chosen = int(eligible[np.argmax(priorities)])
        allocation[chosen] += 1
    while allocation.sum() > sample_size:
        eligible = np.flatnonzero(allocation > 1)
        priorities = allocation[eligible] - ideal[eligible]
        chosen = int(eligible[np.argmax(priorities)])
        allocation[chosen] -= 1

    selected: set[str] = set()
    for level, count in zip(levels, allocation, strict=True):
        candidates = unique_groups[group_strata == level]
        chosen = generator.choice(candidates, size=int(count), replace=False)
        selected.update(str(value) for value in chosen)
    ordered = [str(group) for group in unique_groups if str(group) in selected]
    return _immutable_strings(ordered)


def stability_select_lasso(
    design: PenalizedDesign,
    *,
    lambda_ratio: Real,
    n_resamples: int,
    subsample_fraction: Real,
    selection_threshold: Real,
    random_state: int,
) -> StabilitySelectionResult:
    """Estimate exploratory LASSO selection frequency across member subsamples.

    Args:
        design: Immutable design for the current analysis sample.
        lambda_ratio: Penalty fraction in ``(0, 1]`` recomputed per subsample.
        n_resamples: Number of stratified samples without replacement.
        subsample_fraction: Fraction of unique members retained per sample.
        selection_threshold: Frequency in ``(0, 1]`` required for final listing.
        random_state: Explicit seed for member sampling.

    Returns:
        Raw coefficient and selected matrices, empirical frequencies, selected
        names, and sampled member IDs.

    Side effects:
        None; randomization uses a local generator and inputs remain immutable.

    Statistical intent:
        Measures selection consistency under member perturbation. Frequencies
        are exploratory stability summaries, not significance probabilities.
    """
    if not isinstance(design, PenalizedDesign):
        raise TypeError("design must be a PenalizedDesign")
    if isinstance(lambda_ratio, (bool, np.bool_)) or not isinstance(lambda_ratio, Real):
        raise TypeError("lambda_ratio must be a real scalar")
    validated_ratio = float(lambda_ratio)
    if not np.isfinite(validated_ratio) or not 0.0 < validated_ratio <= 1.0:
        raise ValueError("lambda_ratio must lie in (0, 1]")
    validated_resamples = _validate_positive_integer(n_resamples, "n_resamples")
    for value, label in (
        (subsample_fraction, "subsample_fraction"),
        (selection_threshold, "selection_threshold"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{label} must be a real scalar")
    validated_fraction = float(subsample_fraction)
    validated_threshold = float(selection_threshold)
    if not np.isfinite(validated_fraction) or not 0.0 < validated_fraction < 1.0:
        raise ValueError("subsample_fraction must lie strictly between zero and one")
    if not np.isfinite(validated_threshold) or not 0.0 < validated_threshold <= 1.0:
        raise ValueError("selection_threshold must lie in (0, 1]")
    seed = _validate_random_state(random_state)
    row_groups, _, unique_groups, group_strata = _prepare_group_labels(
        design.group_ids,
        design.strata,
    )
    sample_size = max(1, int(round(validated_fraction * unique_groups.size)))
    if sample_size >= unique_groups.size:
        raise ValueError("subsample_fraction must omit at least one group")

    n_candidates = len(design.candidate_names)
    coefficients = np.empty((validated_resamples, n_candidates), dtype=float)
    selected = np.empty((validated_resamples, n_candidates), dtype=bool)
    sampled_ids = np.empty((validated_resamples, sample_size), dtype=object)
    generator = np.random.default_rng(seed)
    for resample in range(validated_resamples):
        sampled_groups = _stratified_group_subsample(
            unique_groups,
            group_strata,
            sample_size,
            generator,
        )
        sampled_ids[resample] = sampled_groups
        sampled_rows = np.flatnonzero(np.isin(row_groups, sampled_groups))

        # Each fit re-estimates candidate means, scales, and lambda-max on only
        # the sampled members before applying the fixed lambda ratio.
        fitted = fit_partially_penalized_lasso(
            design.outcome[sampled_rows],
            design.base_design[sampled_rows],
            design.candidate_design[sampled_rows],
            design.base_names,
            design.candidate_names,
            validated_ratio,
        )
        coefficients[resample] = fitted.candidate_coefficients
        selected[resample] = np.abs(
            fitted.standardized_candidate_coefficients
        ) > 1e-12

    frequencies = selected.mean(axis=0)
    selected_names = tuple(
        name
        for name, frequency in zip(
            design.candidate_names,
            frequencies,
            strict=True,
        )
        if frequency >= validated_threshold
    )
    return StabilitySelectionResult(
        candidate_names=design.candidate_names,
        lambda_ratio=validated_ratio,
        selection_threshold=validated_threshold,
        coefficient_matrix=coefficients,
        selected_matrix=selected,
        selection_frequencies=frequencies,
        selected_candidate_names=selected_names,
        sampled_group_ids=sampled_ids,
    )


def compare_base_models(
    members: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Compare both base models on identical held-out raw last weights.

    Args:
        members: One row per member with ID, type, first weight, and last weight.
        n_splits: Number of stratified member folds per repeat.
        n_repeats: Number of fold-assignment repeats.
        random_state: Explicit seed shared by both model families.

    Returns:
        Deterministic rows with repeat, fold, model, test size, RMSE, and MAE.

    Side effects:
        None; models are fit in memory and input data is not modified.

    Statistical intent:
        Fits log-weight compound-symmetry marginal GLS and percentage-loss OLS
        on training members, then compares only common raw follow-up prediction
        errors. Cross-family AIC or likelihood is deliberately excluded.
    """
    if not isinstance(members, pd.DataFrame):
        raise TypeError("members must be a pandas DataFrame")
    required = {"member_id", "member_type", "first_weight", "last_weight"}
    missing = sorted(required.difference(members.columns))
    if missing:
        raise ValueError("members is missing required columns: " + ", ".join(missing))
    if members["member_id"].duplicated().any():
        raise ValueError("members must contain one row per member_id")
    comparison_members = members.reset_index(drop=True).copy()
    folds = make_grouped_repeated_folds(
        comparison_members["member_id"].to_numpy(),
        strata=comparison_members["member_type"].to_numpy(),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    records: list[dict[str, int | float | str]] = []
    for fold in folds:
        training = comparison_members.iloc[fold.train_indices]
        testing = comparison_members.iloc[fold.test_indices]
        observed = testing["last_weight"].to_numpy(dtype=float)

        # Both models see the same training and test members. Their predictions
        # are converted to raw last weight before computing paired metrics.
        longitudinal = fit_longitudinal_gls(
            training,
            outcome_scale="log",
            covariance_structure="compound_symmetry",
        )
        percentage = fit_percentage_loss_ols(training)
        predictions = (
            (
                "log_compound_symmetry_gls",
                predict_last_weight_longitudinal(longitudinal, testing),
            ),
            (
                "percentage_loss_ols",
                predict_last_weight_percentage_loss(percentage, testing),
            ),
        )
        for model_name, predicted in predictions:
            residual = observed - predicted
            records.append(
                {
                    "repeat": fold.repeat,
                    "fold": fold.fold,
                    "model": model_name,
                    "n_test": int(testing.shape[0]),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["repeat", "fold", "model", "n_test", "rmse", "mae"],
    )
