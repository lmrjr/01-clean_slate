"""Construct the eligible member cohort and two-occasion weight data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .nineam_data_loading import CaseStudyData

ELIGIBLE_SUBSCRIPTION_STATUSES = frozenset({"ACTIVE", "FINISHED"})
PRIMARY_MEMBER_TYPES = (
    "Active GLP-1 for Weight-loss",
    "Coaching Only",
)
REFERENCE_MEMBER_TYPE = "Coaching Only"
_EXCLUDED_MEMBER_TYPE_AUDIT_LABELS = {
    "Active GLP-1 for Diabetes": "excluded_active_glp1_diabetes",
    "Active Generic Medication for Weight-loss (NOT on GLP-1 for weight-loss)": (
        "excluded_active_generic_weight_loss_medication"
    ),
    "Null": "excluded_null_member_type",
}


@dataclass(frozen=True, slots=True)
class CohortResult:
    """Member-level and long-format cohort tables with exclusion counts."""

    members: pd.DataFrame
    long_weights: pd.DataFrame
    audit_counts: dict[str, int]


def _validate_member_ids(table: pd.DataFrame, source_name: str) -> None:
    """Reject unusable IDs and normalize valid IDs before member-level joins.

    A null or blank identifier has no defensible member-level interpretation.
    Trimming valid IDs also prevents whitespace from creating false unmatched
    records across sources.

    Args:
        table: Member-level dataframe whose identifier requires validation.
        source_name: Human-readable source label used in validation errors.

    Returns:
        None.

    Side effects:
        Replaces ``table['member_id']`` in place with trimmed string values.

    Statistical intent:
        Preserves one valid linkage key per analytical member.
    """
    normalized_ids = table["member_id"].astype("string").str.strip()
    invalid_ids = normalized_ids.isna() | normalized_ids.eq("")
    if invalid_ids.any():
        raise ValueError(
            f"{source_name} contains null or blank member_id values"
        )

    # Store the normalized key before duplicate checks and merges so identity
    # validation uses the exact key that defines an analytical member.
    table["member_id"] = normalized_ids


def _build_long_weights(members: pd.DataFrame) -> pd.DataFrame:
    """Create the two-occasion response table used by longitudinal models.

    Time is fixed at zero for the first weight and one for the last weight,
    matching the confirmed model specification independently of elapsed days.

    Args:
        members: Final member-level cohort containing paired weights and dates.

    Returns:
        A new dataframe with two ordered response rows per member.

    Side effects:
        None; ``members`` is not mutated.

    Statistical intent:
        Encodes the two-time-point repeated-measures response structure.
    """
    identity_columns = ["member_id", "member_type"]

    # Baseline rows define time zero and retain the member grouping variable
    # needed for within-member covariance estimation.
    first_rows = members.loc[
        :, [*identity_columns, "first_weight_at", "first_weight"]
    ].rename(
        columns={
            "first_weight_at": "measurement_at",
            "first_weight": "weight",
        }
    )
    first_rows["time"] = 0
    first_rows["occasion"] = "first"

    # Follow-up rows define time one, ensuring every retained member contributes
    # exactly one paired endpoint observation.
    last_rows = members.loc[
        :, [*identity_columns, "last_weight_at", "last_weight"]
    ].rename(
        columns={
            "last_weight_at": "measurement_at",
            "last_weight": "weight",
        }
    )
    last_rows["time"] = 1
    last_rows["occasion"] = "last"

    # Stable member/time ordering makes model matrices and serialized outputs
    # deterministic across runs.
    long_weights = pd.concat([first_rows, last_rows], ignore_index=True)
    long_weights = long_weights.loc[
        :,
        [
            "member_id",
            "member_type",
            "time",
            "occasion",
            "measurement_at",
            "weight",
        ],
    ]
    return long_weights.sort_values(
        ["member_id", "time"],
        kind="stable",
        ignore_index=True,
    )


def restrict_to_primary_member_types(cohort: CohortResult) -> CohortResult:
    """Keep the two confirmed comparable member types after eligibility.

    Args:
        cohort: Complete-pair eligibility result from ``build_analysis_cohort``.

    Returns:
        A separate cohort result with two long weight rows for each retained
        member and audit counts that reconcile the comparability exclusion.

    Side effects:
        None; the input result and its member table are not mutated.

    Statistical intent:
        Distinguishes complete-pair outcome eligibility from the predefined
        two-condition comparability restriction used for primary analyses.
    """
    # Apply the substantive condition filter after complete-pair eligibility so
    # attrition reporting retains the 633-member intermediate denominator.
    members = cohort.members.loc[
        cohort.members["member_type"].isin(PRIMARY_MEMBER_TYPES)
    ].copy()
    members = members.sort_values("member_id", kind="stable", ignore_index=True)
    audit_counts = dict(cohort.audit_counts)
    audit_counts["pre_member_type_restriction_members"] = int(
        len(cohort.members)
    )
    audit_counts["excluded_nonprimary_member_type"] = int(
        len(cohort.members) - len(members)
    )
    # Retain the supplied source labels in separate audit fields so excluded
    # comparability conditions reconcile without encoding their source totals.
    for member_type, audit_key in _EXCLUDED_MEMBER_TYPE_AUDIT_LABELS.items():
        audit_counts[audit_key] = int(
            cohort.members["member_type"].eq(member_type).sum()
        )
    audit_counts["included_members"] = int(len(members))
    return CohortResult(
        members=members,
        long_weights=_build_long_weights(members),
        audit_counts=audit_counts,
    )


def build_analysis_cohort(data: CaseStudyData) -> CohortResult:
    """Apply the confirmed eligibility rules to the case-study members.

    Filters are applied sequentially so the audit counts are mutually
    exclusive. Eligible members must have an ``ACTIVE`` or ``FINISHED``
    subscription, present and positive first and last weights, and a positive
    interval between those weights.

    Args:
        data: Canonical source tables returned by ``load_case_study_data``.

    Returns:
        Member-level rows, two-row-per-member longitudinal weights, and audit
        counts describing every exclusion stage.

    Side effects:
        None; source dataframes are copied and are not mutated.

    Statistical intent:
        Produces one consistent eligible population for paired and percentage-
        loss analyses while retaining a reconcilable attrition denominator.

    Raises:
        ValueError: If member identifiers are null, blank, or duplicated in a
            member-level source table.
    """
    # Cohort attrition starts with every enrolled member in demographics; body
    # weight records are supporting measurements, not the population frame.
    demographics = data.demographics.copy()
    body_weights = data.body_weights.copy()

    # Invalid identifiers cannot be interpreted as members and must be rejected
    # before any join can silently discard or combine them.
    _validate_member_ids(demographics, "Demographics")
    _validate_member_ids(body_weights, "Body weights")

    # A one-member-one-row contract is required for both the attrition audit
    # and paired-outcome construction.
    if demographics["member_id"].duplicated().any():
        raise ValueError("Demographics contains duplicate member_id values")
    if body_weights["member_id"].duplicated().any():
        raise ValueError("Body weights contains duplicate member_id values")

    # Orphan weight records are a source-quality diagnostic and are not part of
    # demographic attrition because no enrolled member can be assigned to them.
    orphan_weight_rows = ~body_weights["member_id"].isin(
        demographics["member_id"]
    )

    # A demographics-left join preserves the full enrollment denominator and
    # exposes members who never received a body-weight record.
    merged = demographics.merge(
        body_weights,
        how="left",
        on="member_id",
        validate="one_to_one",
        indicator=True,
    )

    audit_counts = {
        "source_weight_rows": int(len(body_weights)),
        "source_demographic_rows": int(len(demographics)),
        "orphan_weight_rows": int(orphan_weight_rows.sum()),
    }

    # Missing weight rows are the first mutually exclusive attrition stage,
    # ensuring all later exclusions reconcile to the demographic denominator.
    has_body_weight_row = merged["_merge"].eq("both")
    audit_counts["excluded_missing_body_weight_row"] = int(
        (~has_body_weight_row).sum()
    )
    candidates = (
        merged.loc[has_body_weight_row].drop(columns="_merge").copy()
    )

    # Status eligibility represents the confirmed active-or-finished analysis
    # population, with paused subscriptions excluded.
    candidates["subscription_status"] = (
        candidates["subscription_status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    eligible_status = candidates["subscription_status"].isin(
        ELIGIBLE_SUBSCRIPTION_STATUSES
    )
    audit_counts["excluded_ineligible_status"] = int(
        (~eligible_status).sum()
    )
    candidates = candidates.loc[eligible_status].copy()

    # Paired analyses require both observed outcomes; incomplete pairs cannot
    # contribute either percentage loss or a two-time-point trajectory.
    weights_present = candidates[["first_weight", "last_weight"]].notna().all(
        axis="columns"
    )
    audit_counts["excluded_missing_weights"] = int(
        (~weights_present).sum()
    )
    candidates = candidates.loc[weights_present].copy()

    # Positive weights keep percentage loss defined and exclude physiologically
    # impossible endpoints before modeling.
    weights_positive = (
        candidates[["first_weight", "last_weight"]] > 0
    ).all(axis="columns")
    audit_counts["excluded_nonpositive_weights"] = int(
        (~weights_positive).sum()
    )
    candidates = candidates.loc[weights_positive].copy()

    # Missing intervals cannot establish longitudinal ordering even when both
    # endpoint values are present.
    weight_days_present = candidates["weight_days"].notna()
    audit_counts["excluded_missing_weight_days"] = int(
        (~weight_days_present).sum()
    )
    candidates = candidates.loc[weight_days_present].copy()

    # A strictly positive interval prevents treating same-day or reversed
    # measurements as longitudinal change.
    weight_days_positive = candidates["weight_days"] > 0
    audit_counts["excluded_nonpositive_weight_days"] = int(
        (~weight_days_positive).sum()
    )
    members = candidates.loc[weight_days_positive].copy()

    # Positive values consistently denote weight reduction for both the
    # continuous pounds-lost response and the percentage-loss response.
    members["absolute_weight_loss"] = (
        members["first_weight"] - members["last_weight"]
    )
    members["percentage_loss"] = (
        100.0 * members["absolute_weight_loss"] / members["first_weight"]
    )
    members["weight_loss_success_5pct"] = members["percentage_loss"].ge(5.0)
    members = members.sort_values(
        "member_id",
        kind="stable",
        ignore_index=True,
    )
    audit_counts["included_members"] = int(len(members))

    # Supply both member-level and paired long-format representations from the
    # same final cohort so downstream approaches use an identical population.
    long_weights = _build_long_weights(members)
    return CohortResult(
        members=members,
        long_weights=long_weights,
        audit_counts=audit_counts,
    )
