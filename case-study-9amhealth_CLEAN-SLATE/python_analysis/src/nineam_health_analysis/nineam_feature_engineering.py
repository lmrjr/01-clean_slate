"""Build member-level engagement and curriculum features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from .nineam_cohort_selection import CohortResult
from .nineam_data_loading import CaseStudyData

_REPEATABLE_EVENTS_PER_MEMBER = 2.0
_REPEATABLE_MINIMUM_MEMBERS = 30
_MINIMUM_RATE_TENURE_DAYS = 7.0

MODULE_DOMAIN_DENOMINATORS: Mapping[str, float] = MappingProxyType(
    {
        "core": 9.0,
        "mindset": 4.0,
        "nutrition": 4.0,
        "physical_activity": 4.0,
    }
)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Member features, event classification, sparse event counts, and audits.

    ``member_event_counts`` contains only retained observed member-event pairs
    with their full-source repeatability classification. Absent pairs are
    intentionally omitted; reporting code expands them as zero only when it
    needs a complete member-level vector for an event type.
    """

    member_features: pd.DataFrame
    event_type_summary: pd.DataFrame
    member_event_counts: pd.DataFrame
    audit_counts: dict[str, int]


def classify_repeatable_events(engagement: pd.DataFrame) -> pd.DataFrame:
    """Classify engagement types using the confirmed frequency thresholds.

    Args:
        engagement: Event-level records containing ``member_id`` and
            ``event_type``.

    Returns:
        One row per event type with event count, distinct-member count,
        events per member, and the repeatable indicator.

    Side effects:
        None; ``engagement`` is not mutated.

    Statistical intent:
        Distinguishes sufficiently common recurring behavior from sparse event
        types before member-level repeatable volume is calculated.

    Raises:
        ValueError: If required columns are absent or an event type has no valid
            member identifiers.
    """
    required_columns = {"member_id", "event_type"}
    missing_columns = required_columns.difference(engagement.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Engagement data is missing required columns: " + missing_display
        )

    # Null or blank grouping keys have no defensible event-type or member-level
    # interpretation and would distort counts or create an empty category.
    for column in ["member_id", "event_type"]:
        normalized_values = engagement[column].astype("string").str.strip()
        invalid_values = normalized_values.isna() | normalized_values.eq("")
        if invalid_values.any():
            raise ValueError(
                f"Engagement {column} contains null or blank values"
            )

    if engagement.empty:
        return pd.DataFrame(
            {
                "event_type": pd.Series(dtype="string"),
                "events": pd.Series(dtype="int64"),
                "distinct_members": pd.Series(dtype="int64"),
                "events_per_member": pd.Series(dtype="float64"),
                "is_repeatable": pd.Series(dtype="bool"),
            }
        )

    # Aggregate both the numerator and denominator at event-type level so the
    # rate matches count(*) / count(distinct member_id) from the supplied R SQL.
    summary = (
        engagement.groupby("event_type", dropna=False, sort=True)
        .agg(
            events=("event_type", "size"),
            distinct_members=("member_id", "nunique"),
        )
        .reset_index()
    )
    if summary["distinct_members"].eq(0).any():
        raise ValueError(
            "Each engagement event type must have a valid member_id"
        )

    # The rule uses inclusive cutoffs: at least two events per contributing
    # member and at least 30 distinct contributing members.
    summary["events_per_member"] = (
        summary["events"] / summary["distinct_members"]
    )
    summary["is_repeatable"] = (
        summary["events_per_member"].ge(_REPEATABLE_EVENTS_PER_MEMBER)
        & summary["distinct_members"].ge(_REPEATABLE_MINIMUM_MEMBERS)
    )
    return summary


def build_member_features(
    data: CaseStudyData,
    cohort: CohortResult,
) -> FeatureResult:
    """Create censored engagement and module features for eligible members.

    Args:
        data: Canonical case-study source tables.
        cohort: Eligible member rows with last-weight dates and tenure days.

    Returns:
        A ``FeatureResult`` containing one enriched row per eligible member,
        full-source type-level repeatability, sparse retained member-event
        counts, and filtering/aggregation counts. Sparse counts omit absent
        pairs; reporting treats those pairs as zero when building full member
        vectors for an event type.

    Side effects:
        None; all source and cohort dataframes are copied before transformation.

    Statistical intent:
        Builds candidate covariates using information observed no later than
        each member's outcome date, preventing post-outcome feature leakage.

    Raises:
        ValueError: If required columns are absent or cohort member IDs repeat.
    """
    required_member_columns = {"member_id", "last_weight_at", "tenure_days"}
    missing_member_columns = required_member_columns.difference(
        cohort.members.columns
    )
    if missing_member_columns:
        missing_display = ", ".join(sorted(missing_member_columns))
        raise ValueError(
            "Cohort members are missing required columns: " + missing_display
        )
    if cohort.members["member_id"].duplicated().any():
        raise ValueError("Cohort members contain duplicate member_id values")

    required_engagement_columns = {"member_id", "activity_at", "event_type"}
    missing_engagement_columns = required_engagement_columns.difference(
        data.engagement.columns
    )
    if missing_engagement_columns:
        missing_display = ", ".join(sorted(missing_engagement_columns))
        raise ValueError(
            "Engagement data is missing required columns: " + missing_display
        )

    required_module_columns = {
        "member_id",
        "questionnaire_title",
        "answered_at",
    }
    missing_module_columns = required_module_columns.difference(
        data.module_completions.columns
    )
    if missing_module_columns:
        missing_display = ", ".join(sorted(missing_module_columns))
        raise ValueError(
            "Module data is missing required columns: " + missing_display
        )

    members = cohort.members.copy().reset_index(drop=True)
    cutoff_columns = members.loc[:, ["member_id", "last_weight_at"]]
    audit_counts = {
        "source_engagement_rows": int(len(data.engagement)),
        "source_module_rows": int(len(data.module_completions)),
        "included_members": int(len(members)),
    }

    # Match the supplied R sequence exactly: classify types on the complete
    # canonical engagement extract before applying cohort or outcome cutoffs.
    # The fixed classification is then applied only to each member's retained
    # records when breadth and volume are calculated below.
    event_type_summary = classify_repeatable_events(data.engagement)
    repeatable_types = set(
        event_type_summary.loc[
            event_type_summary["is_repeatable"], "event_type"
        ]
    )
    audit_counts["repeatable_event_types"] = int(len(repeatable_types))

    # Restrict to eligible members before time censoring; no lower date cutoff
    # is imposed, and events on the last-weight date remain observable.
    cohort_engagement = data.engagement.merge(
        cutoff_columns,
        how="inner",
        on="member_id",
        validate="many_to_one",
    )
    engagement_after_last_weight = cohort_engagement["activity_at"].gt(
        cohort_engagement["last_weight_at"]
    )
    engagement_missing_date = cohort_engagement[
        ["activity_at", "last_weight_at"]
    ].isna().any(axis="columns")
    retained_engagement = cohort_engagement.loc[
        ~engagement_after_last_weight & ~engagement_missing_date
    ].copy()
    audit_counts.update(
        {
            "cohort_engagement_rows": int(len(cohort_engagement)),
            "excluded_noncohort_engagement_rows": int(
                len(data.engagement) - len(cohort_engagement)
            ),
            "excluded_engagement_after_last_weight": int(
                engagement_after_last_weight.sum()
            ),
            "excluded_engagement_missing_date": int(
                engagement_missing_date.sum()
            ),
            "retained_engagement_rows": int(len(retained_engagement)),
        }
    )

    # Retained counts stay sparse: an absent member-event pair is not an
    # observed zero, while repeatability remains classified from the full
    # canonical engagement source above.
    member_event_counts = (
        retained_engagement.groupby(
            ["member_id", "event_type"],
            sort=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "event_count"})
        .merge(
            event_type_summary.loc[:, ["event_type", "is_repeatable"]],
            how="left",
            on="event_type",
            validate="many_to_one",
        )
    )
    member_event_counts["event_count"] = member_event_counts[
        "event_count"
    ].astype("int64")
    member_event_counts["is_repeatable"] = member_event_counts[
        "is_repeatable"
    ].astype("bool")

    # Count distinct observed behaviors for breadth and every retained event of
    # a repeatable type for volume, matching the confirmed R aggregations.
    retained_engagement["_is_repeatable"] = retained_engagement[
        "event_type"
    ].isin(repeatable_types)
    engagement_features = (
        retained_engagement.groupby("member_id", sort=False)
        .agg(
            engagement_breadth=("event_type", "nunique"),
            engagement_volume_repeatable=("_is_repeatable", "sum"),
        )
        .reset_index()
    )
    members = members.merge(
        engagement_features,
        how="left",
        on="member_id",
        validate="one_to_one",
    )
    engagement_count_columns = [
        "engagement_breadth",
        "engagement_volume_repeatable",
    ]
    members[engagement_count_columns] = (
        members[engagement_count_columns].fillna(0).astype("int64")
    )

    # Protect only the denominator at seven days. The requested formula does
    # not multiply by seven, despite the original comment calling it weekly.
    engagement_rate_denominator = members["tenure_days"].clip(
        lower=_MINIMUM_RATE_TENURE_DAYS
    )
    members["engagement_volume_repeatable_rate"] = (
        members["engagement_volume_repeatable"]
        / engagement_rate_denominator
    )

    # Apply the same inclusive last-weight cutoff to module completions while
    # retaining any earlier completion because no lower cutoff was specified.
    cohort_modules = data.module_completions.merge(
        cutoff_columns,
        how="inner",
        on="member_id",
        validate="many_to_one",
    )
    modules_after_last_weight = cohort_modules["answered_at"].gt(
        cohort_modules["last_weight_at"]
    )
    modules_missing_date = cohort_modules[
        ["answered_at", "last_weight_at"]
    ].isna().any(axis="columns")
    retained_modules = cohort_modules.loc[
        ~modules_after_last_weight & ~modules_missing_date
    ].copy()

    # A member-title pair represents one curriculum completion regardless of
    # repeated source rows or multiple recorded completion dates.
    distinct_modules = retained_modules.drop_duplicates(
        subset=["member_id", "questionnaire_title"],
        keep="first",
    ).copy()
    audit_counts.update(
        {
            "cohort_module_rows": int(len(cohort_modules)),
            "excluded_noncohort_module_rows": int(
                len(data.module_completions) - len(cohort_modules)
            ),
            "excluded_module_after_last_weight": int(
                modules_after_last_weight.sum()
            ),
            "excluded_module_missing_date": int(modules_missing_date.sum()),
            "retained_module_rows": int(len(retained_modules)),
            "duplicate_module_rows": int(
                len(retained_modules) - len(distinct_modules)
            ),
            "distinct_module_completions": int(len(distinct_modules)),
        }
    )

    # Map the nine numbered curriculum titles and the three four-title extension
    # curricula to their confirmed analysis domains.
    module_titles = distinct_modules["questionnaire_title"].astype("string")
    distinct_modules["_module_domain"] = pd.Series(
        pd.NA,
        index=distinct_modules.index,
        dtype="string",
    )
    distinct_modules.loc[
        module_titles.str.match(r"^(?:01|0[5-9]|1[0-2])_", na=False),
        "_module_domain",
    ] = "core"
    distinct_modules.loc[
        module_titles.str.match(r"^MINDSET W0[1-4]:", na=False),
        "_module_domain",
    ] = "mindset"
    distinct_modules.loc[
        module_titles.str.match(r"^NUTRITION W0[1-4]:", na=False),
        "_module_domain",
    ] = "nutrition"
    distinct_modules.loc[
        module_titles.str.match(
            r"^PHYSICAL ACTIVITY W0[1-4]:",
            na=False,
        ),
        "_module_domain",
    ] = "physical_activity"
    audit_counts["unmapped_module_completions"] = int(
        distinct_modules["_module_domain"].isna().sum()
    )

    # Aggregate de-duplicated titles within member and curriculum domain, then
    # materialize zero counts for members without completions in a domain.
    module_counts = (
        distinct_modules.dropna(subset=["_module_domain"])
        .groupby(["member_id", "_module_domain"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for domain in MODULE_DOMAIN_DENOMINATORS:
        if domain not in module_counts.columns:
            module_counts[domain] = 0
    module_counts = module_counts.loc[:, list(MODULE_DOMAIN_DENOMINATORS)]
    module_counts.columns = [
        f"module_{domain}_count" for domain in module_counts.columns
    ]
    members = members.merge(
        module_counts.reset_index(),
        how="left",
        on="member_id",
        validate="one_to_one",
    )
    module_count_columns = [
        f"module_{domain}_count" for domain in MODULE_DOMAIN_DENOMINATORS
    ]
    members[module_count_columns] = (
        members[module_count_columns].fillna(0).astype("int64")
    )

    # Normalize each domain by its fixed number of available titles so features
    # represent curriculum proportions rather than incomparable raw counts.
    normalized_module_columns: list[str] = []
    for domain, denominator in MODULE_DOMAIN_DENOMINATORS.items():
        normalized_column = f"module_{domain}"
        count_column = f"{normalized_column}_count"
        members[normalized_column] = members[count_column] / denominator
        normalized_module_columns.append(normalized_column)

    # Preserve the user's exact row-mean formula across the four normalized
    # domains; diagnostics will later flag its deterministic collinearity.
    members["module_mean"] = members[normalized_module_columns].mean(
        axis="columns"
    )

    return FeatureResult(
        member_features=members,
        event_type_summary=event_type_summary,
        member_event_counts=member_event_counts,
        audit_counts=audit_counts,
    )
