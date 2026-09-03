# 9amHealth Python Analysis Design

## Goal

Create a code-only, reproducible Python package containing the analytical code used for the 9amHealth case study. The package must run against the existing `data` directory without copying or modifying source data.

## Scope

- Load and validate the four supplied UTF-16LE, tab-delimited files.
- Construct the confirmed 633-member cohort.
- Reproduce engagement and module feature engineering.
- Fit repeated-measures GLS and percentage-loss OLS models.
- Compare held-out prediction on a common last-weight target.
- Fit partially penalized LASSO models and report selection stability.
- Write deterministic machine-readable results.
- Include unit and source-data benchmark tests.

Presentation styling, charts, slide generation, and business recommendations are out of scope.

## Naming

The Nomenclature section on pages 108–109 of the brand guide uses
underscore-delimited filename components. Literal names beginning with `9am_`
are not valid importable Python module names. Therefore, code filenames use the
compatible lowercase prefix `nineam_`, descriptive snake-case components, and
a lowercase extension, for example `nineam_data_loading.py`. This preserves
the underscore-separated nomenclature while following Python module rules.

## Data and cohort contracts

- Source files are read with `encoding="utf-16"` and `sep="\t"`.
- Trailing all-empty columns are removed.
- Source columns are validated and renamed to snake case.
- Keep subscription statuses `ACTIVE` and `FINISHED`.
- Require positive first and last weights and `weight_days > 0`.
- Expected final cohort: 633 members and 1,266 long-format rows.
- Percentage loss is `100 * (first_weight - last_weight) / first_weight`.

## Feature contracts

- Retain engagement and module records dated on or before each member's last weight date.
- Repeatable engagement types are classified from the full canonical
  engagement table, matching the supplied R/SQL, and satisfy both
  `events / distinct_members >= 2` and `distinct_members >= 30`. That fixed
  classification is then applied to retained cohort-window events.
- `engagement_breadth` is the distinct event-type count.
- `engagement_volume_repeatable` is the count of events whose type is repeatable.
- `engagement_volume_repeatable_rate` is exactly `volume / max(tenure_days, 7)`. Despite the original comment, this is not multiplied by seven.
- De-duplicate module completions by member and title before aggregation.
- Valid domain identifiers are the supplied core titles 01 and 05--12 and
  W01--W04 for mindset, nutrition, and physical activity; unknown titles are
  audited as unmapped and do not enter fixed-denominator proportions.
- Normalize module domains by the available title counts: core 9, mindset 4, nutrition 4, physical activity 4.
- `module_mean` is the row mean of the four normalized domain features. It is retained even though it is exactly collinear with those features because it was in the requested candidate list; diagnostics must flag the dependency.

## Model contracts

- Base longitudinal model: weight or log weight as a function of time and member type.
- Time is coded 0 for first weight and 1 for last weight.
- Covariance options: IID, diagonal, compound symmetry, and unstructured.
- Fit by maximum likelihood. Lognormal likelihoods include the raw-scale Jacobian adjustment.
- The repeated-measures model is described as marginal GLS, not as a
  random-effects or mixed-effects fit; its covariance structures are residual
  covariance models.
- A time-by-member-type interaction is a labeled sensitivity model, not the base specification.
- Percentage-loss OLS uses first weight and member type as unpenalized base predictors.
- Longitudinal and percentage-model likelihood criteria are not used as a direct cross-outcome contest. Repeated member-level validation compares predicted last-weight RMSE and MAE.
- LASSO leaves base terms unpenalized, standardizes penalized terms using
  training data only, and records selection frequency across member-level
  resamples. Selection frequency is not statistical significance.
- For the longitudinal path, accumulated follow-up features enter through
  time-by-feature terms and member blocks are whitened with a covariance fit to
  the supplied training sample. Matrix-only lambda CV cannot refit that
  covariance inside its inner folds, so this path is labeled retrospective
  two-stage exploration, not unbiased nested-CV performance or joint penalized
  mixed-effects ML.
- Run `module_mean` and the four module-domain features in separate candidate
  specifications because including their exact linear dependency together
  makes individual LASSO selections non-unique.

## Output disclosure contract

- Output tables use closed, ordered schemas and contain no member-level rows.
- Subgroup diagnostics report counts only. Levels with fewer than 10 members
  are combined under one suppressed label; modeling inputs are not recoded.
- The writer rejects exact source member IDs in output cells and metadata,
  rejects output locations inside the source-data directory, and validates all
  payloads before publication.
- All eight artifacts are staged before a rollback-safe commit that uses
  per-file atomic replacements. The eight replacements are sequential, so a
  concurrent reader could briefly observe a mixed set. If a commit fails,
  restoration is attempted and temporary stages are removed. Any prior backup
  that cannot be restored remains on disk and its recovery path is reported.

## Dependencies and verification

- Runtime: Python 3.11+, NumPy, and pandas.
- Every function must have a concise explanation at its definition covering
  purpose, inputs, return value, and any important statistical intent.
- Every non-obvious analytical block must also have an inline comment
  explaining its purpose or statistical meaning; function explanations alone
  are not sufficient.
- Tests: standard-library `unittest`; compatible with pytest when installed.
- Verification goldens include cohort and feature counts, raw UN AIC, log-CS AIC, log-CS interaction AIC, and percentage-loss OLS R-squared.
- Random operations use an explicit seed.
