# 9amHealth Scientific Analysis Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current Python package to produce validated scientific tables, post-LASSO HC3 estimates, diagnostics, and standalone figures for the confirmed 534-member analysis.

**Architecture:** The existing ingestion, modeling, penalization, and resampling modules remain the analytical foundation. Cohort and feature modules expose the confirmed two-group population and internal event-level aggregates; focused final-model, reporting, and visualization modules transform those objects into aggregate artifacts. The pipeline selects the base-family winner, locks stable covariates, performs the unpenalized refit, and writes deterministic tables and figures through the existing CLI.

**Tech Stack:** Python 3.11+, NumPy, pandas, statsmodels, matplotlib, standard-library `unittest`, pytest, and Ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-scientific-analysis-reporting-design.md`

## Global Constraints

- Main analyses retain only `Active GLP-1 for Weight-loss` and `Coaching Only`; Coaching Only is the reference.
- Expected primary analysis sample is 534 members and 1,068 longitudinal rows.
- Primary outcome is continuous percentage weight loss; secondary success is at least 5% weight loss.
- No small-cell suppression is applied to approved aggregate analytical outputs, but member identifiers are never exported.
- Base-family ranking uses only common-target held-out raw last-weight RMSE and MAE.
- Mean MAE breaks a mean-RMSE tie; an exact tie on both metrics prefers `percentage_loss_ols`.
- The lower grouped-CV MSE chooses the module-mean or module-domain LASSO specification; an exact tie prefers the mean specification for parsimony.
- Stable LASSO selection within that specification uses frequency at least 0.75; p-values do not select terms.
- HC3 post-selection inference is labeled `conditional_exploratory`.
- The primary locked model keeps the confirmed base/candidate specification; a separate common-support duration sensitivity adds unpenalized linear and quadratic `weight_days` terms.
- Activity tests require reach in at least 30 members, use standardized `log1p(event_count)`, adjust for baseline, member type, and linear/quadratic `weight_days`, and apply one Benjamini--Hochberg family.
- The user-confirmed tenure-normalized engagement rate remains primary; an interval-normalized rate is descriptive sensitivity only.
- Every implementation task that writes a scientific CSV uses the exact ordered schemas in the design spec.
- Use lowercase snake-case Python filenames with the `nineam_` prefix and keep `scripts/nineam_run_analysis.py` as the entry point.
- Every function receives a concise contract docstring and every non-obvious analytical block receives an inline comment.
- Source extracts remain read-only; outputs go to `outputs/tables` and `outputs/figures`.
- This directory has no Git repository, so task gates record tests and review evidence without commit commands.

---

### Task 1: Lock the primary cohort and member-level scientific features

**Files:**
- Modify: `src/nineam_health_analysis/nineam_cohort_selection.py`
- Modify: `src/nineam_health_analysis/nineam_feature_engineering.py`
- Modify: `tests/test_nineam_data_and_cohort.py`
- Modify: `tests/test_nineam_feature_engineering.py`
- Modify: `tests/test_nineam_source_benchmarks.py`
- Modify: `tests/test_nineam_analysis_pipeline.py`

**Interfaces:**
- Consumes: `CaseStudyData` from `nineam_data_loading.py`.
- Produces: `PRIMARY_MEMBER_TYPES`, `REFERENCE_MEMBER_TYPE`, the existing 633-row complete-pair `build_analysis_cohort(data)`, `restrict_to_primary_member_types(cohort) -> CohortResult` with 534 members, `absolute_weight_loss`, `weight_loss_success_5pct`, and `FeatureResult.member_event_counts` with columns `member_id`, `event_type`, `event_count`, and `is_repeatable`.

- [ ] **Step 0: Install the existing development test tools**

Run: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`

Expected: pytest and Ruff are available in the project virtual environment.

- [ ] **Step 1: Write failing cohort and feature tests**

```python
def test_primary_cohort_keeps_only_confirmed_comparable_types(self) -> None:
    complete_pair = build_analysis_cohort(self.case_study_data)
    cohort = restrict_to_primary_member_types(complete_pair)
    self.assertEqual(set(cohort.members["member_type"]), set(PRIMARY_MEMBER_TYPES))
    self.assertEqual(cohort.audit_counts["pre_member_type_restriction_members"], 5)
    self.assertEqual(cohort.audit_counts["excluded_nonprimary_member_type"], 3)

def test_weight_loss_fields_use_confirmed_directions(self) -> None:
    member = build_analysis_cohort(self.case_study_data).members.iloc[0]
    self.assertEqual(member["absolute_weight_loss"], 10.0)
    self.assertEqual(member["percentage_loss"], 5.0)
    self.assertTrue(member["weight_loss_success_5pct"])

def test_member_event_counts_include_zero_free_observed_type_counts(self) -> None:
    result = build_member_features(self.data, self.cohort)
    row = result.member_event_counts.query("member_id == 'A' and event_type == 'read'")
    self.assertEqual(int(row.iloc[0]["event_count"]), 2)
```

- [ ] **Step 2: Run focused tests and observe the missing contracts**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_data_and_cohort.py tests/test_nineam_feature_engineering.py -q`

Expected: FAIL because primary member-type restriction, response fields, and `member_event_counts` do not exist.

- [ ] **Step 3: Implement the separate cohort restriction and internal event aggregates**

```python
PRIMARY_MEMBER_TYPES = (
    "Active GLP-1 for Weight-loss",
    "Coaching Only",
)
REFERENCE_MEMBER_TYPE = "Coaching Only"

# Preserve the complete-pair cohort, then create a new restricted result so
# cohort flow distinguishes eligibility from condition comparability.
def restrict_to_primary_member_types(cohort: CohortResult) -> CohortResult:
    members = cohort.members.loc[
        cohort.members["member_type"].isin(PRIMARY_MEMBER_TYPES)
    ].copy()
    audit_counts = dict(cohort.audit_counts)
    audit_counts["pre_member_type_restriction_members"] = len(cohort.members)
    audit_counts["excluded_nonprimary_member_type"] = len(cohort.members) - len(members)
    audit_counts["included_members"] = len(members)
    return CohortResult(
        members=members,
        long_weights=_build_long_weights(members),
        audit_counts=audit_counts,
    )
```

Derive `absolute_weight_loss` and `weight_loss_success_5pct` in
`build_analysis_cohort`, then rebuild long rows after the primary restriction.

Build long-format member-event counts from the already time-censored engagement rows and attach them to the frozen `FeatureResult`; do not serialize member IDs.

Relabel the synthetic pipeline fixture from `Type A`/`Type B` to `Coaching
Only`/`Active GLP-1 for Weight-loss` so the existing integration tests continue
to exercise two retained conditions after cohort restriction.

- [ ] **Step 4: Run focused tests and source goldens**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_data_and_cohort.py tests/test_nineam_feature_engineering.py tests/test_nineam_source_benchmarks.py -q`

Expected: PASS with exact source assertions for 534 members, 1,068 long rows, 331 GLP-1 Weight-loss members, 203 Coaching Only members, and 99 member-type exclusions.

### Task 2: Fit and diagnose the locked post-LASSO model

**Files:**
- Create: `src/nineam_health_analysis/nineam_final_model.py`
- Create: `tests/test_nineam_final_model.py`
- Modify: `src/nineam_health_analysis/__init__.py`
- Modify: `src/nineam_health_analysis/nineam_statistical_models.py`
- Modify: `src/nineam_health_analysis/nineam_resampling.py`
- Modify: `tests/test_nineam_statistical_models.py`
- Modify: `tests/test_nineam_resampling.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: member features, the two LASSO selection tables, and fold-level base-model comparison.
- Produces: `LockedCandidateSelection`, `LockedModelResult`, `choose_base_model_winner(base_model_cv)`, `select_locked_candidates(mean_selection, domain_selection, threshold)`, and `fit_locked_percentage_model(member_features, selection, winning_base_model)`.
- `LockedCandidateSelection` exposes `module_spec`, `candidates`, `cv_mean_mse`, `selection_threshold`, and candidate-frequency provenance.
- `LockedModelResult` exposes `primary_model_id`, `winning_base_model`, `winning_module_spec`, `selected_candidates`, combined primary/sensitivity `coefficient_table`, `fit_statistics`, `diagnostic_summary`, and internal `diagnostic_points`.

- [ ] **Step 1: Write failing tests from hand-derived regression fixtures**

```python
def test_choose_winner_uses_mean_rmse_then_mae(self) -> None:
    comparison = pd.DataFrame({
        "model": ["percentage_loss_ols", "percentage_loss_ols", "log_compound_symmetry_gls", "log_compound_symmetry_gls"],
        "rmse": [2.0, 4.0, 4.0, 4.0],
        "mae": [2.0, 3.0, 2.0, 2.0],
    })
    self.assertEqual(choose_base_model_winner(comparison), "percentage_loss_ols")

def test_choose_winner_prefers_primary_outcome_model_on_exact_tie(self) -> None:
    comparison = pd.DataFrame({
        "model": ["log_compound_symmetry_gls", "percentage_loss_ols"],
        "rmse": [3.0, 3.0],
        "mae": [2.0, 2.0],
    })
    self.assertEqual(choose_base_model_winner(comparison), "percentage_loss_ols")

def test_locked_model_uses_coaching_reference_and_hc3(self) -> None:
    selection = LockedCandidateSelection(
        module_spec="mean",
        candidates=("engagement_volume_repeatable",),
        cv_mean_mse=4.0,
        selection_threshold=0.75,
        selection_frequencies={"engagement_volume_repeatable": 0.90},
    )
    result = fit_locked_percentage_model(self.members, selection, "percentage_loss_ols")
    coefficient = result.coefficient_table.set_index("term")
    self.assertIn("member_type[Active GLP-1 for Weight-loss]", coefficient.index)
    self.assertEqual(set(coefficient["covariance_estimator"]), {"HC3"})
    self.assertEqual(set(coefficient["inference_status"]), {"conditional_exploratory"})

def test_locked_model_includes_separate_duration_sensitivity(self) -> None:
    result = fit_locked_percentage_model(self.members, self.selection, "percentage_loss_ols")
    self.assertEqual(
        set(result.fit_statistics["model_id"]),
        {"locked_percentage_loss_primary", "locked_percentage_loss_duration_sensitivity"},
    )
    sensitivity = result.coefficient_table.query(
        "model_id == 'locked_percentage_loss_duration_sensitivity'"
    )
    self.assertIn("weight_days", set(sensitivity["term"]))
    self.assertIn("weight_days_squared", set(sensitivity["term"]))

def test_confirmed_models_use_coaching_as_reference_when_present(self) -> None:
    percentage = fit_percentage_loss_ols(self.members)
    design = build_percentage_penalized_design(self.members)
    self.assertEqual(percentage.schema.reference_member_type, "Coaching Only")
    self.assertNotIn("member_type[Coaching Only]", design.base_names)
```

- [ ] **Step 2: Run the new test and observe the missing module**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_final_model.py -q`

Expected: FAIL with `ModuleNotFoundError` for `nineam_final_model`.

- [ ] **Step 3: Implement winner selection, locked candidate resolution, HC3 refit, and diagnostics**

```python
@dataclass(frozen=True, slots=True)
class LockedCandidateSelection:
    module_spec: str
    candidates: tuple[str, ...]
    cv_mean_mse: float
    selection_threshold: float
    selection_frequencies: Mapping[str, float]

@dataclass(frozen=True, slots=True)
class LockedModelResult:
    primary_model_id: str
    winning_base_model: str
    winning_module_spec: str
    selected_candidates: tuple[str, ...]
    coefficient_table: pd.DataFrame
    fit_statistics: pd.DataFrame
    diagnostic_summary: pd.DataFrame
    diagnostic_points: pd.DataFrame

def fit_locked_percentage_model(
    member_features: pd.DataFrame,
    selection: LockedCandidateSelection,
    winning_base_model: str,
) -> LockedModelResult:
    """Refit primary and duration-sensitivity equations with HC3 inference."""
```

Construct a numeric primary design explicitly: intercept, first weight, an indicator for `Active GLP-1 for Weight-loss` versus `Coaching Only`, and selected continuous or treatment-coded sex candidates. Fit `statsmodels.api.OLS(outcome, design).fit(cov_type="HC3")`. Fit a second model on the member-type common observed `weight_days` range with centered linear and quadratic duration terms appended as unpenalized sensitivity adjustments. Return estimate, HC3 standard error, statistic, p-value, 95% confidence limits, units, role, reference, n, selection provenance, and inference label for both model identifiers. Return R-squared, adjusted R-squared, residual RMSE, negative-two-log-likelihood, AIC, BIC, condition number, maximum leverage, and maximum Cook's distance. Keep fitted/residual/standardized-residual/leverage/Cook's values internal to the aggregate result and omit member IDs.

When `Coaching Only` is observed, use it as the reference in longitudinal GLS,
percentage-loss OLS, and both penalized designs. Preserve the existing sorted
reference fallback for generic synthetic factors that do not contain that
level.

Add `statsmodels>=0.14,<1` to runtime dependencies and reinstall the editable
package before running the green test.

`select_locked_candidates` chooses the complete module specification by its
selected-penalty `cv_mean_mse`, uses `mean` as the deterministic tie-break, and
then returns every candidate in that specification with selection frequency at
least the threshold. It never unions module mean with its component domains.

- [ ] **Step 4: Run model tests and regression model tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_final_model.py tests/test_nineam_statistical_models.py -q`

Expected: PASS; changing the reference group, covariance estimator, stable threshold, or coefficient order must break at least one behavior test.

### Task 3: Build scientific descriptive and evidence tables

**Files:**
- Create: `src/nineam_health_analysis/nineam_reporting.py`
- Create: `tests/test_nineam_reporting.py`
- Modify: `src/nineam_health_analysis/__init__.py`

**Interfaces:**
- Consumes: `CaseStudyData`, `CohortResult`, `FeatureResult`, base-model CV rows, both LASSO tables, and `LockedModelResult`.
- Produces: `ReportingResult(tables: Mapping[str, pd.DataFrame])` and `build_reporting_tables(data, cohort, features, base_model_cv, lasso_mean_selection, lasso_domain_selection, locked_model) -> ReportingResult` with the exact approved `nineam_*.csv` keys.

- [ ] **Step 1: Write failing tests for scientific estimands and output schemas**

```python
def test_wilson_interval_matches_hand_checked_result(self) -> None:
    lower, upper = wilson_confidence_interval(5, 10)
    self.assertAlmostEqual(lower, 0.236593, places=6)
    self.assertAlmostEqual(upper, 0.763407, places=6)

def test_outcome_table_reports_overall_and_two_member_types(self) -> None:
    tables = build_reporting_tables(self.inputs).tables
    outcomes = tables["nineam_outcomes_by_member_type.csv"]
    self.assertEqual(set(outcomes["scope"]), {"Overall", *PRIMARY_MEMBER_TYPES})
    response = outcomes.query("outcome == 'weight_loss_success_5pct'")
    self.assertTrue(response["ci_method"].eq("Wilson").all())

def test_activity_associations_are_fdr_adjusted_and_exploratory(self) -> None:
    activity = build_reporting_tables(self.inputs).tables["nineam_engagement_activity_summary.csv"]
    self.assertIn("fdr_adjusted_p_value", activity.columns)
    self.assertTrue(activity["interpretation"].eq("exploratory_association").all())

def test_diagnostic_source_rows_are_aggregated(self) -> None:
    diagnostics = build_reporting_tables(self.inputs).tables["nineam_model_diagnostics.csv"]
    self.assertTrue(diagnostics["bin_count"].ge(2).all())
    self.assertNotIn("member_id", diagnostics.columns)
```

- [ ] **Step 2: Run reporting tests and observe the missing module**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_reporting.py -q`

Expected: FAIL with `ModuleNotFoundError` for `nineam_reporting`.

- [ ] **Step 3: Implement the complete aggregate table builder**

```python
@dataclass(frozen=True, slots=True)
class ReportingResult:
    tables: Mapping[str, pd.DataFrame]

def build_reporting_tables(
    data: CaseStudyData,
    cohort: CohortResult,
    features: FeatureResult,
    base_model_cv: pd.DataFrame,
    lasso_mean_selection: pd.DataFrame,
    lasso_domain_selection: pd.DataFrame,
    locked_model: LockedModelResult,
) -> ReportingResult:
    """Create the aggregate scientific tables used by leadership visuals."""
```

Create these exact tables: `nineam_cohort_flow.csv`, `nineam_sample_characteristics.csv`, `nineam_outcomes_by_member_type.csv`, `nineam_engagement_by_member_type.csv`, `nineam_engagement_activity_summary.csv`, `nineam_modules_by_member_type.csv`, `nineam_base_model_comparison_summary.csv`, `nineam_lasso_mean_selection.csv`, `nineam_lasso_domain_selection.csv`, `nineam_locked_model_coefficients_hc3.csv`, `nineam_locked_model_fit_statistics.csv`, `nineam_model_diagnostics.csv`, `nineam_hypothesis_evidence.csv`, `nineam_findings_and_implications.csv`, and `nineam_limitations.csv`. Bin or quantile-summarize fitted/residual, Q-Q, leverage, and Cook's diagnostics with row counts of at least two; never place raw diagnostic points in `ReportingResult`.

For activity associations, export all 19 types, zero-fill each 534-member count
vector, and test only types reaching at least 30 members. Use a standardized
`log1p(event_count)` predictor in HC3 percentage-loss OLS adjusted for first
weight, Coaching-referenced member type, and centered linear/quadratic
`weight_days`; apply one Benjamini--Hochberg correction across that eligible
family. Hypothesis and implication rows must identify the question,
prespecified expectation, evidence artifact, result status, noncausal
interpretation, proposed leadership action, and future randomized-pilot KPI.

- [ ] **Step 4: Run reporting and feature tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_reporting.py tests/test_nineam_feature_engineering.py -q`

Expected: PASS with no exported `member_id` column and no category suppression labels.

### Task 4: Render deterministic table-sourced scientific figures

**Files:**
- Create: `src/nineam_health_analysis/nineam_visualizations.py`
- Create: `tests/test_nineam_visualizations.py`
- Modify: `src/nineam_health_analysis/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: aggregate reporting tables and a destination directory.
- Produces: `write_analysis_figures(tables, output_dir) -> tuple[Path, ...]` and both PNG/SVG versions of nine named figures.

- [ ] **Step 1: Write failing visual artifact tests**

```python
def test_writer_creates_png_svg_and_source_csv_for_every_figure(self) -> None:
    paths = write_analysis_figures(self.tables, self.output_dir)
    self.assertEqual(len(paths), 18)
    self.assertTrue((self.output_dir / "nineam_weight_loss_by_member_type.png").is_file())
    self.assertTrue((self.output_dir / "nineam_weight_loss_by_member_type.svg").is_file())

def test_svg_output_is_deterministic(self) -> None:
    first = write_analysis_figures(self.tables, self.first_dir)
    second = write_analysis_figures(self.tables, self.second_dir)
    self.assertEqual(Path(first[1]).read_bytes(), Path(second[1]).read_bytes())
```

- [ ] **Step 2: Run visualization tests and observe the missing module**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_visualizations.py -q`

Expected: FAIL with `ModuleNotFoundError` for `nineam_visualizations`.

- [ ] **Step 3: Implement nine plots with deterministic metadata**

```python
FIGURE_STEMS = (
    "nineam_cohort_flow",
    "nineam_weight_loss_by_member_type",
    "nineam_responder_rate_by_member_type",
    "nineam_engagement_patterns",
    "nineam_module_completion",
    "nineam_base_model_performance",
    "nineam_lasso_stability",
    "nineam_locked_model_coefficients",
    "nineam_model_diagnostics",
)

def write_analysis_figures(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render aggregate scientific figures from validated source tables."""
```

Use a non-interactive Matplotlib backend, fixed dimensions, accessible colors, explicit units and confidence intervals, `svg.hashsalt`, and save metadata with no timestamp. Close every figure after writing. Visuals must consume reporting tables; they must not recompute scientific estimands.

Add `matplotlib>=3.9,<4` to runtime dependencies and reinstall the editable
package before running the green test.

- [ ] **Step 4: Run visualization tests twice**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_visualizations.py -q`

Expected: PASS twice with identical PNG and SVG hashes for identical table inputs.

### Task 5: Integrate dependencies, pipeline artifacts, and CLI output directories

**Files:**
- Modify: `src/nineam_health_analysis/nineam_analysis_pipeline.py`
- Modify: `src/nineam_health_analysis/__init__.py`
- Modify: `scripts/nineam_run_analysis.py`
- Modify: `tests/test_nineam_analysis_pipeline.py`

**Interfaces:**
- `run_analysis(data_dir, config=None) -> AnalysisResult` returns base outputs plus locked model and reporting tables.
- `write_analysis_outputs(result, output_dir) -> tuple[Path, ...]` validates closed schemas, stages every CSV/JSON payload before rollback-safe replacement in `tables/`, and writes deterministic PNG/SVG artifacts to `figures/` only after table publication succeeds.

- [ ] **Step 1: Add failing integration tests**

```python
def test_pipeline_uses_primary_cohort_and_exposes_scientific_artifacts(self) -> None:
    result = run_analysis(self.data_dir, config=self.fast_config)
    self.assertEqual(result.metadata["included_members"], 534)
    self.assertEqual(result.locked_model.winning_base_model, "percentage_loss_ols")
    self.assertIn("nineam_sample_characteristics.csv", result.reporting_tables)

def test_writer_uses_tables_and_figures_subdirectories(self) -> None:
    written = write_analysis_outputs(self.result, self.output_dir)
    self.assertTrue((self.output_dir / "tables" / "nineam_locked_model_coefficients_hc3.csv").is_file())
    self.assertTrue((self.output_dir / "figures" / "nineam_locked_model_coefficients.png").is_file())
    self.assertFalse(any("member_id" in path.read_text(encoding="utf-8") for path in written if path.suffix == ".csv"))
```

- [ ] **Step 2: Run pipeline tests and observe missing result fields**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_analysis_pipeline.py -q`

Expected: FAIL because the result does not yet include locked/reporting artifacts and the writer has no nested figure output.

- [ ] **Step 3: Integrate the analytical sequence and verify dependencies**

```toml
dependencies = [
    "numpy>=1.26,<3",
    "pandas>=2.1,<4",
    "statsmodels>=0.14,<1",
    "matplotlib>=3.9,<4",
]
```

After both LASSO summaries exist, resolve stable candidates, aggregate the CV winner, require `percentage_loss_ols` for the current locked HC3 path, fit the locked model, build reporting tables, and serialize all outputs. Metadata must record the 633-member pre-restriction eligible cohort, 99 nonprimary exclusions, 534 included members, Coaching reference, 5% threshold, winner rule, HC3 covariance, and conditional-exploratory inference label.

- [ ] **Step 4: Install the editable package and run integration tests**

Run: `.\.venv\Scripts\python.exe -m pip install -e .[dev]`

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_nineam_analysis_pipeline.py tests/test_nineam_final_model.py tests/test_nineam_reporting.py tests/test_nineam_visualizations.py -q`

Expected: PASS and no schema, identifier, output-boundary, or determinism failure.

### Task 6: Update documentation, source benchmarks, and full-run verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_nineam_source_benchmarks.py`
- Create: `outputs/tables/*.csv` through the CLI
- Create: `outputs/figures/*.png` through the CLI
- Create: `outputs/figures/*.svg` through the CLI

**Interfaces:**
- The documented command remains `.\.venv\Scripts\python.exe scripts\nineam_run_analysis.py --data-dir "B:\07_Luis\case-study-9amhealth_CLEAN-SLATE\data"`; metadata is written to `outputs/nineam_analysis_metadata.json`.
- The source benchmark records exact audited n=534 results after one successful full run.

- [ ] **Step 1: Run the full analysis and inspect the first artifact set**

Run: `.\.venv\Scripts\python.exe scripts\nineam_run_analysis.py --data-dir "B:\07_Luis\case-study-9amhealth_CLEAN-SLATE\data"`

Expected: exit code 0; fifteen scientific CSV tables and eighteen image files are present; every table reconciles to n=534 where applicable.

- [ ] **Step 2: Lock source benchmarks and update README interpretation guidance**

```python
self.assertEqual(len(features.member_features), 534)
self.assertEqual(cohort.audit_counts["excluded_nonprimary_member_type"], 99)
self.assertEqual(cohort.audit_counts["included_members"], 534)
self.assertEqual(result.locked_model.coefficient_table["covariance_estimator"].unique().tolist(), ["HC3"])
```

Document data definitions, exact cohort flow, event/module assumptions, two base families, common-target winner rule, LASSO stability, locked HC3 inference, output dictionary, run command, and the observational/post-selection limitations.

- [ ] **Step 3: Run style, compilation, and the complete regression suite**

Run: `.\.venv\Scripts\python.exe -m ruff check src scripts tests`

Run: `.\.venv\Scripts\python.exe -m compileall -q src scripts tests`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: Ruff clean, compilation exit code 0, and all tests pass.

- [ ] **Step 4: Prove deterministic reruns**

Run the CLI into two empty temporary output directories with the same configuration. Compare every CSV, JSON, PNG, and SVG by SHA-256.

Expected: identical filename sets and identical hashes for every artifact.

- [ ] **Step 5: Perform statistical and code review gates**

Review the exact source results against the approved spec: cohort denominators, reference coding, winner rule, stable-candidate provenance, HC3 labeling, activity FDR, figure/table consistency, no causal wording, no member IDs, and all requested limitations. Resolve every blocking finding and rerun the smallest affected tests, then the complete suite once.
