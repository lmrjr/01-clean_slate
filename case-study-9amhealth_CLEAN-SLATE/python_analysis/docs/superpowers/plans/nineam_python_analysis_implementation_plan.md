# 9amHealth Python Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python package that reproduces the case-study analysis code and verified numerical benchmarks.

**Architecture:** A `src` package separates ingestion, cohort logic, feature engineering, statistical models, penalization, resampling, and orchestration. A thin executable script invokes the package. Tests exercise behavior with hand-built fixtures and verify source-data integration goldens.

**Tech Stack:** Python 3.11+, NumPy, pandas, standard-library `unittest`, optional Ruff and pytest.

**Spec:** `docs/superpowers/specs/nineam_python_analysis_design.md`

## Global Constraints

- Use lowercase snake-case Python filenames with the `nineam_` prefix.
- Keep source data read-only and outside the package.
- Use only NumPy, pandas, and the Python standard library at runtime.
- Give every function a concise definition-level explanation of its purpose,
  inputs, return value, and important statistical intent.
- Add concise inline comments before every non-obvious filtering,
  transformation, matrix, likelihood, and resampling block.
- Write tests before production functions and observe the expected failure.
- Preserve the user's formulas exactly, including the engagement-rate denominator and module row mean.
- Use member-level splits for every resampling operation.
- Do not compare likelihood criteria across different outcome definitions to declare a winner.

---

### Task 1: Data loading and cohort construction

**Files:**
- Create: `src/nineam_health_analysis/nineam_data_loading.py`
- Create: `src/nineam_health_analysis/nineam_cohort_selection.py`
- Test: `tests/test_nineam_data_and_cohort.py`

**Interfaces:**
- Produces `CaseStudyData`, `CohortResult`, `load_case_study_data(data_dir)`, and `build_analysis_cohort(data)`.
- Cohort output supplies canonical member rows to every later task.

- [x] Write fixture tests for UTF-16/tab loading, schema failures, status exclusion, missing-weight exclusion, and zero-interval exclusion.
- [x] Run the test module and verify imports fail because production modules do not exist.
- [x] Implement schema validation, canonical renaming, date parsing, and cohort audit counts.
- [x] Run the test module and verify all tests pass.

### Task 2: Feature engineering

**Files:**
- Create: `src/nineam_health_analysis/nineam_feature_engineering.py`
- Test: `tests/test_nineam_feature_engineering.py`

**Interfaces:**
- Consumes canonical data and cohort rows.
- Produces `FeatureResult`, `classify_repeatable_events`, and `build_member_features`.

- [x] Write tests for full-source repeatable-event thresholds, last-weight
  cutoff, the protected seven-day denominator, module de-duplication, bounded
  domain identifiers, zero-filled members, domain proportions, and
  `module_mean`.
- [x] Run the tests and verify the missing production module causes the expected failure.
- [x] Implement the minimum aggregation and audit logic needed by the tests.
- [x] Run the tests and verify all pass.

### Task 3: Base statistical models

**Files:**
- Create: `src/nineam_health_analysis/nineam_statistical_models.py`
- Test: `tests/test_nineam_statistical_models.py`

**Interfaces:**
- Produces `fit_longitudinal_gls`, `fit_percentage_loss_ols`, model-result dataclasses, and prediction functions.

- [x] Write synthetic behavior tests for design matrices, covariance constraints, likelihood parameter counts, lognormal Jacobian, OLS coefficients, and conditional follow-up prediction.
- [x] Run the tests and verify the missing module fails.
- [x] Implement maximum-likelihood GLS and OLS using NumPy linear algebra.
- [x] Run the tests and verify all pass.

### Task 4: LASSO and member-level resampling

**Files:**
- Create: `src/nineam_health_analysis/nineam_penalized_models.py`
- Create: `src/nineam_health_analysis/nineam_resampling.py`
- Test: `tests/test_nineam_penalized_models.py`
- Test: `tests/test_nineam_resampling.py`

**Interfaces:**
- Produces partially penalized LASSO fits, lambda selection, stability frequencies, collinearity diagnostics, and repeated member-level model comparison.

- [x] Write tests showing lambda-max zeroes penalized terms, unpenalized terms survive, a synthetic signal is selected, held-out scaling is train-only, folds keep member rows together, and seeds are deterministic.
- [x] Run the tests and verify missing modules fail.
- [x] Implement residualized coordinate descent, grouped lambda tuning,
  stability selection, separate module-mean/domain specifications, paired
  last-weight prediction comparison, and a clearly labeled retrospective
  fixed-prewhitening longitudinal design.
- [x] Run both test modules and verify all pass.

### Task 5: Pipeline, documentation, and source-data benchmarks

**Files:**
- Create: `src/nineam_health_analysis/nineam_analysis_pipeline.py`
- Create: `src/nineam_health_analysis/__init__.py`
- Create: `scripts/nineam_run_analysis.py`
- Create: `tests/test_nineam_source_benchmarks.py`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `.gitignore`

**Interfaces:**
- The CLI consumes a data directory and output directory and writes deterministic CSV and JSON summaries.

- [x] Write integration assertions for the 633-member cohort, feature totals, three GLS AIC values, and percentage-model R-squared.
- [x] Run the benchmark test and verify the missing pipeline fails.
- [x] Implement orchestration, privacy-safe transactional output writers,
  package metadata, and concise technical documentation.
- [x] Run all tests, compile all source files, and execute the CLI against the supplied data.
- [x] Run the CLI a second time and verify deterministic output equality.
