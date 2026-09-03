# 9amHealth case-study Python analysis

This code-only project reproduces the exploratory case-study analysis with a
small, testable Python package. It loads the supplied source tables, constructs
the confirmed analytical cohort, engineers engagement and curriculum features,
fits the two base model families, evaluates partially penalized LASSO models
with member-level resampling, refits the locked percentage-loss model with HC3
inference, and produces the aggregate tables and figures needed for the case-
study presentation.

## Source-data boundary

The four source files remain in the parent project's `data` directory. The
analysis opens them for reading and never edits, copies, or overwrites them.
Although their extensions are `.csv`, the supplied files are UTF-16,
tab-delimited tables; that source-specific handling is isolated in
`nineam_data_loading.py`.

Generated results belong in `outputs/`. The approved aggregate scientific
tables do not apply small-cell suppression, but member identifiers and row-level
diagnostic points are never exported. Before publication, the writer validates
closed schemas, rejects source member IDs, stages the artifacts, and preserves
the prior output set if a replacement fails.

## Project structure

```text
python analysis/
├── src/nineam_health_analysis/  # Reusable analytical package
├── scripts/                     # Thin command-line entry point
├── tests/                       # Standard-library unittest suite
├── outputs/
│   ├── tables/                  # 15 slide-ready scientific CSV tables
│   ├── figures/                 # 9 figures in PNG and SVG formats
│   ├── legacy_pre_reporting/    # Superseded outputs retained for traceability
│   └── nineam_analysis_metadata.json
├── docs/                        # Design and implementation records
└── pyproject.toml               # Package and tool configuration
```

## Setup

Python 3.11 or newer is required. Runtime dependencies are NumPy, pandas,
statsmodels, and Matplotlib.

```powershell
Set-Location 'B:\07_Luis\case-study-9amhealth_CLEAN-SLATE\python analysis'
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e '.[dev]'
```

The `dev` extra installs only the optional test and lint tools. To install the
runtime package alone, use `python -m pip install -e .`.

## Run the analysis

The default destination is this project's `outputs` directory. The explicit
command for the supplied data is:

```powershell
.\.venv\Scripts\python.exe scripts\nineam_run_analysis.py `
  --data-dir 'B:\07_Luis\case-study-9amhealth_CLEAN-SLATE\data'
```

Paths and resampling controls can be set explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\nineam_run_analysis.py `
  --data-dir '..\data' `
  --output-dir '.\outputs' `
  --seed 2026 `
  --cv-folds 5 `
  --cv-repeats 10 `
  --stability-resamples 200
```

Use the script's `--help` option for the current defaults and argument details.

## Test and lint

Run the complete suite and style checks with the development dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

The tests include hand-built statistical checks and source-data benchmark
checks. Randomized procedures use explicit seeds so results are reproducible.

## Statistical boundaries

- The complete-pair eligibility cohort contains 633 members. The primary
  analysis then retains only `Active GLP-1 for Weight-loss` and `Coaching Only`
  (534 members) and reports the other 99 members as a comparability limitation.
  `Coaching Only` is the model reference condition.
- The repeated-measures implementation is a **two-occasion marginal GLS**
  model. Its covariance structures describe within-member residual covariance;
  it is not a random-effects mixed model and does not estimate member-specific
  random effects.
- The second base model is ordinary least squares for percentage weight loss,
  with baseline weight and member type in the unpenalized base specification.
- The two base models are compared on the same held-out members and the same
  raw-scale last-weight target using RMSE and MAE. Likelihood criteria are
  reported within a model family, not used to rank models with different
  outcomes or outcome scales. The winner has the strictly lowest mean RMSE;
  mean MAE breaks an RMSE tie, and an exact tie on both favors percentage-loss
  OLS.
- LASSO leaves the confirmed base terms unpenalized and evaluates additional
  covariates with member-level resampling. A predictor's stability-selection
  frequency records how consistently it is selected; it is **not** a p-value
  or evidence of statistical significance.
- `module_mean` and the four curriculum-domain totals are assessed in separate
  specifications because `module_mean` is their exact linear combination.
- The module-mean and domain LASSO specifications use the same grouped-CV fold
  plan. The lower selected-penalty CV MSE wins; a numeric tie within `1e-12`
  favors the mean specification. Within the winning specification, only
  candidates selected in at least 75% of resamples enter the locked model.
- The locked percentage-loss model is refit without a penalty using
  statsmodels OLS and HC3 covariance. Its confidence intervals and p-values are
  explicitly labeled `conditional_exploratory` because they do not account for
  the preceding model-selection process.
- A separate sensitivity refit adds centered linear and quadratic follow-up
  days on the common observed support. It does not replace the primary model.
- The confirmed engagement-rate candidate is repeatable-event volume divided by
  `max(tenure_days, 7)`. A weight-observation-day rate is descriptive only.
- Activity-level models cover all 19 event types. Only types reaching at least
  30 primary-cohort members are tested, using standardized `log1p` counts, HC3,
  baseline weight, member type, and linear/quadratic follow-up adjustment; one
  Benjamini--Hochberg correction covers that eligible family.
- The longitudinal LASSO path is a retrospective, two-stage penalized marginal
  GLS analysis. Its design uses covariance whitening from the supplied
  training fit; grouped lambda CV refits LASSO scaling and lambda-max, but not
  that covariance, inside each fold. It is therefore not presented as unbiased
  nested-CV performance.

These models are exploratory associations, not estimates of causal program,
medication, engagement, or module effects. Eligibility filtering, two weight
occasions, variable follow-up, same-window exposure, absent age, unverified
extension-module availability, post-selection inference, and no external
validation limit generalization and causal interpretation.

## How statsmodels is used

The custom two-occasion GLS likelihood remains explicit and benchmark-tested so
the requested covariance structures and fit indices are transparent. The final
unpenalized percentage-loss refit and the activity association models use
statsmodels OLS with HC3 covariance. In other words, statsmodels is used where
its robust regression implementation directly matches the reporting estimand;
the specialized two-time-point marginal likelihood remains the tested custom
component.

## Output dictionary

`outputs/tables` contains cohort flow, sample characteristics, outcomes,
engagement summaries, all activity types, module completion, base-model
comparison, both LASSO specifications, locked HC3 coefficients and fit
statistics, diagnostics, hypothesis evidence, findings and implications, and
limitations. These 15 tables are the numerical source of record.

`outputs/figures` contains cohort flow, continuous percentage weight loss, 5%
response, engagement patterns, module completion, base-model performance,
LASSO stability, locked coefficients, and diagnostics. Each is written as PNG
and SVG, and every visual reads the validated reporting tables rather than
recomputing estimates.

`outputs/nineam_analysis_metadata.json` records the cohort denominators,
resampling controls, reference condition, outcome threshold, winner and
selection rules, inference labels, and complete output inventory without a
timestamp, so identical runs are byte-comparable.

`outputs/legacy_pre_reporting` preserves the seven superseded CSVs from the
earlier exploratory pipeline. They are not part of the current source of record.

## Python and filename conventions

The Nomenclature section on pages 108–109 of the supplied brand guide uses
underscore-separated filename components beginning with `9am_`. Python import
identifiers cannot begin with a digit, so importable code adapts that prefix to
`nineam_`, followed by a clear snake-case description, for example
`nineam_statistical_models.py`. Package, function, variable, and test names
otherwise follow standard Python naming conventions.

Every function and test helper has a concise definition-level docstring that
states its purpose, inputs, return value, side effects, and statistical intent
where relevant. Inline comments explain non-obvious filtering,
transformations, matrix operations, likelihood calculations, scaling, and
resampling decisions close to the code they describe.
