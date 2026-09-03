# 9amHealth Scientific Analysis and Reporting Design

## Goal

Extend the existing reproducible Python analysis so every analytical result
needed for a 5--7 slide scientific leadership presentation is available as a
validated table or standalone figure. This work produces analysis artifacts;
it does not create or edit a PowerPoint deck.

## Confirmed analysis population

- Start from all 865 enrolled members in the demographics extract.
- Apply the existing eligibility and complete-pair rules: ACTIVE or FINISHED,
  positive first and last weights, and a positive measurement interval.
- Preserve that 633-member complete-pair cohort as a documented intermediate.
  From it, retain only `Active GLP-1 for Weight-loss`
  and `Coaching Only` for every primary descriptive, model, selection, and
  figure artifact.
- Use `Coaching Only` as the model reference group.
- Report the 99 members in Null, GLP-1 for Diabetes, and active generic
  weight-loss medication only in cohort accounting and limitations.
- Expected primary analytic sample after the separate comparability restriction:
  534 members and 1,068 long rows.

## Outcomes and estimands

- Primary outcome: continuous percentage weight loss,
  `100 * (first_weight - last_weight) / first_weight`.
- Secondary outcome: at least 5% weight loss.
- Also report first weight, last weight, absolute pounds lost, and measurement
  interval overall and by the two retained member types.
- Continuous summaries contain observed and missing counts, mean, standard
  deviation, median, quartiles, minimum, and maximum.
- Binary response summaries contain numerator, denominator, percentage, and a
  Wilson 95% confidence interval.

## Engagement and module analysis

- Preserve the confirmed engagement feature definitions: breadth, repeatable
  volume, repeatable-volume rate, and tenure days.
- Keep event repeatability classification based on the full canonical
  engagement table, then calculate cohort-window event summaries on the 534
  retained members through each member's last-weight date.
- Report activity type reach, event count, events per reached member,
  repeatability, and exploratory associations with percentage loss. Activity
  tests use false-discovery-rate-adjusted p-values and remain descriptive and
  noncausal.
- Treat a unique answered member-title record as a module completion, disclose
  that assumption, and report the overall module mean plus core, mindset,
  nutrition, and physical-activity domains.
- Use ethnicity in sample description only. Use sex in the confirmed LASSO
  candidate set. Age cannot be analyzed because it is absent from the supplied
  demographics extract.

## Model sequence

1. Fit the longitudinal marginal model and the percentage-loss OLS base model
   to the same 534 members.
2. Compare families only with paired held-out raw last-weight RMSE and MAE.
   Never use AIC, BIC, or negative-two-log-likelihood to rank different outcome
   scales or model families. Lowest mean RMSE wins, mean MAE breaks an RMSE
   tie, and an exact tie on both metrics prefers percentage-loss OLS because it
   directly models the prespecified primary outcome.
3. Run separate module-mean and module-domain partially penalized LASSO
   specifications. Baseline weight and member type remain unpenalized.
4. Choose between the module-mean and module-domain LASSO specifications by
   the lower grouped-CV mean squared error at the selected penalty; prefer the
   mean specification if those values are numerically tied. Within the chosen
   specification, select additional covariates by the prespecified stability
   threshold of 0.75 across member-level resamples, not by the smallest
   p-value.
5. Refit the winning family without an L1 penalty using the locked stable
   covariates. For a percentage-loss winner, use OLS with HC3 covariance.
6. Label post-selection coefficient intervals and p-values
   `conditional_exploratory`; HC3 does not account for data-driven selection.

The confirmed penalized candidates are repeatable volume, repeatable-volume
rate, breadth, tenure days, module mean or separate curriculum domains, and
sex. Module mean and its four exact component domains remain in separate
specifications.

## Scientific outputs

Write all CSV tables under `outputs/tables` and all figures as both PNG and SVG
under `outputs/figures`. Write run metadata to
`outputs/nineam_analysis_metadata.json`. Tables include cohort flow, Table 1 sample
characteristics, outcomes by member type, aggregate and activity-level
engagement, module completion, base-model performance, both LASSO stability
specifications, locked-model coefficients, locked-model fit statistics,
diagnostics, a hypothesis/evidence register, findings and implications, and
limitations. Every figure has a corresponding CSV source table.

Figures cover cohort flow, continuous weight loss by member type, 5% response
by member type, engagement patterns, module completion, base-model predictive
performance, LASSO stability, locked-model coefficients, and model
diagnostics. Figures are standalone scientific artifacts and do not contain
member identifiers.

## Interpretation rules

- Use association language throughout. The data are observational and do not
  establish that medication, coaching, engagement, or module completion caused
  weight loss.
- The two retained conditions are substantively more comparable than the
  excluded conditions; that decision does not itself prove residual, linearity,
  independence, or confounding assumptions.
- Recommendations are framed as actions to test in a future randomized pilot,
  with a measurable outcome, rather than as proven treatment effects.
- Include attrition, same-window exposure/reverse causation, variable follow-up,
  absent age, post-selection inference, multiple testing, and lack of external
  validation among the limitations.

## Prespecified sensitivity analyses

The confirmed primary approach remains unchanged: its base terms are first
weight and member type, its engagement-rate candidate uses the supplied
`tenure_days` denominator, and its module mean is the supplied four-domain row
mean. Separate sensitivities test the risks that those choices create without
silently redefining the primary analysis:

- Refit the locked percentage-loss model on the common observed follow-up range
  with unpenalized linear and quadratic `weight_days` terms. Report its
  coefficients and fit statistics under a distinct model identifier; do not
  use it to replace the primary LASSO selection.
- Report an interval-normalized engagement rate,
  `repeatable_volume / max(weight_days, 7)`, as a descriptive sensitivity only.
  Keep the user-confirmed tenure-normalized rate in the primary candidate set.
- Report all 19 activity types. Test activity associations only when at least
  30 primary-cohort members reached the type, using a standardized
  `log1p(event_count)` predictor, HC3 OLS, baseline weight, member type, and
  linear/quadratic `weight_days` adjustment. Apply Benjamini--Hochberg once
  across the eligible activity family; low-reach types remain descriptive.
- Report that availability of extension modules cannot be verified from the
  extract and label extension-domain coefficient support as GLP-1-only when
  Coaching has no observed completions. Retain the confirmed mean/domain LASSO
  specifications but do not describe these terms as cross-group causal effects.
- Use the same grouped-CV fold plan for the mean and domain LASSO
  specifications so their selected-penalty MSE values are paired. A numeric tie
  within `1e-12` prefers the mean specification.

## Closed CSV schemas

Columns appear in the exact order shown; fields that do not apply to a row are
empty rather than imputed.

1. `nineam_cohort_flow.csv`
   `flow_order, row_type, stage_id, stage_label, starting_n, excluded_n, retained_n, percent_of_enrolled, exclusion_definition, notes`
2. `nineam_sample_characteristics.csv`
   `scope_order, scope, characteristic_order, characteristic, characteristic_label, level_order, level, summary_type, unit, n_total, n_observed, n_missing, count, percentage, mean, standard_deviation, median, q1, q3, minimum, maximum, notes`
3. `nineam_outcomes_by_member_type.csv`
   `scope_order, scope, outcome_order, outcome, outcome_label, outcome_type, unit, n_total, n_observed, n_missing, mean, standard_deviation, median, q1, q3, minimum, maximum, numerator, denominator, percentage, ci_lower, ci_upper, ci_level, ci_method`
4. `nineam_engagement_by_member_type.csv`
   `scope_order, scope, metric_order, metric, metric_label, unit, n_total, n_observed, n_missing, zero_n, reached_n, mean, standard_deviation, median, q1, q3, minimum, maximum, exposure_window_definition, denominator_definition`
5. `nineam_engagement_activity_summary.csv`
   `event_order, event_type, actionability_class, is_repeatable, n_analysis, n_reached, n_zero, total_events, events_per_reached_member, median_events_among_reached, q1_events_among_reached, q3_events_among_reached, predictor, predictor_transform, effect_unit, estimate, standard_error, test_statistic, reference_distribution, degrees_of_freedom, p_value, ci_lower, ci_upper, ci_level, fdr_method, fdr_family, fdr_test_count, fdr_adjusted_p_value, covariance_estimator, adjustment_terms, reference_member_type, test_status, interpretation`
6. `nineam_modules_by_member_type.csv`
   `scope_order, scope, module_order, module_variable, module_label, module_group, available_title_denominator, availability_status, support_scope, n_members, n_members_with_completion, n_members_without_completion, member_completion_percentage, total_unique_completions, mean_completion_count, standard_deviation_count, median_completion_count, q1_completion_count, q3_completion_count, minimum_completion_count, maximum_completion_count, mean_completion_proportion, standard_deviation_proportion, zero_completion_group_flag, completion_definition, cross_group_comparability, interpretation`
7. `nineam_base_model_comparison_summary.csv`
   `model_order, model_id, model_family, modeled_outcome, prediction_target, metric_order, metric, metric_unit, n_members, cv_repeats, cv_folds, n_fold_scores, n_test_predictions, mean_score, standard_deviation, minimum_score, maximum_score, fold_plan_id, score_aggregation, rank, is_winner, winner_rule, tie_tolerance, validation_scope`
8. `nineam_lasso_mean_selection.csv`
   `specification_order, module_spec, is_winning_specification, candidate_order, candidate, candidate_label, candidate_role, support_scope, reference_level, penalty_status, selected_lambda_ratio, full_sample_lambda, full_sample_lambda_max, cv_selection_rule, cv_mean_mse, cv_standard_error, fold_plan_id, n_resamples, subsample_fraction, selection_threshold, full_sample_coefficient, full_sample_standardized_coefficient, selection_count, selection_frequency, selected_at_threshold, eligible_for_locked_model, locked_model_status, exclusion_reason, coefficient_unit, interpretation`
9. `nineam_lasso_domain_selection.csv` uses the same ordered schema as the
   mean-selection table.
10. `nineam_locked_model_coefficients_hc3.csv`
    `model_id, term_order, term, term_label, term_role, support_scope, contrast, reference, estimate, standard_error, test_statistic, reference_distribution, degrees_of_freedom, p_value, ci_lower, ci_upper, ci_level, unit, n_members, covariance_estimator, winning_base_model, winning_module_spec, selection_frequency, selection_provenance, inference_status`
11. `nineam_locked_model_fit_statistics.csv`
    `model_id, winning_base_model, winning_module_spec, analysis_population, modeled_outcome, model_formula, reference_member_type, duration_adjustment, n_members, n_parameters, residual_degrees_of_freedom, covariance_estimator, r_squared, adjusted_r_squared, residual_rmse, negative_two_log_likelihood, aic, bic, likelihood_use, standardized_design_condition_number, maximum_leverage, maximum_cooks_distance, selection_rule, inference_status, fit_status`
12. `nineam_model_diagnostics.csv`
    `model_id, diagnostic_order, diagnostic_type, row_type, series, bin_order, bin_method, bin_lower, bin_upper, bin_count, x_value, y_value, y_lower, y_upper, metric, value, threshold, flag, status, interpretation`
13. `nineam_hypothesis_evidence.csv`
    `hypothesis_order, hypothesis_id, question, prespecified_expectation, estimand, population, exposure_or_predictor, outcome, adjustment_set, evidence_table, evidence_filter, result_status, effect_summary, uncertainty_summary, multiplicity_control, inference_status, causal_status, noncausal_interpretation, leadership_action, randomized_pilot_population, randomized_pilot_comparator, randomized_pilot_kpi, randomized_pilot_time_horizon`
14. `nineam_findings_and_implications.csv`
    `finding_order, finding_id, finding, quantitative_evidence, evidence_table, evidence_filter, interpretation, certainty, causal_status, leadership_implication, recommended_test, pilot_population, pilot_intervention, pilot_comparator, pilot_primary_kpi, pilot_time_horizon`
15. `nineam_limitations.csv`
    `limitation_order, limitation_id, category, limitation, empirical_evidence, affected_estimand, potential_impact, direction_of_bias, affected_outputs, mitigation_in_current_analysis, recommended_future_design, severity`

Controlled values used by table and figure filters are:

- cohort `row_type`: `stage`;
- base-model `prediction_target`: `raw_last_weight`;
- LASSO `penalty_status`: `penalized`;
- coefficient `term_role`: `intercept`, `base`, `selected_candidate`, or
  `sensitivity_adjustment`;
- diagnostic `diagnostic_type`: `residuals_vs_fitted`, `normal_qq`, `leverage`,
  `cooks_distance`, or `global_metric`;
- diagnostic `series`: `binned_mean`, `quantile_pair`, or `summary_metric`;
- responder `percentage`, `ci_lower`, and `ci_upper`: percentage-point scale
  from 0 to 100, with `ci_method=Wilson` and `ci_level=0.95`.

Continuous descriptive whiskers represent the reported range or interquartile
range, not confidence intervals. Fold-performance spreads represent standard
deviation or range, not independent-fold confidence intervals.

## Code and verification constraints

- Preserve the existing `nineam_` lowercase snake-case naming convention.
- Keep `scripts/nineam_run_analysis.py` as the command-line entry point.
- Add `statsmodels` for HC3 inference and `matplotlib` for deterministic plots.
- Every function has a concise docstring explaining purpose, inputs, return
  value, side effects, and statistical intent when applicable.
- Add inline comments before non-obvious filters, transformations, model
  choices, and output logic.
- Tests use hand-derived fixtures and are written and observed failing before
  production changes.
- Source extracts remain read-only. Outputs contain aggregate results only and
  no member identifiers. Small-cell suppression is not applied to the approved
  analytical summaries.
- A fixed seed controls resampling and plotting. Two complete CLI runs must
  produce identical tables and figures byte for byte where the file format
  permits deterministic serialization.
