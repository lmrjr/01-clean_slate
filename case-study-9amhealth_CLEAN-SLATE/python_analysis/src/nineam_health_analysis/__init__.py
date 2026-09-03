"""Expose the stable public interfaces for the 9amHealth analysis package."""

from .nineam_analysis_pipeline import (
    AnalysisConfig,
    AnalysisResult,
    run_analysis,
    write_analysis_outputs,
)
from .nineam_cohort_selection import (
    PRIMARY_MEMBER_TYPES,
    REFERENCE_MEMBER_TYPE,
    CohortResult,
    build_analysis_cohort,
    restrict_to_primary_member_types,
)
from .nineam_data_loading import CaseStudyData, load_case_study_data
from .nineam_feature_engineering import (
    MODULE_DOMAIN_DENOMINATORS,
    FeatureResult,
    build_member_features,
    classify_repeatable_events,
)
from .nineam_final_model import (
    LockedCandidateSelection,
    LockedModelResult,
    choose_base_model_winner,
    fit_locked_percentage_model,
    select_locked_candidates,
)
from .nineam_penalized_models import (
    CollinearityDiagnostic,
    PartiallyPenalizedLassoResult,
    diagnose_collinearity,
    fit_partially_penalized_lasso,
    predict_partially_penalized_lasso,
)
from .nineam_reporting import (
    ACTIONABILITY_CLASSES,
    LIMITATION_IDS,
    REPORTING_SCHEMAS,
    ReportingResult,
    build_reporting_tables,
    wilson_confidence_interval,
)
from .nineam_resampling import (
    GroupedFold,
    LambdaCVResult,
    PenalizedDesign,
    StabilitySelectionResult,
    build_longitudinal_penalized_design,
    build_percentage_penalized_design,
    compare_base_models,
    make_grouped_repeated_folds,
    select_lambda_by_grouped_cv,
    stability_select_lasso,
)
from .nineam_statistical_models import (
    LongitudinalGLSResult,
    ModelSchema,
    PercentageLossOLSResult,
    fit_longitudinal_gls,
    fit_percentage_loss_ols,
    predict_last_weight_longitudinal,
    predict_last_weight_percentage_loss,
)
from .nineam_visualizations import FIGURE_STEMS, write_analysis_figures

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ACTIONABILITY_CLASSES",
    "CaseStudyData",
    "CohortResult",
    "CollinearityDiagnostic",
    "FeatureResult",
    "FIGURE_STEMS",
    "LockedCandidateSelection",
    "LockedModelResult",
    "LIMITATION_IDS",
    "GroupedFold",
    "LambdaCVResult",
    "LongitudinalGLSResult",
    "ModelSchema",
    "MODULE_DOMAIN_DENOMINATORS",
    "PartiallyPenalizedLassoResult",
    "PenalizedDesign",
    "PercentageLossOLSResult",
    "PRIMARY_MEMBER_TYPES",
    "REFERENCE_MEMBER_TYPE",
    "REPORTING_SCHEMAS",
    "ReportingResult",
    "StabilitySelectionResult",
    "build_analysis_cohort",
    "build_longitudinal_penalized_design",
    "build_member_features",
    "build_percentage_penalized_design",
    "build_reporting_tables",
    "classify_repeatable_events",
    "choose_base_model_winner",
    "compare_base_models",
    "diagnose_collinearity",
    "fit_longitudinal_gls",
    "fit_locked_percentage_model",
    "fit_partially_penalized_lasso",
    "fit_percentage_loss_ols",
    "load_case_study_data",
    "make_grouped_repeated_folds",
    "predict_last_weight_longitudinal",
    "predict_partially_penalized_lasso",
    "predict_last_weight_percentage_loss",
    "run_analysis",
    "restrict_to_primary_member_types",
    "select_lambda_by_grouped_cv",
    "select_locked_candidates",
    "stability_select_lasso",
    "write_analysis_outputs",
    "write_analysis_figures",
    "wilson_confidence_interval",
]

__version__ = "0.1.0"
