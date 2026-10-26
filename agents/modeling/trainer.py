import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from agents.modeling.registry import build_classification_candidates
from shared.modeling import (
    CandidateCVResult,
    ModelSelectionDecision,
    TrainingInput,
    TrainingSummary,
)
from shared.modeling_plan import ModelingPlan


def validate_classification_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_input: TrainingInput,
    plan: ModelingPlan,
) -> int:
    if training_input.task_type != "classification":
        raise ValueError(
            "This trainer currently supports classification only."
        )

    if plan.task_type != training_input.task_type:
        raise ValueError(
            "The modeling plan and training input have different tasks."
        )

    if X_train.empty:
        raise ValueError("Training features cannot be empty.")

    if not X_train.index.equals(y_train.index):
        raise ValueError(
            "Training features and targets are not aligned."
        )

    if X_train.columns.tolist() != training_input.feature_columns:
        raise ValueError(
            "Training feature columns do not match TrainingInput."
        )

    if y_train.name != training_input.target_column:
        raise ValueError(
            "The training target does not match TrainingInput."
        )

    if y_train.isna().any():
        raise ValueError("Training targets contain missing values.")

    class_counts = y_train.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Classification training requires at least two classes."
        )

    cv_folds = min(5, int(class_counts.min()))

    if cv_folds < 2:
        raise ValueError(
            "Each training class needs at least two examples "
            "for stratified cross-validation."
        )

    return cv_folds


def compare_classification_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_input: TrainingInput,
    plan: ModelingPlan,
    cv_folds: int,
) -> tuple[list[CandidateCVResult], dict[str, Pipeline]]:
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=training_input.random_state,
    )

    folds = list(cv.split(X_train, y_train))

    candidates = build_classification_candidates(
        X_train=X_train,
        random_state=training_input.random_state,
        plan=plan,
    )

    candidate_results: list[CandidateCVResult] = []

    for candidate in plan.candidates:
        pipeline = candidates[candidate.candidate_id]

        cv_output = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            scoring=training_input.primary_metric,
            cv=folds,
            n_jobs=1,
            error_score="raise",
        )

        scores = cv_output["test_score"]
        fit_times = cv_output["fit_time"]

        if not np.isfinite(scores).all():
            raise ValueError(
                f"{candidate.candidate_id} produced invalid scores."
            )

        if (
            not np.isfinite(fit_times).all()
            or (fit_times < 0).any()
        ):
            raise ValueError(
                f"{candidate.candidate_id} produced invalid fitting times."
            )

        candidate_results.append(
            CandidateCVResult(
                candidate_id=candidate.candidate_id,
                model_name=candidate.model_name,
                parameters=candidate.model_dump(
                    exclude={"candidate_id", "model_name"},
                ),
                fold_scores=scores.tolist(),
                mean_score=float(scores.mean()),
                std_score=float(scores.std()),
                mean_fit_time_seconds=float(fit_times.mean()),
            )
        )

    return candidate_results, candidates


def select_classification_candidate(
    candidate_results: list[CandidateCVResult],
    plan: ModelingPlan,
    metric: str = "f1_weighted",
) -> ModelSelectionDecision:
    if not candidate_results:
        raise ValueError("Cannot select from empty candidate results.")

    planned_ids = {c.candidate_id for c in plan.candidates}
    result_ids = {r.candidate_id for r in candidate_results}

    missing_from_plan = result_ids - planned_ids
    if missing_from_plan:
        raise ValueError(
            f"Evaluated candidates {missing_from_plan} are not in the modeling plan."
        )

    # Selection rule: highest mean CV score.
    # Ties preserve the candidate order in the modeling plan.
    candidate_order = {c.candidate_id: i for i, c in enumerate(plan.candidates)}

    best_result = max(
        candidate_results,
        key=lambda r: (r.mean_score, -candidate_order.get(r.candidate_id, 0)),
    )

    reason = (
        f"Selected candidate '{best_result.candidate_id}' ({best_result.model_name}) "
        f"with the highest mean {metric} of {best_result.mean_score:.4f} "
        f"(std: {best_result.std_score:.4f})."
    )

    return ModelSelectionDecision(
        selected_candidate_id=best_result.candidate_id,
        selected_model=best_result.model_name,
        selection_rule="highest_mean_score",
        selection_reason=reason,
    )


def fit_selected_pipeline(
    candidates: dict[str, Pipeline],
    selection: ModelSelectionDecision,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    if selection.selected_candidate_id not in candidates:
        raise ValueError(
            f"Selected candidate '{selection.selected_candidate_id}' was not found in available pipelines."
        )

    selected_pipeline = candidates[selection.selected_candidate_id]
    selected_pipeline.fit(X_train, y_train)
    return selected_pipeline


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_input: TrainingInput,
    plan: ModelingPlan,
) -> tuple[TrainingSummary, Pipeline]:
    cv_folds = validate_classification_inputs(
        X_train=X_train,
        y_train=y_train,
        training_input=training_input,
        plan=plan,
    )

    candidate_results, candidates = compare_classification_candidates(
        X_train=X_train,
        y_train=y_train,
        training_input=training_input,
        plan=plan,
        cv_folds=cv_folds,
    )

    selection = select_classification_candidate(
        candidate_results=candidate_results,
        plan=plan,
        metric=training_input.primary_metric,
    )

    selected_pipeline = fit_selected_pipeline(
        candidates=candidates,
        selection=selection,
        X_train=X_train,
        y_train=y_train,
    )

    summary = TrainingSummary(
        run_id=training_input.run_id,
        primary_metric=training_input.primary_metric,
        training_rows=len(X_train),
        cv_folds=cv_folds,
        candidates=candidate_results,
        selected_candidate_id=selection.selected_candidate_id,
        selected_model=selection.selected_model,
        selection_reason=selection.selection_reason,
    )

    return summary, selected_pipeline