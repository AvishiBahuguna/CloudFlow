import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from shared.evaluation import ClassificationMetrics, EvaluationSummary
from shared.modeling import TrainingInput, TrainingSummary


def evaluate_classification_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    training_summary: TrainingSummary,
    training_input: TrainingInput,
) -> tuple[EvaluationSummary, pd.Series]:
    if X_test.empty:
        raise ValueError("Test features cannot be empty.")

    if not X_test.index.equals(y_test.index):
        raise ValueError("Test features and targets are not aligned.")

    if X_test.columns.tolist() != training_input.feature_columns:
        raise ValueError("Test feature columns do not match TrainingInput.")

    if y_test.name != training_input.target_column:
        raise ValueError("Test target does not match TrainingInput.")

    if y_test.isna().any():
        raise ValueError("Test targets contain missing values.")

    predictions = pipeline.predict(X_test)
    y_pred = pd.Series(predictions, index=y_test.index, name="prediction")

    labels = sorted(y_test.unique().tolist())

    acc = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    prec = float(
        precision_score(y_test, predictions, average="weighted", zero_division=0)
    )
    rec = float(
        recall_score(y_test, predictions, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_test, predictions, labels=labels).tolist()

    class_metrics = ClassificationMetrics(
        accuracy=acc,
        f1_weighted=f1,
        precision_weighted=prec,
        recall_weighted=rec,
        confusion_matrix=cm,
        labels=labels,
    )

    metric_lookup = {
        "accuracy": acc,
        "f1_weighted": f1,
    }
    primary_score = metric_lookup[training_input.primary_metric]

    cv_mean = None
    baseline_score = None
    for candidate in training_summary.candidates:
        if candidate.candidate_id == training_summary.selected_candidate_id:
            cv_mean = candidate.mean_score
        if candidate.model_name == "dummy_classifier":
            baseline_score = candidate.mean_score

    if cv_mean is None:
        raise ValueError(
            f"Selected candidate {training_summary.selected_candidate_id} "
            "was not found in training summary candidates."
        )

    beats_baseline = (
        True if baseline_score is None else primary_score >= baseline_score
    )
    generalization_gap = float(cv_mean - primary_score)

    summary = EvaluationSummary(
        run_id=training_summary.run_id,
        task_type="classification",
        candidate_id=training_summary.selected_candidate_id,
        model_name=training_summary.selected_model,
        test_rows=len(X_test),
        primary_metric=training_input.primary_metric,
        primary_metric_score=primary_score,
        metrics=class_metrics,
        cv_mean_score=cv_mean,
        generalization_gap=generalization_gap,
        baseline_score=baseline_score,
        beats_baseline=beats_baseline,
    )

    return summary, y_pred
