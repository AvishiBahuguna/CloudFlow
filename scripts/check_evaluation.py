import csv
from pathlib import Path

from agents.modeling.evaluator import evaluate_classification_model
from agents.modeling.trainer import train_classification
from shared.dataset_split import split_dataset
from shared.dataset_validation import load_dataset
from shared.modeling import TrainingInput
from shared.modeling_plan import ModelingPlan


def main():
    try:
        training_input = TrainingInput(
            run_id="development_run_001",
            dataset_path="data/churn.csv",
            problem_stmt="Predict whether a customer will churn.",
            target_column="Churn",
            task_type="classification",
            feature_columns=["Age", "MonthlyCharges", "Contract"],
            primary_metric="f1_weighted",
            test_size=0.2,
            random_state=42,
        )

        plan_text = Path("configs/modeling_plan.json").read_text(
            encoding="utf-8"
        )
        plan = ModelingPlan.model_validate_json(plan_text)

        df = load_dataset(training_input.dataset_path)

        X_train, X_test, y_train, y_test = split_dataset(
            df,
            training_input,
        )

        summary, fitted_pipeline = train_classification(
            X_train=X_train,
            y_train=y_train,
            training_input=training_input,
            plan=plan,
        )

        eval_summary, predictions = evaluate_classification_model(
            pipeline=fitted_pipeline,
            X_test=X_test,
            y_test=y_test,
            training_summary=summary,
            training_input=training_input,
        )

        assert eval_summary.test_rows == len(X_test), (
            "Evaluation test rows count mismatch."
        )

        assert len(predictions) == len(X_test), (
            "Prediction length does not match test rows."
        )

        assert eval_summary.candidate_id == summary.selected_candidate_id, (
            "Evaluated candidate does not match selected candidate."
        )

        assert 0.0 <= eval_summary.metrics.accuracy <= 1.0, (
            "Accuracy score out of range [0, 1]."
        )

        assert 0.0 <= eval_summary.metrics.f1_weighted <= 1.0, (
            "Weighted F1 score out of range [0, 1]."
        )

        total_cm_cells = sum(
            sum(row) for row in eval_summary.metrics.confusion_matrix
        )
        assert total_cm_cells == len(X_test), (
            "Confusion matrix total does not match test row count."
        )

        assert eval_summary.beats_baseline is True, (
            "Selected model failed to outperform baseline."
        )

    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Evaluation check failed: {error}")

    print("Test evaluation checks passed successfully.")
    print("\nEvaluation Summary:")
    print(eval_summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
