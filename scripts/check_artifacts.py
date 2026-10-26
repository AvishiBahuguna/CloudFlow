import csv
from pathlib import Path

import numpy as np
import pandas as pd

from agents.modeling.evaluator import evaluate_classification_model
from agents.modeling.trainer import train_classification
from shared.artifacts import (
    RunManifest,
    load_run_manifest,
    load_trained_model,
    save_run_artifacts,
)
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

        train_summary, fitted_pipeline = train_classification(
            X_train=X_train,
            y_train=y_train,
            training_input=training_input,
            plan=plan,
        )

        eval_summary, y_pred = evaluate_classification_model(
            pipeline=fitted_pipeline,
            X_test=X_test,
            y_test=y_test,
            training_summary=train_summary,
            training_input=training_input,
        )

        # 1. Save all run artifacts to disk
        manifest = save_run_artifacts(
            pipeline=fitted_pipeline,
            training_summary=train_summary,
            evaluation_summary=eval_summary,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            base_dir="artifacts/runs",
        )

        assert isinstance(manifest, RunManifest), (
            "save_run_artifacts must return a RunManifest."
        )

        # 2. Check that all expected files were written and are non-empty
        run_dir = Path("artifacts/runs") / training_input.run_id
        expected_files = [
            "model.joblib",
            "run_manifest.json",
            "training_summary.json",
            "evaluation_summary.json",
            "test_predictions.csv",
        ]

        for file_name in expected_files:
            file_path = run_dir / file_name
            assert file_path.exists(), f"Expected artifact missing: {file_name}"
            assert (
                file_path.stat().st_size > 0
            ), f"Artifact file is empty: {file_name}"

        # 3. Reload the model and verify identical predictions
        reloaded_pipeline = load_trained_model(manifest.model_path)
        reloaded_preds = reloaded_pipeline.predict(X_test)
        original_preds = y_pred.values

        assert np.array_equal(reloaded_preds, original_preds), (
            "Reloaded model predictions do not match in-memory model!"
        )

        # 4. Reload and verify the manifest
        loaded_manifest = load_run_manifest(run_dir)
        assert loaded_manifest.run_id == training_input.run_id, (
            "Manifest run_id mismatch."
        )
        assert (
            loaded_manifest.primary_metric_score
            == eval_summary.primary_metric_score
        ), "Manifest metric score mismatch."

        # 5. Check the saved predictions CSV structure
        saved_csv = pd.read_csv(manifest.predictions_path)
        assert len(saved_csv) == len(X_test), "Predictions CSV row mismatch."
        for col in training_input.feature_columns:
            assert col in saved_csv.columns, f"Feature column missing: {col}"
        assert "actual" in saved_csv.columns, "Target column 'actual' missing."
        assert (
            "prediction" in saved_csv.columns
        ), "Prediction column 'prediction' missing."

    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Artifact verification failed: {error}")

    print("Artifact persistence and model reload checks passed successfully.")
    print("\nSaved Run Manifest:")
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
