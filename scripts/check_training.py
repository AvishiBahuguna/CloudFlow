import csv
from pathlib import Path

from agents.modeling.registry import build_classification_candidates
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

        # Validate the plan before constructing or training models.
        plan_text = Path(
            "configs/modeling_plan.json"
        ).read_text(encoding="utf-8")

        plan = ModelingPlan.model_validate_json(plan_text)

        df = load_dataset(training_input.dataset_path)

        X_train, X_test, y_train, y_test = split_dataset(
            df,
            training_input,
        )

        # Check that the registry applies the requested settings.
        preview_candidates = build_classification_candidates(
            X_train=X_train,
            random_state=training_input.random_state,
            plan=plan,
        )

        for candidate in plan.candidates:
            pipeline = preview_candidates[candidate.candidate_id]
            actual_parameters = pipeline.named_steps[
                "model"
            ].get_params()

            expected_parameters = candidate.model_dump(
                exclude={"candidate_id", "model_name"},
            )

            for name, value in expected_parameters.items():
                assert actual_parameters[name] == value, (
                    f"{candidate.candidate_id}: "
                    f"parameter {name} was not applied correctly."
                )

        summary, fitted_pipeline = train_classification(
            X_train,
            y_train,
            training_input,
            plan,
        )

        expected_ids = {
            candidate.candidate_id
            for candidate in plan.candidates
        }

        actual_ids = {
            result.candidate_id
            for result in summary.candidates
        }

        assert actual_ids == expected_ids, (
            "Some planned candidates are missing from the results."
        )

        assert len(summary.candidates) == len(plan.candidates), (
            "The number of results does not match the plan."
        )

        sample_rows = X_train.head(2)
        predictions = fitted_pipeline.predict(sample_rows)

        assert len(predictions) == len(sample_rows), (
            "Prediction count does not match the input row count."
        )

    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Training failed: {error}")

    print("Modeling plan execution and prediction checks passed.")

    print("\nTraining summary:")
    print(summary.model_dump_json(indent=2))

    print(f"\nReserved test rows: {len(X_test)}")
    print("Final test evaluation is pending.")


if __name__ == "__main__":
    main()