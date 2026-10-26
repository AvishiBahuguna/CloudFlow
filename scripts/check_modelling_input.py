from pydantic import ValidationError

from shared.modeling import TrainingInput


def main():
    sample_input = {
        "run_id": "development_run_001",
        "dataset_path": "data/churn.csv",
        "problem_stmt": "Predict whether a customer will churn.",
        "target_column": "Churn",
        "task_type": "classification",
        "feature_columns": [
            "Age",
            "MonthlyCharges",
            "Contract",
        ],
        "primary_metric": "f1_weighted",
    }

    # Check a valid configuration.
    training_input = TrainingInput.model_validate(sample_input)

    print("VALID INPUT:")
    print(training_input.model_dump_json(indent=2))

    # Deliberately include the target among the features.
    invalid_input = {
        **sample_input,
        "feature_columns": ["Age", "Churn"],
    }

    print("\nINVALID INPUT:")

    try:
        TrainingInput.model_validate(invalid_input)
    except ValidationError as error:
        print(error)


if __name__ == "__main__":
    main()