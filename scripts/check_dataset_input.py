import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.modeling import TrainingInput
from shared.dataset_validation import (
    load_dataset,
    validate_dataset,
)


def main():
    training_input = TrainingInput(
        run_id="development_run_001",
        dataset_path="data/churn.csv",
        problem_stmt="Predict whether a customer will churn.",
        target_column="Churn",
        task_type="classification",
        feature_columns=[
            "Age",
            "MonthlyCharges",
            "Contract",
        ],
        primary_metric="f1_weighted",
    )

    try:
        df = load_dataset(training_input.dataset_path)
        validate_dataset(df, training_input)

    except (OSError, ValueError, csv.Error) as error:
        print(f"Dataset validation failed: {error}")
        return

    print("Dataset loaded and validated successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nTarget counts:")
    print(df[training_input.target_column].value_counts())

    print("\nMissing feature values:")
    print(
        df[training_input.feature_columns]
        .isna()
        .sum()
    )


if __name__ == "__main__":
    main()