import csv
import numpy as np
from shared.modeling import TrainingInput
from shared.dataset_validation import load_dataset
from shared.dataset_split import split_dataset
from shared.preprocessing import build_preprocessor

def main():
  training_input = TrainingInput(
    run_id="development_run_001",dataset_path="data/churn.csv",
    problem_stmt= "Predict whether a customer will churn.",
    target_column="Churn",
    task_type="classification",
    feature_columns=["Age","MonthlyCharges","Contract"],
    primary_metric="f1_weighted",
    test_size=0.2,
    random_state=42,
  )
  try:
        df = load_dataset(training_input.dataset_path)

        X_train, X_test, y_train, y_test = split_dataset(
            df,
            training_input,
        )

        preprocessor = build_preprocessor(X_train)

        # Learn preprocessing rules from training data.
        # Then apply those rules to the training data.
        X_train_prepared = preprocessor.fit_transform(X_train)

        # Apply the already learned rules to test data.
        X_test_prepared = preprocessor.transform(X_test)

        feature_names = preprocessor.get_feature_names_out()

  except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Preprocessing failed: {error}")

    # Processing must preserve the number of examples.
  assert X_train_prepared.shape[0] == len(y_train), (
        "The training row count changed."
    )

  assert X_test_prepared.shape[0] == len(y_test), (
        "The test row count changed."
    )

    # Both sets must have the same output feature structure.
  assert (
        X_train_prepared.shape[1]
        == X_test_prepared.shape[1]
        == len(feature_names)
    ), "Transformed feature counts do not match."

    # The resulting arrays must contain finite numbers.
  assert np.isfinite(X_train_prepared).all(), (
        "Training features contain missing or infinite values."
    )

  assert np.isfinite(X_test_prepared).all(), (
        "Test features contain missing or infinite values."
    )

  print("Preprocessing checks passed.")

  print(f"\nBefore preprocessing:")
  print(f"Training shape: {X_train.shape}")
  print(f"Test shape: {X_test.shape}")

  print(f"\nAfter preprocessing:")
  print(f"Training shape: {X_train_prepared.shape}")
  print(f"Test shape: {X_test_prepared.shape}")

  print("\nOutput feature names:")
  print(feature_names.tolist())

  print("\nFirst three processed training rows:")
  print(X_train_prepared[:3].round(3))


if __name__ == "__main__":
    main()