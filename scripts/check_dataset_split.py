import csv
from shared.modeling import TrainingInput 
from shared.dataset_validation import load_dataset
from shared.dataset_split import split_dataset

def main():
  training_input = TrainingInput(
    run_id = "development_run_001",
    dataset_path = "data/churn.csv",
    problem_stmt = "Predict whether a customer will churn.",
    target_column = "Churn",
    task_type="classification",
    feature_columns=["Age", "MonthlyCharges", "Contract"],
    primary_metric="f1_weighted",
    test_size=0.2,
    random_state=42,
  )
  try:
    df = load_dataset(training_input.dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(df, training_input)

  except (OSError, ValueError, csv.Error) as error:
    raise SystemExit(f"Dataset split failed: {error}")

  assert X_train.index.equals(y_train.index),(
    "Training features and targets are misaligned."
  )
  assert X_test.index.equals(y_test.index),("Test featiures and targets are misaligned")

  #the csv loader gives each source row a unique index, so the training and test sets should not overlap
  assert X_train.index.intersection(X_test.index).empty, (
        "A source row appears in both training and test sets."
    )

  assert len(X_train) + len(X_test) == len(df), (
        "Some source rows were lost during splitting."
    )

  print("Dataset split checks passed.")

  print(f"\nOriginal rows: {len(df)}")
  print(f"Selected features: {X_train.columns.tolist()}")
  print(f"Target: {y_train.name}")

  print(f"\nX_train shape: {X_train.shape}")
  print(f"X_test shape: {X_test.shape}")
  print(f"y_train shape: {y_train.shape}")
  print(f"y_test shape: {y_test.shape}")

  if training_input.task_type == "classification":
      print("\nTraining class counts:")
      print(y_train.value_counts())

      print("\nTest class counts:")
      print(y_test.value_counts())


if __name__ == "__main__":
    main()