import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    if X_train.empty:
        raise ValueError("Training features cannot be empty.")

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    supported_columns = set(
        numeric_columns + categorical_columns
    )

    unsupported_columns = [
        column
        for column in X_train.columns
        if column not in supported_columns
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported feature types in columns: "
            f"{unsupported_columns}"
        )

    empty_columns = X_train.columns[
        X_train.isna().all()
    ].tolist()

    if empty_columns:
        raise ValueError(
            "These features contain only missing training values: "
            f"{empty_columns}. Review the selected features."
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor