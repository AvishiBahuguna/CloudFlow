from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)


class TrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    problem_stmt: str = Field(min_length=1)
    target_column: str = Field(min_length=1)

    task_type: Literal["classification", "regression"]

    feature_columns: list[str] = Field(min_length=1)

    primary_metric: Literal[
        "accuracy",
        "f1_weighted",
        "mae",
        "rmse",
        "r2",
    ]

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
    )

    random_state: int = Field(
        default=42,
        ge=0,
        le=4294967295,
    )

    @field_validator(
        "run_id",
        "dataset_path",
        "problem_stmt",
        "target_column",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be blank.")

        return value

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(
        cls,
        columns: list[str],
    ) -> list[str]:
        if any(not column.strip() for column in columns):
            raise ValueError("Feature names cannot be blank.")

        if len(columns) != len(set(columns)):
            raise ValueError("Feature names must be unique.")

        return columns

    @model_validator(mode="after")
    def validate_training_plan(self):
        if self.target_column in self.feature_columns:
            raise ValueError(
                "The target column cannot also be a feature."
            )

        allowed_metrics = {
            "classification": {"accuracy", "f1_weighted"},
            "regression": {"mae", "rmse", "r2"},
        }

        if self.primary_metric not in allowed_metrics[self.task_type]:
            raise ValueError(
                "The selected metric does not match the task type."
            )

        return self


class CandidateCVResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    parameters: dict[str, int | float | str]

    fold_scores: list[FiniteFloat] = Field(min_length=2)
    mean_score: FiniteFloat
    std_score: FiniteFloat = Field(ge=0)

    mean_fit_time_seconds: FiniteFloat = Field(ge=0)


class TrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    task_type: Literal["classification"] = "classification"
    primary_metric: Literal["accuracy", "f1_weighted"]

    training_rows: int = Field(ge=2)
    cv_folds: int = Field(ge=2)

    candidates: list[CandidateCVResult] = Field(min_length=1)

    selected_candidate_id: str = Field(min_length=1)
    selected_model: str = Field(min_length=1)