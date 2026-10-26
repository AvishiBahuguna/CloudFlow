from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
)


class ClassificationMetrics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    accuracy: FiniteFloat = Field(ge=0.0, le=1.0)
    f1_weighted: FiniteFloat = Field(ge=0.0, le=1.0)
    precision_weighted: FiniteFloat = Field(ge=0.0, le=1.0)
    recall_weighted: FiniteFloat = Field(ge=0.0, le=1.0)

    confusion_matrix: list[list[int]] = Field(min_length=2)
    labels: list[str | int] = Field(min_length=2)


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    run_id: str = Field(min_length=1)
    task_type: Literal["classification"] = "classification"
    candidate_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    test_rows: int = Field(ge=2)
    primary_metric: Literal["accuracy", "f1_weighted"]
    primary_metric_score: FiniteFloat = Field(ge=0.0, le=1.0)

    metrics: ClassificationMetrics

    cv_mean_score: FiniteFloat = Field(ge=0.0, le=1.0)
    generalization_gap: FiniteFloat

    baseline_score: FiniteFloat | None = None
    beats_baseline: bool
