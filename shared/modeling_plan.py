from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)


class CandidateBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    candidate_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )


class DummyCandidate(CandidateBase):
    model_name: Literal["dummy_classifier"]

    strategy: Literal["most_frequent"] = "most_frequent"


class LogisticCandidate(CandidateBase):
    model_name: Literal["logistic_regression"]

    max_iter: int = Field(default=1000, ge=100, le=5000)
    C: FiniteFloat = Field(default=1.0, ge=0.001, le=100)


class RandomForestCandidate(CandidateBase):
    model_name: Literal["random_forest"]

    n_estimators: int = Field(default=100, ge=10, le=300)
    max_depth: int = Field(default=8, ge=1, le=20)


class GradientBoostingCandidate(CandidateBase):
    model_name: Literal["gradient_boosting"]

    n_estimators: int = Field(default=100, ge=10, le=300)
    learning_rate: FiniteFloat = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
    )
    max_depth: int = Field(default=2, ge=1, le=5)


CandidatePlan = Annotated[
    DummyCandidate
    | LogisticCandidate
    | RandomForestCandidate
    | GradientBoostingCandidate,
    Field(discriminator="model_name"),
]


class ModelingPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    task_type: Literal["classification"] = "classification"

    candidates: list[CandidatePlan] = Field(
        min_length=2,
        max_length=6,
    )

    @model_validator(mode="after")
    def validate_candidates(self):
        candidate_ids = [
            candidate.candidate_id
            for candidate in self.candidates
        ]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate IDs must be unique.")

        baseline_count = sum(
            candidate.model_name == "dummy_classifier"
            for candidate in self.candidates
        )

        if baseline_count != 1:
            raise ValueError(
                "The plan must contain exactly one dummy baseline."
            )

        return self