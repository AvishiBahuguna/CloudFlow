import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from shared.modeling_plan import ModelingPlan
from shared.preprocessing import build_preprocessor


MODEL_CLASSES = {
    "dummy_classifier": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_classification_candidates(
    X_train: pd.DataFrame,
    random_state: int,
    plan: ModelingPlan,
) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {}

    for candidate in plan.candidates:
        model_class = MODEL_CLASSES[candidate.model_name]

        parameters = candidate.model_dump(
            exclude={"candidate_id", "model_name"},
        )

        # Execution settings come from the backend.
        if candidate.model_name in {
            "random_forest",
            "gradient_boosting",
        }:
            parameters["random_state"] = random_state

        if candidate.model_name == "random_forest":
            parameters["n_jobs"] = 1

        model = model_class(**parameters)

        candidates[candidate.candidate_id] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

    return candidates