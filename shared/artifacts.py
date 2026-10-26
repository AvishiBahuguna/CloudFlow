from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from shared.evaluation import EvaluationSummary
from shared.modeling import TrainingSummary


class RunManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    task_type: Literal["classification"] = "classification"
    selected_candidate_id: str = Field(min_length=1)
    selected_model: str = Field(min_length=1)
    primary_metric: Literal["accuracy", "f1_weighted"]
    primary_metric_score: float = Field(ge=0.0, le=1.0)
    beats_baseline: bool
    model_path: str = Field(min_length=1)
    predictions_path: str = Field(min_length=1)
    training_summary_path: str = Field(min_length=1)
    evaluation_summary_path: str = Field(min_length=1)


def save_run_artifacts(
    pipeline: Pipeline,
    training_summary: TrainingSummary,
    evaluation_summary: EvaluationSummary,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    base_dir: str | Path = "artifacts/runs",
) -> RunManifest:
    if training_summary.run_id != evaluation_summary.run_id:
        raise ValueError(
            "Training and evaluation summaries have different run IDs."
        )

    run_dir = Path(base_dir) / training_summary.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save serialized model pipeline
    model_file = run_dir / "model.joblib"
    joblib.dump(pipeline, model_file)

    # 2. Save training summary JSON
    train_summary_file = run_dir / "training_summary.json"
    train_summary_file.write_text(
        training_summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 3. Save evaluation summary JSON
    eval_summary_file = run_dir / "evaluation_summary.json"
    eval_summary_file.write_text(
        evaluation_summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 4. Save test predictions CSV (X_test + actual + prediction)
    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["prediction"] = y_pred.values
    predictions_file = run_dir / "test_predictions.csv"
    test_df.to_csv(predictions_file, index=False)

    # 5. Create and save RunManifest
    manifest = RunManifest(
        run_id=training_summary.run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        task_type="classification",
        selected_candidate_id=training_summary.selected_candidate_id,
        selected_model=training_summary.selected_model,
        primary_metric=evaluation_summary.primary_metric,
        primary_metric_score=evaluation_summary.primary_metric_score,
        beats_baseline=evaluation_summary.beats_baseline,
        model_path=str(model_file.as_posix()),
        predictions_path=str(predictions_file.as_posix()),
        training_summary_path=str(train_summary_file.as_posix()),
        evaluation_summary_path=str(eval_summary_file.as_posix()),
    )

    manifest_file = run_dir / "run_manifest.json"
    manifest_file.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return manifest


def load_trained_model(model_path_or_run_dir: str | Path) -> Pipeline:
    path = Path(model_path_or_run_dir)
    if path.is_dir():
        path = path / "model.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    loaded = joblib.load(path)
    if not isinstance(loaded, Pipeline):
        raise TypeError(
            f"Loaded object is {type(loaded)}, expected scikit-learn Pipeline."
        )

    return loaded


def load_run_manifest(run_dir: str | Path) -> RunManifest:
    manifest_file = Path(run_dir)
    if manifest_file.is_dir():
        manifest_file = manifest_file / "run_manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at: {manifest_file}")

    return RunManifest.model_validate_json(
        manifest_file.read_text(encoding="utf-8")
    )
