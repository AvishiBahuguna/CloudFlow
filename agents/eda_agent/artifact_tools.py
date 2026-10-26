from pathlib import Path

from shared.eda_schemas import EDAResult

def get_run_directory(
    run_id: int,
    artifact_root: str | Path = "workspace",
) -> Path:
    if run_id < 1:
        raise ValueError("run_id must be a positive integer.")

    return Path(artifact_root) / f"run_{run_id:03d}"

def save_eda_report(
    result: EDAResult,
    run_id: int,
    artifact_root: str | Path = "workspace",
) -> Path:
    run_dir = get_run_directory(run_id, artifact_root)

    report_path = run_dir / "eda_report.json"

    json_text = result.model_dump_json(indent=2)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open("x", encoding="utf-8") as output_file:
        output_file.write(json_text)

    return report_path


def load_eda_report(
    report_path: str | Path,
) -> EDAResult:
    json_text = Path(report_path).read_text(
        encoding="utf-8"
    )

    return EDAResult.model_validate_json(json_text)