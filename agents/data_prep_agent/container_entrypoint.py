import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from agents.data_prep_agent.agent import run_data_prep

def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

"""Write the container execution status to JSON."""
def write_status(status_path: Path, status_data: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as file:
        json.dump(status_data, file, indent=4, default=str)


def main() -> int:
    # Read configuration supplied by the API when starting the container.
    run_id = os.environ.get("RUN_ID", "").strip()
    objective = os.environ.get("OBJECTIVE", "").strip()
    workspace_root = Path(os.environ.get("WORKSPACE_ROOT", "/workspace")).resolve()

    # Validate the run ID
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        print("Error: RUN_ID is missing or invalid.", file=sys.stderr)
        return 1
    
    run_dir = workspace_root / run_id
    status_path = run_dir / "container_status.json"

    try:
        if not objective:
            raise ValueError("OBJECTIVE must not be empty.")

        # Default input location; API can override it
        input_path = Path(
            os.environ.get("INPUT_PATH", str(run_dir / "input.csv"))).resolve()
        
        if not input_path.is_file():
            raise FileNotFoundError(f"Input dataset not found: {input_path}")
        
        print(f"Starting Data Prep Agent for run: {run_id}")
        print(f"Input dataset: {input_path}")
        print(f"Workspace: {workspace_root}")
        print(f"Objective: {objective}")

        # Execute the existing data preparation pipeline.
        prepared_df, report = run_data_prep(
            input_path=str(input_path),
            problem_statement=objective,
            run_id=run_id,
            workspace_root=workspace_root,
        )

        # Do not recreate or overwrite the agent's existing reports.
        # Record only the overall container execution status.
        status_data = {
            "run_id": run_id,
            "status": "completed",
            "completed_at": current_timestamp(),
            "rows_after": int(prepared_df.shape[0]),
            "columns_after": int(prepared_df.shape[1]),
            "output_directory": str(run_dir / "data-prep"),
        }

        write_status(status_path, status_data)

        print(f"Data Prep Agent completed successfully: {run_id}")
        print(f"Rows after preparation: {prepared_df.shape[0]}")
        print(f"Columns after preparation: {prepared_df.shape[1]}")
        print(f"Status written to: {status_path}")

        return 0
    
    except Exception as exc:
        traceback.print_exc()

        status_data = {
            "run_id": run_id,
            "status": "failed",
            "failed_at": current_timestamp(),
            "error": f"{type(exc).__name__}: {exc}",
        }

        try:
            write_status(status_path, status_data)
        except Exception:
            traceback.print_exc()

        return 1
    


if __name__ == "__main__":
    sys.exit(main())