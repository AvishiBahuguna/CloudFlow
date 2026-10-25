import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_OBJECTIVE_LENGTH = 2000

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CloudFlow API",
    description="Backend API for the CloudFlow data-analysis pipeline.",
    version="0.1.0"
)

# Allow the React/Vite development server to call the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prototype storage: run records exist only while this process is running.
RUNS: dict[str, dict[str, Any]] = {}
RUNS_LOCK = Lock()

def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_run_or_404(run_id: str) -> dict[str, Any]:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found.")
    # Return a shallow copy so endpoints don't modify the stored record.
    return dict(run)

"""
    Execute the Data Prep Agent in the background.

    This is a synchronous function. FastAPI runs synchronous background
    tasks outside the request-handling event loop.
"""
def execute_run(run_id: str) -> None:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return
        
        run["status"] = "running"
        run["started_at"] = current_timestamp()
        input_path = run["input_path"]
        objective = run["objective"]

        try:
            from agents.data_prep_agent.agent import run_data_prep

            prepared_df, report = run_data_prep(
            input_path=input_path,
            problem_statement=objective,
            run_id=run_id,
            workspace_root=WORKSPACE_ROOT,
            )

            report_json = json.loads(json.dumps(jsonable_encoder(report), default=str))

            result = {
                "rows_after": int(prepared_df.shape[0]),
                "columns_after": int(prepared_df.shape[1]),
                "columns": [str(column) for column in prepared_df.columns],
                "report": report_json,
            }

        except Exception as exc:
            import traceback
            traceback.print_exc()
            with RUNS_LOCK:
                RUNS[run_id]["status"] = "failed"
                RUNS[run_id]["completed_at"] = current_timestamp()
                RUNS[run_id]["error"] = (f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


"""
    Accept a CSV upload and start a pipeline run.

    Returns the run ID immediately. The agent executes in the background.
"""
@app.post("/runs", status_code=202)
async def create_pipeline_run(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    objective: str = Form(...),
) -> dict[str, str]:
    
    objective = objective.strip()
    if not objective:
        raise HTTPException(status_code=422, detail="Please provide an analysis objective.")
    
    if len(objective) > MAX_OBJECTIVE_LENGTH:
        raise HTTPException(status_code=422, detail=f"The objective must not exceed {MAX_OBJECTIVE_LENGTH} characters.")
    
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    # Read the upload once, then enforce the same 10 MB limit as the frontend.
    contents = await file.read()
    await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"The maximum upload size is {MAX_FILE_SIZE_MB} MB.")
    
    # Check that pandas can parse the uploaded file before creating a run.
    # This is a lightweight validation; the agent will load it again to
    # perform the actual data-preparation workflow.
    try:
        pd.read_csv(BytesIO(contents), nrows=5)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file could not be read as a CSV: {exc}",
        ) from exc

    run_id = uuid.uuid4().hex
    run_dir = WORKSPACE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_path = run_dir / "input.csv"

    try:
        input_path.write_bytes(contents)
    except OSError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Could not save the uploaded dataset.") from exc
    
    run_record = {
        "run_id": run_id,
        "status": "queued",
        "filename": filename,
        "objective": objective,
        "created_at": current_timestamp(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
        # Internal field; excluded from the API response below.
        "input_path": str(input_path),
    }

    with RUNS_LOCK:
        RUNS[run_id] = run_record

    background_tasks.add_task(execute_run, run_id)

    return {
        "run_id": run_id,
        "status": "queued",
    }


"""Return all runs, newest first."""
@app.get("/runs")
def list_pipeline_runs() -> dict[str, Any]:
    with RUNS_LOCK:
        runs = [dict(run) for run in RUNS.values()]

    runs.sort(key=lambda run: run["created_at"], reverse=True)

    # Do not expose server-side filesystem paths.
    public_runs = [
        {key: value for key, value in run.items() if key != "input_path"}
        for run in runs
    ]

    return {
        "runs": public_runs,
        "total": len(public_runs),
    }


"""Return the status and results of a specific run."""
@app.get("/runs/{run_id}")
def get_pipeline_run(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID.")

    run = get_run_or_404(run_id)

    # Never expose internal filesystem paths.
    run.pop("input_path", None)

    return run