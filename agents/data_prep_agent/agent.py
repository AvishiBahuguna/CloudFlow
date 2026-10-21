import json
import re

from pathlib import Path
from typing import Any

import pandas as pd

from agents.data_prep_agent.profiler import profile_df
from agents.data_prep_agent.planner import create_data_prep_plan, validate_plan
from agents.data_prep_agent.code_generator import generate_notebook_code
from agents.data_prep_agent.notebook_runner import run_notebook
from agents.data_prep_agent.operations import OPERATION_CATALOGUE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "workspace"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def save_json(data: Any, output_path: Path) -> None:
    """Save a Python object as a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def to_dict(data: Any) -> dict:
    """Convert a Pydantic model or dictionary to a dictionary."""

    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")

    if isinstance(data, dict):
        return data

    raise TypeError(
        f"Expected a Pydantic model or dictionary, "
        f"received {type(data).__name__}."
    )


def validate_run_id(run_id: str) -> str:
    """Reject run IDs that could escape the workspace directory."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "run_id may contain only letters, numbers, "
            "underscores, and hyphens."
        )

    return run_id


def validate_operation_catalogue(plan: Any) -> None:
    """Ensure every planned operation is supported."""

    plan_dict = to_dict(plan)

    for operation in plan_dict.get("operations", []):
        operation_name = operation["operation"]

        if operation_name not in OPERATION_CATALOGUE:
            raise ValueError(
                f"Unsupported operation in plan: {operation_name}"
            )


# --------------------------------------------------
# Main Data Prep Agent
# --------------------------------------------------

def run_data_prep(input_path: str | Path, problem_statement: str, run_id: str, workspace_root: str | Path = DEFAULT_WORKSPACE) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete Data Prep Agent workflow.

    Parameters
    ----------
    input_path:
        Path to the input CSV file.

    problem_statement:
        User's question or goal for the dataset.

    run_id:
        Unique identifier for this execution.

    workspace_root:
        Root directory where run artifacts are stored.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Prepared dataset and execution report.
    """

    # ----------------------------------------------
    # 1. Validate inputs and create the workspace
    # ----------------------------------------------

    run_id = validate_run_id(run_id)

    input_path = Path(input_path).expanduser().resolve()
    workspace_root = Path(workspace_root).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError("Currently, the Data Prep Agent accepts CSV files only.")

    if not problem_statement.strip():
        raise ValueError("problem_statement cannot be empty.")

    run_dir = workspace_root / run_id / "data-prep"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Data Prep Agent run: %s", run_id)

    # ----------------------------------------------
    # 2. Load the original dataset
    # ----------------------------------------------

    print("Loading dataset: %s", input_path)

    original_df = pd.read_csv(input_path)

    if original_df.empty:
        raise ValueError("The input dataset contains no rows.")

    print("Loaded dataset with %d rows and %d columns.", len(original_df), len(original_df.columns))

    # ----------------------------------------------
    # 3. Profile the original dataset
    # ----------------------------------------------

    print("Profiling the original dataset.")

    initial_profile = profile_df(original_df)

    initial_profile_dict = to_dict(initial_profile)

    save_json(initial_profile_dict, run_dir / "initial_profile.json",)

    # ----------------------------------------------
    # 4. Ask the planner to create a plan
    # ----------------------------------------------

    print("Generating data preparation plan.")

    plan = create_data_prep_plan(problem_statement=problem_statement, dataset_profile=initial_profile_dict)

    plan_dict = to_dict(plan)

    save_json(plan_dict, run_dir / "operation_plan.json")

    # ----------------------------------------------
    # 5. Validate the plan
    # ----------------------------------------------

    print("Validating the generated plan.")

    # Check that all planned operations are supported.
    validate_operation_catalogue(plan)

    # Validate the plan against the actual dataset.
    validation_errors = validate_plan(plan, original_df)

    if validation_errors:
        raise ValueError("The generated data preparation plan is invalid:\n + \n".join(str(error) for error in validation_errors))

    print("Plan validation passed.")

    # ----------------------------------------------
    # 6. Generate Python notebook code
    # ----------------------------------------------

    print("Generating Python code for the plan.")

    generated_code = generate_notebook_code(
        problem_statement=problem_statement,
        plan=plan,
        dataset_profile=initial_profile_dict,
    )

    # ----------------------------------------------
    # 7. Execute the generated notebook
    # ----------------------------------------------

    print("Executing the data preparation notebook.")

    prepared_df, report = run_notebook(
        generated_code=generated_code,
        input_path=input_path,
        output_dir=run_dir,
    )

    # ----------------------------------------------
    # 8. Profile the prepared dataset
    # ----------------------------------------------

    print("Profiling the prepared dataset.")

    final_profile = profile_df(prepared_df)

    final_profile_dict = to_dict(final_profile)

    save_json(
        final_profile_dict,
        run_dir / "final_profile.json",
    )

    # ----------------------------------------------
    # 9. Update and save the final report
    # ----------------------------------------------

    report["run_id"] = run_id
    report["problem_statement"] = problem_statement
    report["status"] = "completed"

    report["initial_profile"] = initial_profile_dict
    report["final_profile"] = final_profile_dict
    report["plan"] = plan_dict

    report["artifacts"] = {
        **report.get("artifacts", {}),
        "initial_profile": str(run_dir / "initial_profile.json"),
        "operation_plan": str(run_dir / "operation_plan.json"),
        "final_profile": str(run_dir / "final_profile.json"),
        "prepared_dataset": str(run_dir / "prepared.csv"),
        "notebook": str(run_dir / "preparation.ipynb"),
        "operation_results": str(run_dir / "operation_results.json"),
        "report": str(run_dir / "preparation_report.json"),
    }

    save_json(report, run_dir / "preparation_report.json")

    print("Data preparation completed. Output: %s", run_dir / "prepared.csv")

    return prepared_df, report