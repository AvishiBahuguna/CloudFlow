# Translate validated plan to Python code

import ast
import json
import os
import sys

from pathlib import Path
from typing import Any
import textwrap

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

# CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

# Allow imports from agent directory
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from planner import get_operation_catalogue
from shared.schemas import DataPrepPlan


# Configure the LLM API client.
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")

if not api_key:
    raise ValueError("LLM_API_KEY is missing from .env")

if not base_url:
    raise ValueError("LLM_BASE_URL is missing from .env")

if not model:
    raise ValueError("LLM_MODEL is missing from .env")

client = OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# GENERATED CODE SCHEMAS
# ============================================================
"""A Python notebook cell generated for one planned operation."""
class GeneratedCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    operation: str
    title: str
    code: str

"""Structured output returned by the code-generation LLM."""
class GeneratedNotebookCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cells: list[GeneratedCell] = Field(min_length=1)


# ============================================================
# CODE VALIDATION
# ============================================================

FORBIDDEN_IMPORTS = {
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "importlib",
    "pickle",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
}

"""
    Check that generated code has valid Python syntax and
    reject several obviously unsafe constructs.

    (This is a basic validation layer, NOT a security sandbox.)
"""
def validate_generated_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated code contains invalid Python syntax: {exc}") from exc
    
    for node in ast.walk(tree):
        # Reject imports of selected modules
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in FORBIDDEN_IMPORTS:
                    raise ValueError(f"Generated code cannot import '{module_name}'.")
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in FORBIDDEN_IMPORTS:
                    raise ValueError(f"Generated code cannot import from '{module_name}'.")
                
        # Reject selected dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise ValueError(f"Generated code cannot call '{node.func.id}()'.")
                


# ============================================================
# CODE GENERATION
# ============================================================
"""
    Generate Python notebook cells from a validated DataPrepPlan.

    Args:
        problem_statement:
            The user's original question.

        plan:
            A validated DataPrepPlan produced by planner.py.

        dataset_profile:
            The profile of the original dataset.

    Returns:
        GeneratedNotebookCode containing executable Python cells.
"""
def generate_notebook_code(problem_statement: str, plan: DataPrepPlan, dataset_profile: dict[str, Any]) -> GeneratedNotebookCode:
    # Validate plan's structure
    if isinstance(plan, dict):
        plan = DataPrepPlan.model_validate(plan)

    operation_catalogue = get_operation_catalogue()
    plan_data = plan.model_dump(mode='json')
    # Include only the relevant operation definitions
    planned_operation_names = {operation["operation"] for operation in plan_data["operations"]}
    supported_operations = {
        name: definition
        for name, definition in operation_catalogue.items()
        if name in planned_operation_names
    }

    if not plan_data["operations"]:
        raise ValueError("The validated plan contains no operations to execute.")
    
    system_prompt = """
        You are the Python code-generation component of CloudFlow,
        an automated data preparation agent.

        Your job is to translate a VALIDATED DataPrepPlan into executable
        Python notebook cells.

        IMPORTANT RULES:

        1. Follow the validated plan exactly.
        2. Generate exactly one Python cell for each planned operation.
        3. Preserve the order of operations in the plan.
        4. Do not add, remove, reorder or replace planned operations.
        5. Implement the operations in Python using pandas and numpy.
        6. Use the provided operation catalogue to understand the
           meaning and parameters of each operation.
        7. Do not call execute_operation(), HANDLERS, or any operation
           handler from operations.py. You must write the implementation
           code yourself.
        8. Do not invent dataset columns. Use only columns present in
           the supplied profile.
        9. Do not perform transformations that are not authorized by
           the plan.
        10. Do not silently discard rows, fill missing values, remove
            duplicates, or change column types unless the plan
            authorizes the corresponding action.
        11. When converting values, report values that could not be
            converted and any resulting missing values.
        12. Inspection operations must record their findings without
            modifying the DataFrame.
        13. Verification operations must inspect the dataset without
            modifying it.
        14. Do not train machine-learning models or perform EDA.
        15. Do not read or write files. The notebook runner manages
            dataset loading and artifact saving.
        16. Do not use shell commands, subprocesses, network requests,
            eval(), exec(), or dynamic imports.
        17. Do not install packages.
        18. Do not use input() or require user interaction.
        19. Do not generate markdown cells. Generate Python code only.

        NOTEBOOK EXECUTION ENVIRONMENT:

        The notebook runner will initialize these variables before
        your generated cells execute:

        - df: pandas DataFrame containing the original dataset.
        - pd: imported pandas module.
        - json: imported json module.
        - operation_results: an initially empty dictionary.

        Each cell executes sequentially. Later cells can use variables
        created by earlier cells.

        CELL REQUIREMENTS:

        For every planned operation:

        - Generate one cell with the correct step number and operation name.
        - Implement the operation in Python.
        - Store its findings or transformation details in
          operation_results["step_<number>"].
        - The stored result must be a dictionary containing:
            "operation": the operation name
            "status": "completed"
            "result": a dictionary of actual findings or changes

        For example, a missing-value inspection could store:

        operation_results["step_1"] = {
            "operation": "inspect_missing",
            "status": "completed",
            "result": missing_report
        }

        The result must come from actual execution against df.
        Do not invent findings or hardcode expected results.

        If an operation cannot be completed, let the execution fail
        with a useful error message. Do not mark a failed operation
        as completed.

        OUTPUT FORMAT:

        Return a JSON object with this structure:

        {
            "cells": [
                {
                    "step": 1,
                    "operation": "inspect_missing",
                    "title": "Inspect missing values",
                    "code": "valid Python code here"
                }
            ]
        }

        Return no markdown fences or explanatory text outside the JSON.
    """

    user_prompt = f"""
        USER'S QUESTION: {problem_statement},
        VALIDATED DATA PREPARATION PLAN: {json.dumps(plan_data, indent=2, default=str)},,
        DATASET PROFILE: {json.dumps(dataset_profile, indent=2, default=str)},
        SUPPORTED OPERATION DEFINITIONS: {json.dumps(supported_operations, indent=2, default=str)},

        Generate the Python notebook cells now.

        The number of cells must equal the number of planned operations.
        Each cell must correspond to the operation at the same position
        in the validated plan.
    """

    response = client.chat.completions.create(
        model = model,
        messages= [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("The LLM returned an empty code-generation response.")

    try:
        response_data = json.loads(response_text)
        generated = GeneratedNotebookCode.model_validate(response_data)

    except Exception as exc:
        raise ValueError(f"The LLM returned an invalid code-generation response: {exc}") from exc
    
    # Ensure the LLM generated precisely the requested operations.
    planned_operations = plan_data["operations"]

    if len(generated.cells) != len(planned_operations):
        raise ValueError("Generated cell count does not match the number of planned operations.")
    
    for index, (cell, operation) in enumerate(zip(generated.cells, planned_operations), start=1):
        expected_operation = operation["operation"]
        
        if cell.step != index:
            raise ValueError(f"Expected step {index}, but received step {cell.step}.")
        
        if cell.operation != expected_operation:
            raise ValueError(f"Step {index} should implement '{expected_operation}', but the LLM generated '{cell.operation}'.")
        
        # Remove unnecessary indentation from the generated code.
        cell.code = textwrap.dedent(cell.code).strip()
        
        if not cell.code.strip():
            raise ValueError(f"Generated code for step {index} is empty.")
        
        validate_generated_code(cell.code)

    return generated