import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

from shared.schemas import DataPrepPlan

# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

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


# Return the available operations and their descriptions.
def get_operation_catalogue() -> dict:
    import sys

    agent_dir = str(PROJECT_ROOT / "agents" / "data-prep-agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    from operations import OPERATION_CATALOGUE
    return OPERATION_CATALOGUE


"""
    Use the LLM to plan dataset preparation.

    The LLM identifies relevant columns and selects operations.
    Python will validate and execute the resulting plan.
"""
def create_data_prep_plan(problem_statement: str, dataset_profile: dict) -> DataPrepPlan:
    system_prompt = """
        You are the planning component of a data preparation agent.

        You receive:
        1. A user's data analysis question.
        2. A profile describing a dataset.
        3. A list of available Python operations.

        Your job is to create a sensible data preparation plan.

        Rules:
        - Use the user's question to determine which columns are relevant.
        - Identify a target column only if the task requires one.
        - Set target_required to false for tasks that do not require a target.
        - Identify identifier columns such as customer IDs when supported by
        the column names and dataset profile.
        - Do not automatically remove identifier columns.
        - Choose operations based on evidence in the profile and the question.
        - Prefer inspecting a problem before modifying the dataset.
        - Do not invent column names, operations or parameter names.
        - Do not recommend transformations without a clear reason.
        - Avoid unnecessary cleaning.
        - Do not assume missing values or duplicates must always be removed.
        - If the target column is ambiguous, use null for target_column.
        - Only select operations from the provided catalogue.
        - Explain the purpose of each operation briefly.
        - Distinguish between facts established by the dataset profile and checks
        that have not yet been executed.
        - Do not claim that an inspection found no issues unless the profile
        contains evidence supporting that conclusion.
        - An inspection operation means the check is planned, not completed.
        - For convert_types, supported target types are:
        numeric, integer, string, datetime, category.
        - Do not use raw pandas dtype names such as float64 or int64.

        Return a JSON object with exactly these fields:
        {
            "target_required": true,
            "target_column": "Churn",
            "identifier_columns": ["customer_id"],
            "relevant_columns": ["tenure", "MonthlyCharges", "Churn"],
            "operations": [
                {
                    "operation": "inspect_missing",
                    "params": {},
                    "reason": "Check relevant columns for missing values."
                }
            ],
            "reasoning": "Explain the overall preparation strategy."
        }

        For the convert_types operation, the "conversions" parameter
        must be a list of dictionaries.

        Each dictionary must contain:
        - "column": the name of the column to convert
        - "type": the target type

        Supported target types are:
        numeric, integer, string, datetime, category.

        Example:
        {
            "conversions": [
                {
                    "column": "TotalCharges",
                    "type": "numeric"
                },
                {
                    "column": "SeniorCitizen",
                    "type": "integer"
                }
            ]
        }

        Requirements:
        - Every operation MUST contain operation, params and reason.
        - The top-level object MUST contain every field shown above.
        - Use null for target_column when no target is needed or identifiable.
        - Use true or false for target_required.
        - Use empty lists when there are no identifier or relevant columns.
        - Use an empty operations list if no operations are necessary.
        - The example values are illustrative. Choose values based on the actual input.
    """

    user_prompt = f"""
        USER QUESTION:
        {problem_statement}

        DATASET PROFILE:
        {json.dumps(dataset_profile, indent=2)}

        AVAILABLE OPERATIONS:
        {json.dumps(get_operation_catalogue(), indent=2)}

        Create a data preparation plan based on this information.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("The LLM returned an empty planning response.")

    # Parse the JSON and validate it against the Pydantic schema.
    plan_data = json.loads(response_text)

    try:
        return DataPrepPlan.model_validate(plan_data)
    except Exception as e:
        print("LLM returned an invalid DataPrepPlan:")
        print(json.dumps(plan_data, indent=4))
        raise ValueError("The LLM response does not match the DataPrepPlan schema.") from e
    

"""
    Validate an LLM-generated data preparation plan.

    Checks:
    1. Referenced columns exist.
    2. The target column exists if required.
    3. Identifier and relevant columns exist.
    4. Operations are supported.
    5. Operation parameters are valid.

    Raises ValueError if the plan is invalid.
"""
def validate_plan(plan: DataPrepPlan, df: pd.DataFrame) -> None:
    from operations import OPERATION_CATALOGUE
    errors = []
    columns = set(df.columns)

    # Validate target column
    if plan.target_required:
        if not plan.target_column:
            errors.append("A target column is required, but none was specified.")
        elif plan.target_column not in columns:
            errors.append(f"Target column '{plan.target_column}' does not exist in the dataset.")
        
    # Validate identifier columns
    for col in plan.identifier_columns:
        if col not in columns:
            errors.append(f"Identifier column '{col}' does not exist.")
        
    # Validate relevant columns
    for col in plan.relevant_columns:
        if col not in columns:
            errors.append(f"Relevant column '{col}' does not exist.")

    # Validate operations and their params
    for idx, operation in enumerate(plan.operations):
        operation_name = operation.operation
        params = operation.params

        if operation_name not in OPERATION_CATALOGUE:
            errors.append(f"Operation {idx + 1}: '{operation_name}' is not supported.")
            continue

        # Validate cols explicitely referenced in params
        param_columns = params.get("columns", [])

        if isinstance(param_columns, str):
            param_columns = [param_columns]

        if not isinstance(param_columns, list):
            errors.append(f"Operation {idx + 1} ({operation_name}): 'columns' must be a list of column names.")
        else:
            for col in param_columns:
                if col not in columns:
                    errors.append(f"Operation {idx + 1} ({operation_name}): Column '{col}' does not exist.")

        # Validate columns specified in conversion mappings
        if operation_name == "convert_types":
            conversions = params.get("conversions")

            if not isinstance(conversions, list) or not conversions:
                errors.append(f"Operation {idx + 1} ({operation_name}): 'conversions' must be a non-empty list of dictionaries.")
                continue

            for conv in conversions:
                if not isinstance(conv, dict):
                    errors.append(f"Operation {idx + 1} ({operation_name}): Each conversion must be a dictionary.")
                    continue

                column = conv.get("column")
                target_type = conv.get("type")

                if not column:
                    errors.append(f"Operation {idx + 1} ({operation_name}): A conversion must specify a column.")
                    continue

                if column not in columns:
                    errors.append(f"Operation {idx + 1} ({operation_name}): Conversion column '{column}' does not exist.")

                if target_type not in {"numeric", "integer", "string", "datetime", "category"}:
                    errors.append(f"Operation {idx + 1} ({operation_name}): Unsupported target type '{target_type}'.")
                

    if errors:
        error_message = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Data preparation plan validation failed:\n{error_message}")

    print("Data preparation plan validation passed.")