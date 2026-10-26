from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# Data Preparation Agent Scemas

OperationName = Literal[
    "inspect_missing",
    "inspect_duplicates",
    "inspect_types",
    "inspect_invalid_values",
    "handle_missing",
    "remove_duplicates",
    "convert_types",
    "handle_invalid_values",
    "verify_dataset",
]

class PlannedOperation(BaseModel):
    operation: OperationName
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str

class DataPrepPlan(BaseModel):
    target_required: bool
    target_column: Optional[str] = None

    identifier_columns: list[str] = Field(default_factory=list)
    relevant_columns: list[str] = Field(default_factory=list)

    operations: list[PlannedOperation] = Field(default_factory=list)
    reasoning: str