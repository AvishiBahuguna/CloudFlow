# Defines what data gets passed form 1 agent to the next

from pydantic import BaseModel
from typing import Optional

# For data prep agent
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    n_unique: int
    sample_values: list[str]

class DataPrepResult(BaseModel):
    rows_before: int
    rows_after: int
    duplicates_removed: int
    issues_found: list[str]
    cleaning_steps: list[str]
    cleaned_dataset_path: list[str]

class PipelineState(BaseModel):
    run_id: int
    problem_stmt: str
    dataset_path: str
    target_col: Optional[str] = None
    data_prep: Optional[DataPrepResult] = None