from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
)
from typing import Literal

class NumericSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finite_count: NonNegativeInt
    infinite_count: NonNegativeInt

    mean: FiniteFloat | None = None
    median: FiniteFloat | None = None
    std: FiniteFloat | None = None
    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None

class CategoryFrequency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    value_type: str
    count: NonNegativeInt
    percentage: FiniteFloat = Field(ge=0, le=100)


class CategoricalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    non_missing_count: NonNegativeInt
    unique_count: NonNegativeInt

    top_values: list[CategoryFrequency] = Field(
        default_factory=list
    )

    other_count: NonNegativeInt = 0

class PlotArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["categorical_bar"] = "categorical_bar"
    column: str
    path: str
    title: str
    description: str
class DatasetOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cells: NonNegativeInt
    missing_cells: NonNegativeInt

    missing_cell_percentage: FiniteFloat = Field(
        ge=0,
        le=100,
    )

    rows_with_missing: NonNegativeInt
    complete_rows: NonNegativeInt

    duplicate_rows: NonNegativeInt

    duplicate_row_percentage: FiniteFloat = Field(
        ge=0,
        le=100,
    )

    empty_columns: list[str] = Field(
        default_factory=list
    )

    constant_columns: list[str] = Field(
        default_factory=list
    )

    unique_counts: dict[str, NonNegativeInt] = Field(
        default_factory=dict
    )

    dtype_counts: dict[str, NonNegativeInt] = Field(
        default_factory=dict
    )

ColumnRole = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "identifier",
    "text",
    "unknown",
]


class ColumnInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    role: ColumnRole

    source: Literal[
        "inferred",
        "override",
    ] = "inferred"

    non_missing_count: NonNegativeInt
    unique_count: NonNegativeInt

    reason: str

    warnings: list[str] = Field(
        default_factory=list
    )

class CSVReadLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
        strict=True,
    )

    max_rows: int = Field(
        default=100_000,
        gt=0,
        strict=True,
    )

    max_columns: int = Field(
        default=150,
        gt=0,
        strict=True,
    )

    max_cells: int = Field(
        default=5_000_000,
        gt=0,
        strict=True,
    )
    
class EDAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_count: NonNegativeInt
    column_count: NonNegativeInt

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    other_columns: list[str] = Field(default_factory=list)

    missing_counts: dict[str, NonNegativeInt] = Field(
        default_factory=dict
    )

    missing_percentages: dict[str, FiniteFloat] = Field(
        default_factory=dict
    )

    numeric_summaries: dict[str, NumericSummary] = Field(
        default_factory=dict
    )
    categorical_summaries: dict[str, CategoricalSummary] = Field(
        default_factory=dict
    )
    plots: list[PlotArtifact] = Field(default_factory=list)
    dataset_overview: DatasetOverview | None = None
    column_interpretations: dict[
        str,
        ColumnInterpretation,
    ] = Field(default_factory=dict)