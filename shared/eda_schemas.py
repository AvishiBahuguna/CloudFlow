from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
)

ChartKind = Literal[
    "categorical_bar",
    "correlation_heatmap",
    "group_comparison",
    "numeric_distribution",
    "categorical_comparison",
    "time_row_counts",
    "time_numeric_mean",
]
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

    kind: ChartKind = "categorical_bar"

    column: str | None = None

    columns: list[str] = Field(
        default_factory=list
    )

    path: str
    title: str
    description: str



class ChartConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_charts: int = Field(
        default=12,
        ge=0,
        le=30,
        strict=True,
    )

    max_per_kind: int = Field(
        default=3,
        ge=1,
        le=12,
        strict=True,
    )


class ChartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ChartKind
    columns: list[str]

    source: Literal["dataset", "target", "time"]
    source_index: NonNegativeInt | None = None

    priority: NonNegativeInt
    usable_rows: NonNegativeInt
    reason: str
    path: str | None = None


class SkippedChart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ChartKind
    columns: list[str]
    reason: str


class ChartPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ChartConfig
    eligible_count: NonNegativeInt
    duplicates_removed: NonNegativeInt

    selected: list[ChartRequest] = Field(
        default_factory=list
    )

    skipped: list[SkippedChart] = Field(
        default_factory=list
    )

    notes: list[str] = Field(default_factory=list)

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

class CorrelationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["pearson"] = "pearson"

    min_pairs: int = Field(
        default=3,
        ge=3,
        strict=True,
    )

    columns: list[str] = Field(
        default_factory=list
    )

    matrix: dict[
        str,
        dict[str, FiniteFloat | None],
    ] = Field(default_factory=dict)

    pair_counts: dict[
        str,
        dict[str, NonNegativeInt],
    ] = Field(default_factory=dict)

    excluded_columns: dict[str, str] = Field(
        default_factory=dict
    )

    notes: list[str] = Field(
        default_factory=list
    )

class GroupStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    value_type: str
    count: NonNegativeInt
    mean: FiniteFloat
    median: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat


class GroupComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_column: str
    numeric_column: str
    usable_rows: NonNegativeInt
    excluded_rows: NonNegativeInt

    groups: list[GroupStatistics] = Field(
        default_factory=list
    )

    notes: list[str] = Field(default_factory=list)

class NumericDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finite_count: NonNegativeInt
    missing_count: NonNegativeInt
    infinite_count: NonNegativeInt

    q1: FiniteFloat | None = None
    median: FiniteFloat | None = None
    q3: FiniteFloat | None = None
    iqr: FiniteFloat | None = None

    lower_fence: FiniteFloat | None = None
    upper_fence: FiniteFloat | None = None
    lower_whisker: FiniteFloat | None = None
    upper_whisker: FiniteFloat | None = None

    potential_outlier_count: NonNegativeInt | None = None

    potential_outlier_percentage: FiniteFloat | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    outlier_sample: list[FiniteFloat] = Field(
        default_factory=list
    )

    histogram_edges: list[FiniteFloat] = Field(
        default_factory=list
    )

    histogram_counts: list[NonNegativeInt] = Field(
        default_factory=list
    )

    notes: list[str] = Field(default_factory=list)

class CategoryLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    value_type: str


class CategoricalComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_column: str
    comparison_column: str

    usable_rows: NonNegativeInt
    excluded_rows: NonNegativeInt

    group_values: list[CategoryLabel]
    comparison_values: list[CategoryLabel]

    group_counts: list[NonNegativeInt]

    counts: list[list[NonNegativeInt]]
    percentages: list[list[FiniteFloat]]

    notes: list[str] = Field(
        default_factory=list
    )
class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str = Field(
        min_length=1,
        strict=True,
    )

    task_type: Literal[
        "classification",
        "regression",
    ]


class TargetCorrelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str

    coefficient: FiniteFloat | None = Field(
        default=None,
        ge=-1,
        le=1,
    )

    pair_count: NonNegativeInt


class TargetAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TargetConfig

    missing_count: NonNegativeInt
    usable_rows: NonNegativeInt
    excluded_rows: NonNegativeInt
    unique_count: NonNegativeInt

    class_summary: CategoricalSummary | None = None
    numeric_summary: NumericSummary | None = None

    numeric_correlations: list[TargetCorrelation] = Field(
        default_factory=list
    )

    group_comparisons: list[GroupComparison] = Field(
        default_factory=list
    )

    categorical_comparisons: list[
        CategoricalComparison
    ] = Field(default_factory=list)

    plot_paths: list[str] = Field(
        default_factory=list
    )

    notes: list[str] = Field(
        default_factory=list
    )

class TimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_column: str = Field(
        min_length=1,
        strict=True,
    )

    frequency: Literal[
        "day",
        "week",
        "month",
        "year",
    ] = "month"

    max_periods: int = Field(
        default=120,
        ge=2,
        le=240,
        strict=True,
    )


class TimeNumericTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usable_rows: NonNegativeInt
    excluded_rows: NonNegativeInt

    counts: list[NonNegativeInt]
    means: list[FiniteFloat | None]

    notes: list[str] = Field(default_factory=list)


class TimeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TimeConfig

    missing_count: NonNegativeInt
    invalid_count: NonNegativeInt
    usable_rows: NonNegativeInt
    excluded_rows: NonNegativeInt

    first_date: str | None = None
    last_date: str | None = None

    period_count: NonNegativeInt = 0
    observed_period_count: NonNegativeInt = 0

    period_starts: list[str] = Field(
        default_factory=list
    )

    row_counts: list[NonNegativeInt] = Field(
        default_factory=list
    )

    numeric_trends: dict[
        str,
        TimeNumericTrend,
    ] = Field(default_factory=dict)

    plot_paths: list[str] = Field(
        default_factory=list
    )

    notes: list[str] = Field(default_factory=list)
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

    correlation_summary: CorrelationSummary | None = None
    group_comparisons: list[GroupComparison] = Field(
        default_factory=list
    )

    group_comparison_notes: list[str] = Field(
        default_factory=list
    )
    numeric_distributions: dict[
        str,
        NumericDistribution,
    ] = Field(default_factory=dict)

    distribution_notes: list[str] = Field(
        default_factory=list
    )
    categorical_comparisons: list[
        CategoricalComparison
    ] = Field(default_factory=list)

    categorical_comparison_notes: list[str] = Field(
        default_factory=list
    )
    target_analysis: TargetAnalysis | None = None
    time_analysis: TimeAnalysis | None = None
    chart_plan: ChartPlan | None = None

    