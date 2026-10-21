import re
from typing import get_args

import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from shared.eda_schemas import (
    ColumnInterpretation,
    ColumnRole,
)


def _is_real_numeric(
    series: pd.Series,
) -> bool:
    return (
        is_numeric_dtype(series.dtype)
        and not is_bool_dtype(series.dtype)
        and not is_complex_dtype(series.dtype)
    )

def _interpret_column(
    name: str,
    series: pd.Series,
) -> ColumnInterpretation:
    observed = series.dropna()

    unique_count = int(
        observed.nunique()
    )

    warnings = []

    if observed.empty:
        role = "unknown"
        reason = (
            "No observed values; the analysis "
            "role cannot be inferred."
        )

    elif is_bool_dtype(series.dtype):
        role = "boolean"
        reason = (
            "Pandas stores Boolean values."
        )

    elif is_datetime64_any_dtype(series.dtype):
        role = "datetime"
        reason = (
            "Pandas already stores dates "
            "or timestamps."
        )

    else:
        normalized_name = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            name,
        ).casefold()

        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized_name,
        ).strip("_")

        id_name = (
            normalized_name in {
                "id",
                "uuid",
                "guid",
            }
            or normalized_name.endswith("_id")
        )

        if id_name:
            role = "identifier"
            reason = (
                "The name suggests an identifier; "
                "review this inference."
            )

            if unique_count < len(observed):
                warnings.append(
                    "Identifier values repeat; "
                    "records may describe repeated "
                    "observations of an entity."
                )

        elif isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            role = "categorical"
            reason = (
                "Pandas explicitly stores categories."
            )

        elif _is_real_numeric(series):
            role = "numeric"
            reason = (
                "Pandas stores real numerical values."
            )

            if unique_count <= 10:
                warnings.append(
                    "Few distinct numerical values; "
                    "review whether this is a "
                    "coded category."
                )

        elif (
            is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        ):
            strings = observed.astype("string")

            iso_shape = strings.str.fullmatch(
                r"\d{4}-\d{2}-\d{2}(?:[T ].+)?"
            )

            if bool(iso_shape.all()):
                parsed = pd.to_datetime(
                    strings,
                    format="ISO8601",
                    errors="coerce",
                    utc=True,
                )

                if bool(parsed.notna().all()):
                    role = "datetime"
                    reason = (
                        "All observed strings are "
                        "valid ISO-style dates "
                        "or timestamps."
                    )

                else:
                    role = "categorical"
                    reason = (
                        "Date-shaped strings contain "
                        "invalid or unsupported dates."
                    )

                    warnings.append(
                        "Review date values and their "
                        "intended format during "
                        "preparation."
                    )

            elif (
                float(
                    strings.str.len().mean()
                ) >= 80
                or float(
                    strings.str.split().str.len().mean()
                ) >= 10
            ):
                role = "text"
                reason = (
                    "Long or multiword passages "
                    "suggest free text; review "
                    "this heuristic."
                )

            else:
                role = "categorical"
                reason = (
                    "Short labels are provisionally "
                    "treated as categories."
                )

                if strings.str.contains(
                    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                    regex=True,
                ).any():
                    warnings.append(
                        "Slash-formatted dates may "
                        "be present; select an "
                        "explicit format during "
                        "preparation."
                    )

        else:
            role = "unknown"
            reason = (
                "This dtype is not supported by "
                "automatic chart selection yet."
            )

    if role == "categorical" and unique_count > 20:
        warnings.append(
            "High cardinality; use top-value charts "
            "or skip crowded comparisons."
        )

    return ColumnInterpretation(
        dtype=str(series.dtype),
        role=role,
        non_missing_count=len(observed),
        unique_count=unique_count,
        reason=reason,
        warnings=warnings,
    )
def interpret_columns(
    df: pd.DataFrame,
    *,
    role_overrides: dict[
        str,
        ColumnRole,
    ] | None = None,
) -> dict[str, ColumnInterpretation]:
    if df.empty:
        raise ValueError(
            "The dataset must contain "
            "rows and columns."
        )

    if not df.columns.is_unique or any(
        not isinstance(column, str)
        or not column.strip()
        for column in df.columns
    ):
        raise ValueError(
            "Column names must be unique, "
            "non-empty strings."
        )

    overrides = role_overrides or {}

    unknown_columns = (
        set(overrides) - set(df.columns)
    )

    if unknown_columns:
        raise ValueError(
            "Overrides reference missing columns: "
            f"{sorted(unknown_columns)}"
        )

    allowed_roles = set(
        get_args(ColumnRole)
    )

    interpretations = {}

    for name in df.columns:
        interpretation = _interpret_column(
            name,
            df[name],
        )

        if name in overrides:
            role = overrides[name]

            if role not in allowed_roles:
                raise ValueError(
                    f"Unsupported role {role!r} "
                    f"for {name!r}."
                )

            if (
                role == "numeric"
                and not _is_real_numeric(df[name])
            ):
                raise ValueError(
                    f"{name!r} must have a real "
                    "numeric dtype before using "
                    "the numeric role."
                )

            if (
                role == "datetime"
                and interpretation.role != "datetime"
            ):
                raise ValueError(
                    f"Prepare {name!r} with an "
                    "explicit date format before "
                    "assigning the datetime role."
                )

            if (
                role == "boolean"
                and interpretation.unique_count > 2
            ):
                raise ValueError(
                    f"{name!r} has more than two "
                    "observed values; it cannot "
                    "use the Boolean role."
                )

            interpretation.role = role
            interpretation.source = "override"

            interpretation.reason = (
                "Analysis role explicitly selected "
                "by the caller; source values "
                "are unchanged."
            )

        interpretations[name] = interpretation

    return interpretations
