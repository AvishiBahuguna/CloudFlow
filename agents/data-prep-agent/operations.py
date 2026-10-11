# Defines exactly what the Data Prep agent is capable of doing

from typing import Any
import pandas as pd

# ============================================================
# OPERATION CATALOGUE
# ============================================================

OPERATION_CATALOGUE = {
    # -------------------- Inspection --------------------

    "inspect_missing": {
        "category": "inspection",
        "description": "Inspect missing values in selected columns",
        "modifies_data": False,
        "params": ["columns"],
    },

    "inspect_duplicates": {
        "category": "inspection",
        "description": "Inspect duplicate rows in the dataset",
        "modifies_data": False,
        "params": [],
    },

    "inspect_types": {
        "category": "inspection",
        "description": "Inspect data types of selected columns",
        "modifies_data": False,
        "params": ["columns"],
    },

    "inspect_invalid_values": {
        "category": "inspection",
        "description": "Identify values that violate explicitely provided rules",
        "modifies_data": False,
        "params": ["rules"],
    },

    # -------------------- Identification --------------------

    "identify_target": {
        "category": "identification",
        "description": (
            "Identify potential target columns using the user's question, "
            "dataset characteristics and column names. Determine whether "
            "the question requires a target column, such as for prediction "
            "and validate candidate columns against the dataset."
        ),
        "modifies_data": False,
        "params": ["problem_stmt"],
    },

    "identify_identifiers": {
        "category": "identification",
        "description": (
            "Identify columns that may contain record identifiers"
        ),
        "modifies_data": False,
        "params": [],
    },

    # -------------------- Transformation -------------------

    "handle_missing": {
        "category": "transformation",
        "description": "Handle missing values using a selected strategy",
        "modifies_data": True,
        "params": ["columns", "strategy", "value"],
    },

    "remove_duplicates": {
        "category": "transformation",
        "description": "Remove duplicate row",
        "modifies_data": True,
        "params": ["keep"],
    },

    "convert_types": {
        "category": "transformation",
        "description": "Convert selected columns to specified data types",
        "modifies_data": True,
        "params": ["conversions"],
    },

    "handle_invalid_values": {
        "category": "transformation",
        "description": "Treat values that violate explicit rules",
        "modifies_data": True,
        "params": ["rules", "strategy"],
    },

    # -------------------- Validation --------------------

    "verify_dataset": {
        "category": "validation",
        "description": "Check dataset structure and report data-quality metrics",
        "modifies_data": False,
        "params": [],
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

""" Ensure every requested col exists in df """
def _validate_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns: {missing}")
    
""" Get requested cols, defaulting to all cols """
def _require_cols(df: pd.DataFrame, params: dict[str, Any]) -> list[str]:
    cols = params.get("columns")
    if cols is None:
        return list(df.columns)
    
    if not isinstance(cols, list):
        raise ValueError("'columns' must be a list")
    
    _validate_cols(df, cols)

    return cols

"""
    Identify numeric values outside explicitely supplied bounds.
    Agent should not assume that every unusual value is invalid.
    That's why inspect_invalid_values requires explicit rules.

    For example:
    rules = {
        "age": {
            "type": "numeric_range",
            "min": 0,
            "max": 120,
        }
    }

    Here,
    - series is a pandas Series of the form
      {idx, <col-values>} representing if the values in a col
    - invalid is a boolean tracking Series of the form {idx, True/False}
    True - invalid; False - valid
"""
def _check_numeric_bound(series: pd.Series, rule: dict[str, Any]) -> pd.Series:
    # Check if series has numeric values
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Numeric bounds require a numeric column, got {series.dtype}.")

    invalid = pd.Series(False, index=series.index)
    minimum = rule.get("min")
    maximum = rule.get("max")

    if minimum is not None:
        invalid |= series.notna() & (series < minimum)
    if maximum is not None:
        invalid |= series.notna() & (series > maximum)

    return invalid

"""
    Evaluate explicit validation rules.

    Supported rule types:
    - numeric_range
    - allowed_values
    - not_null
"""
def _get_invalid_mask(series: pd.Series, rule: dict[str, Any]) -> pd.Series:
    rule_type = rule.get("type")

    if rule_type == "numeric_range":
        return _check_numeric_bound(series, rule)
    
    if rule_type == "allowed_values":
        allowed = rule.get("values")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("'allowed_values' requires a non-empty 'values' list.")
        return series.notna() & ~series.isin(allowed)
    
    if rule_type == "not_null":
        return series.isna()
    
    raise ValueError(f"Unsupported validation rule: {rule_type}")


# ============================================================
# INSPECTION OPERATIONS
# ============================================================

"""
    Returns a dict of the form:
    {
        missing_values: {
            result: {
                col1: {
                    "missing_count" : int,
                    "missing_percentage": float,
                },

                ...
            }
        }
    }
"""
def inspect_missing(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    cols = _require_cols(df, params)
    result = {
        col: {
            "missing_count": int(df[col].isna().sum()),
            "missing_percentage": round(float(df[col].mean() * 100), 2),
        } for col in cols
    }
    return {"missing_values": result}


def inspect_duplicates(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    duplicate_rows = df.duplicated()
    return {
        "duplicate_rows": int(duplicate_rows.sum()),
        "duplicated_indices": [str(idx) for idx in df.index[duplicate_rows]]
    }
