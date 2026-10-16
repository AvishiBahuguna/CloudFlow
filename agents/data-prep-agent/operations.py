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
            "missing_percentage": round(float(df[col].isna().mean() * 100), 2),
        } for col in cols
    }
    return {"missing_values": result}


def inspect_duplicates(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    duplicate_rows = df.duplicated()
    return {
        "duplicate_rows": int(duplicate_rows.sum()),
        "duplicated_indices": [str(idx) for idx in df.index[duplicate_rows]]
    }

"""
    Returns
    {
        column_types: {
            col1: {
                "dtype":
                "inferred_type":
            }

            ...
        }
    }
    pd.api.types.infer_dtype(df[col].dropna(), skipna=True):
    Returns single type for consistant type of data and returns 'mixed' / 'mixed-integer 
    for mixed types in a col
"""
def inspect_types(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    cols = _require_cols(df, params)
    return {
        "column_types": {
            col: {
                "dtype": str(df[col].dtype),
                "inferred_type": str(pd.api.types.infer_dtype(df[col].dropna(), skipna=True)),
            } for col in cols
        }
    }

"""
    Returns
    invalid_values: {
        col1: {
            "invalid_count": int
            "invalid_indices": str
            "invalid_samples": str
        }

        ...
    }
"""
def inspect_invalid_values(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    rules = params.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("'rules' must map column names to validation rules.")
    
    _validate_cols(df, list(rules.keys()))
    
    result = {}
    for col, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Rule for '{col}' must be a dictionary.")
        
        invalid_mask = _get_invalid_mask(df[col], rule)
        result[col] = {
            "invalid_count": int(invalid_mask.sum()),
            "invalid_indices": [str(idx) for idx in df.index[invalid_mask]],
            "invalid_samples": [str(val) for val in df.loc[invalid_mask, col].head(5)],
        }

    return {"invalid_values": result}


# ============================================================
# TRANSFORMATION OPERATIONS
# ============================================================

"""
    Returns (result_df, details):
    details = {
        strategy: str,
        missing_before: int,
        missing_after: int,
        fill_value: str,
    }
"""

def handle_missing(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = _require_cols(df, params)
    strategy = params.get("strategy")
    valid_strategies = {"drop_rows", "mean", "median", "mode", "constant"}

    if strategy not in valid_strategies:
        raise ValueError(f"Unsupported missing-value strategy: {strategy}")
    
    result_df = df.copy()
    details = {}

    if strategy == "drop_rows":
        nrows_before = len(result_df)
        result_df = result_df.dropna(subset=cols).copy()
        details = {
            "strategy": strategy,
            "columns": cols,
            "rows_removed": nrows_before - len(result_df)
        }
        return result_df, details
    
    for col in cols:
        missing_before = int(df[col].isna().sum())

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(result_df[col]):
                raise ValueError(f"Mean imputation requires numeric column '{col}'.")
            fill_val = result_df[col].mean()

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(result_df[col]):
                raise ValueError(f"Median imputation requires numeric column '{col}'.")
            fill_val = result_df[col].median()

        elif strategy == "mode":
            modes = result_df[col].mode(dropna=True)
            if modes.empty:
                raise ValueError(f"Cannot calculate mode for entirely missing column '{col}'.")
            fill_val = modes.iloc[0]

        else: # strategy == "constant"
            if "value" not in params:
                raise ValueError("The constant strategy requires a 'value' parameter.")
            fill_val = params.get("value")

        if pd.isna(fill_val):
            raise ValueError(f"Cannot impute '{col}': no usable fill value.")
        
        result_df[col] = result_df[col].fillna(fill_val)
        details[col] = {
            "strategy": strategy,
            "missing_before": missing_before,
            "missing_after": int(result_df[col].isna().sum()),
            "fill_value": str(fill_val)
        }

    return result_df, details

"""
    Possible values of keep key in params:
    - "first" : Keeps the 1st occurance of the duplicate row
    - "last" : Keeps the last occurance of the duplicate row
    - False : Removes every row belonging to a group of duplicate rows
"""

def remove_duplicates(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep = params.get("keep", "first")          # Default to keeping first occurance if no param is provided
    if keep not in {"first", "last", False}:
        raise ValueError("'keep' must be 'first', 'last', or False.")

    rows_before = len(df)

    result_df = df.drop_duplicates(keep=keep).copy()

    return result_df, {
        "rows_before": rows_before,
        "rows_after": len(result_df),
        "duplicates_removed": rows_before - len(result_df),
        "keep": keep,
    }

"""
    params contains a key "conversions" which maps column names to supported target types
"""
def convert_types(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    conversions = params.get("conversions")
    if not isinstance(conversions, dict) or not conversions:
        raise ValueError("'conversions' must map column names to target types.")
    
    _validate_cols(df, list(conversions.keys()))

    result_df = df.copy()
    details = {}

    supported_types = {"numeric", "integer", "string", "datetime", "category"}

    for col, target_type in conversions.items():
        if target_type not in supported_types:
            raise ValueError(f"Unsupported target type '{target_type}' for '{col}'.")
        
        before_dtype = str(result_df[col].dtype)

        if target_type == "numeric":
            result_df[col] = pd.to_numeric(result_df[col], errors="raise")

        elif target_type == "interger":
            result_df[col] = pd.to_numeric(result_df[col], errors="raise")
            non_missing = result_df[col].dropna()

            if not non_missing.map(lambda value: float(value).is_integer()).all():
                raise ValueError(f"Column '{col}' contains non-integer values.")

            if result_df[col].isna().any():
                result_df[col] = result_df[col].astype("Int64")
            else:
                result_df[col] = result_df[col].astype("int64")

        elif target_type == "string":
            result_df[col] = result_df[col].astype("string")

        elif target_type == "datetime":
            result_df[col] = pd.to_datetime(result_df[col], errors="raise")

        elif target_type == "category":
            result_df[col] = result_df[col].astype("category")

        details[col] = {
            "before": before_dtype,
            "after": str(result_df[col].dtype),
        }

    return result_df, {"conversions": details}

"""
    Treat values violating explicit validation rules.

    Supported strategies:
    - replace_with_nan
    - drop_rows
    - clip

    For clip, provide numeric_range rules with min/max bounds.
"""
def handle_invalid_values(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rules = params.get("rules")
    strategy = params.get("strategy")

    if strategy not in {"replace_with_nan", "drop_rows", "clip"}:
        raise ValueError( f"Unsupported invalid-value strategy: {strategy}")
    
    if not isinstance(rules, dict) or not rules:
        raise ValueError("'rules' must map column names to validation rules.")
    
    _validate_cols(df, list(rules.keys()))

    result_df = df.copy()
    details = {}

    for col, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Rule for '{col}' must be a dictionary.")
        
        if strategy == "clip":
            if rule.get("type") != "numeric_range":
                raise ValueError("Clipping requires a numeric_range rule.")
            
            # Clipping non-numeric columns would be ambiguous.
            if not pd.api.types.is_numeric_dtype((result_df[col])):
                raise ValueError(f"Clipping requires numeric column '{col}'.")
            
            minimum = rule.get("min")
            maximum = rule.get("max")

            if minimum is None and maximum is None:
                raise ValueError(f"Rule for '{col}' must specify min or max.")
            
            invalid_mask = _get_invalid_mask(result_df[col], rule)

            result_df[col] = result_df[col].clip(lower=minimum, upper=maximum)

        else:
            invalid_mask = _get_invalid_mask(result_df[col], rule)

            if strategy == "replace_with_nan":
                result_df.loc[invalid_mask, col] = pd.NA

        details[col] = {
            "startegy": strategy,
            "invalid_values_found": int(invalid_mask.sum()),
        }

    # Recalculate invalid masks against the original rows, then
    # remove a row if any selected column violates its rule.
    if strategy == "drop_rows":
        invalid_rows = pd.Series(False, index=result_df.index)
        for col, rule in rules.items():
            invalid_rows |= _get_invalid_mask(result_df[col], rule)
        
        rows_before = len(result_df)

        result_df = result_df.loc[~invalid_rows].copy()

        details["rows_removed"] = rows_before - len(result_df)

    return result_df, {"results": details}



# ============================================================
# VALIDATION OPERATION
# ============================================================

def verify_dataset(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(col) for col in df.columns],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            str(col): int(df[col].isna().sum())
            for col in df.columns
        },
        "data_types": {
            str(col): str(df[col].dtype)
            for col in df.columns
        },
        "has_duplicate_column_names": bool(df.columns.duplicated().any())
    }




# ============================================================
# OPERATION DISPATCHER
# ============================================================

HANDLERS = {
    "inspect_missing": inspect_missing,
    "inspect_duplicates": inspect_duplicates,
    "inspect_types": inspect_types,
    "inspect_invalid_values": inspect_invalid_values,
    "handle_missing": handle_missing,
    "remove_duplicates": remove_duplicates,
    "convert_types": convert_types,
    "handle_invalid_values": handle_invalid_values,
    "verify_dataset": verify_dataset,
}

"""
    Execute a supported operation.

    Returns:
        updated_df: The current dataset, possibly transformed.
        result: A dictionary describing the operation's result.
"""
def execute_operation(df: pd.DataFrame, operation: str, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if operation not in OPERATION_CATALOGUE:
        raise ValueError(f"Unknown operation: {operation}")
    
    params = params or {}

    handler = HANDLERS[operation]
    result = handler(df, params)

    if OPERATION_CATALOGUE[operation]["modifies_data"]:
        updated_df, details = result
        return updated_df, details
    
    return df, result