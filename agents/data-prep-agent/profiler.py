# Turning the df into a compact summary the LLM can reason about

import pandas as pd
from shared.state import ColumnProfile

def profile_df(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles = []
    for col in df.columns:
        profiles.append(ColumnProfile(
            name= col,
            dtype= str(df[col].dtype),
            missing_count= int(df[col].isna().sum()),
            n_unique= int(df[col].nunique()),
            sample_values= [str(v) for v in df[col].dropna().unique()[:5]],
        ))
    return profiles

# df = pd.read_csv("~/CloudFlow/data/churn.csv")
# print(profile_df(df))