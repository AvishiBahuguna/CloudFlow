# Turning the df into a compact summary the LLM can reason about
import json
from pathlib import Path
import pandas as pd
from shared.state import ColumnProfile, DatasetProfiler


def profile_df(df: pd.DataFrame) -> DatasetProfiler:
    profiles = []
    for col in df.columns:
        profiles.append(ColumnProfile(
            name= str(col),
            dtype= str(df[col].dtype),
            missing_count = int((df[col].isna() | df[col].astype("string").str.strip().eq("").fillna(False)).sum()),
            n_unique= int(df[col].nunique()),
            sample_values= [str(v) for v in df[col].dropna().unique()[:5]],
        ))

    return DatasetProfiler(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        duplicate_rows= int(df.duplicated().sum()),
        duplicate_col_names= [str(col) for col in df.columns[df.columns.duplicated()].unique()],
        column_profiles= profiles
    )

def save_profile(profile: DatasetProfiler, output_path: str)->None:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(profile.model_dump(), file, indent=4)

    print(f"Profile saved to: {output}")



if __name__ == "__main__":
    dataset_path = Path("~/CloudFlow/data/churn.csv").expanduser()
    output_path = "~/CloudFlow/workspace/inital_profile.json"

    df = pd.read_csv(dataset_path)
    profile = profile_df(df)
    print(profile.model_dump_json(indent=4))
    save_profile(profile, output_path)