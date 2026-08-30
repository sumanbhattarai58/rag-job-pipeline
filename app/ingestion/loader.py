"""
Load the LF Jobs dataset from CSV into a validated pandas DataFrame.
"""
import pandas as pd

EXPECTED_COLUMNS = [
    "ID",
    "Job Category",
    "Job Title",
    "Company Name",
    "Publication Date",
    "Job Location",
    "Job Level",
    "Tags",
    "Job Description",
]


def load_jobs(csv_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {csv_path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df["Publication Date"] = pd.to_datetime(df["Publication Date"], errors="coerce", utc=True)

    return df 