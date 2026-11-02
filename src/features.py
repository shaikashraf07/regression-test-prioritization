"""Feature construction for historical regression-test execution records."""
from __future__ import annotations
import pandas as pd

NUMERIC_INPUTS = [
    "execution_time", "previous_failures", "code_change_size", "code_coverage",
    "module_criticality", "days_since_last_failure", "recent_failure_count",
    "module_failure_rate", "avg_module_execution_time", "coverage_gap",
]
CATEGORICAL_INPUTS = ["module_name", "change_type"]
TARGET = "failure_label"
REQUIRED = ["test_case_id", "module_name", "execution_time", "pass_fail_status",
            "previous_failures", "code_change_size", "code_coverage", "build_id"]

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create per-test features. Historical aggregates use prior rows only."""
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = raw.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="stable")
    df["failure_label"] = df["pass_fail_status"].astype(str).str.lower().map(
        {"fail": 1, "failed": 1, "1": 1, "pass": 0, "passed": 0, "0": 0})
    if df["failure_label"].isna().any():
        raise ValueError("pass_fail_status must contain pass or fail")
    for col in ["execution_time", "previous_failures", "code_change_size", "code_coverage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["module_criticality"] = pd.to_numeric(df.get("module_criticality", 1.0), errors="coerce")
    if "change_type" not in df.columns:
        df["change_type"] = "unknown"
    df["change_type"] = df["change_type"].fillna("unknown").astype(str)
    grouped = df.groupby("module_name", sort=False)
    # Shift before rolling/expanding so the current test outcome is never a feature.
    prior_fail = grouped["failure_label"].shift(1)
    df["recent_failure_count"] = prior_fail.groupby(df["module_name"]).transform(
        lambda s: s.rolling(5, min_periods=1).sum()).fillna(0)
    df["module_failure_rate"] = prior_fail.groupby(df["module_name"]).transform(
        lambda s: s.expanding(min_periods=1).mean()).fillna(0)
    prior_time = grouped["execution_time"].shift(1)
    df["avg_module_execution_time"] = prior_time.groupby(df["module_name"]).transform(
        lambda s: s.expanding(min_periods=1).mean()).fillna(df["execution_time"].median())
    df["days_since_last_failure"] = pd.to_numeric(
        df.get("days_since_last_failure", 30), errors="coerce")
    df["coverage_gap"] = (100 - df["code_coverage"]).clip(lower=0)
    df["module_name"] = df["module_name"].fillna("unknown").astype(str)
    return df

def model_columns():
    return NUMERIC_INPUTS + CATEGORICAL_INPUTS
