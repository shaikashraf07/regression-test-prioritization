"""Compare prioritization order with a random-order baseline using APFD."""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from features import build_features
from prioritize import prioritize
ROOT = Path(__file__).resolve().parents[1]

def apfd(order_labels):
    n = len(order_labels); total = int(np.sum(order_labels))
    if total == 0 or n == 0: return float("nan")
    positions = np.flatnonzero(np.asarray(order_labels) == 1) + 1
    return float(1 - positions.sum()/(n*total) + 1/(2*n))

if __name__ == "__main__":
    raw = pd.read_csv(ROOT / "data" / "test_execution_history.csv")
    # Evaluation is a demonstration on a held-out tail, not a production benchmark.
    test = raw.tail(max(100, len(raw)//5)).copy()
    model = joblib.load(ROOT / "models" / "failure_predictor.joblib")
    ranked = prioritize(test, model)
    baseline = test.sample(frac=1, random_state=42)
    result = {"prioritized_apfd": apfd(ranked["failure_label"] if "failure_label" in ranked else
               build_features(ranked)["failure_label"]),
              "random_order_apfd": apfd(build_features(baseline)["failure_label"])}
    out = ROOT / "reports" / "apfd_comparison.json"
    out.write_text(__import__("json").dumps(result, indent=2))
    print(result)
