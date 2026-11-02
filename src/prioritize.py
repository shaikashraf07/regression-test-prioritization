from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
try:
    from .features import build_features, NUMERIC_INPUTS, CATEGORICAL_INPUTS
except ImportError:
    from features import build_features, NUMERIC_INPUTS, CATEGORICAL_INPUTS

ROOT = Path(__file__).resolve().parents[1]

def prioritize(raw: pd.DataFrame, model, impact_weight=None):
    featured = build_features(raw)
    X = featured[NUMERIC_INPUTS + CATEGORICAL_INPUTS]
    probability = model.predict_proba(X)[:, 1]
    impact = featured["module_criticality"].fillna(1.0) if impact_weight is None else impact_weight
    runtime = featured["execution_time"].clip(lower=.1)
    featured["predicted_failure_probability"] = probability
    featured["priority_score"] = probability * impact / runtime
    featured["priority_rank"] = featured["priority_score"].rank(method="first", ascending=False).astype(int)
    return featured.sort_values(["priority_score", "predicted_failure_probability"], ascending=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "test_execution_history.csv"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "prioritized_tests.csv"))
    args = parser.parse_args()
    model_path = ROOT / "models" / "failure_predictor.joblib"
    if not model_path.exists():
        raise SystemExit("Model missing. Run: python src/train.py")
    ranked = prioritize(pd.read_csv(args.input), joblib.load(model_path))
    Path(args.output).parent.mkdir(exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(ranked[["test_case_id", "module_name", "predicted_failure_probability", "priority_score", "priority_rank"]].head(20).to_string(index=False))
