"""Generate a reproducible demonstration dataset (not real production evidence)."""
import numpy as np
import pandas as pd
from pathlib import Path

def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    modules = np.array(["catalog", "cart", "checkout", "payment", "user", "search", "orders"])
    criticality_map = {"catalog": 1.0, "cart": 1.2, "checkout": 1.5, "payment": 1.7,
                       "user": 0.8, "search": 1.0, "orders": 1.3}
    module = rng.choice(modules, n, p=[.16,.14,.18,.13,.13,.14,.12])
    execution = np.maximum(.2, rng.lognormal(1.4, .65, n))
    prev = rng.poisson(.7, n)
    churn = rng.poisson(18, n)
    coverage = rng.uniform(45, 100, n)
    change_type = rng.choice(["feature", "bugfix", "refactor", "dependency", "config"], n)
    days = rng.integers(0, 90, n)
    criticality = np.array([criticality_map[m] for m in module])
    logit = -3.1 + .42*prev + .018*churn + .018*(100-coverage) + .55*(criticality-1) \
            + .35*(change_type == "dependency") + .25*(change_type == "refactor")
    probability = 1/(1+np.exp(-logit))
    failed = rng.binomial(1, probability)
    start = pd.Timestamp("2025-01-01")
    ts = start + pd.to_timedelta(np.arange(n), unit="h")
    out = pd.DataFrame({
        "test_case_id": [f"TC-{i+1:05d}" for i in range(n)], "module_name": module,
        "execution_time": execution.round(2), "pass_fail_status": np.where(failed==1,"fail","pass"),
        "previous_failures": prev, "code_change_size": churn, "code_coverage": coverage.round(1),
        "build_id": [f"BUILD-{i//20+1:04d}" for i in range(n)], "timestamp": ts,
        "module_criticality": criticality, "change_type": change_type,
        "days_since_last_failure": days,
    })
    return out

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data" / "test_execution_history.csv"
    path.parent.mkdir(exist_ok=True)
    generate().to_csv(path, index=False)
    print(f"Created {path}")
