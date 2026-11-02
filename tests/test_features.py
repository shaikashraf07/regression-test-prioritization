import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build_features

def sample():
    return pd.DataFrame({"test_case_id":["a","b"],"module_name":["cart","cart"],
      "execution_time":[2,3],"pass_fail_status":["fail","pass"],"previous_failures":[1,0],
      "code_change_size":[5,2],"code_coverage":[80,90],"build_id":["1","2"]})

def test_label_and_historical_features():
    df = build_features(sample())
    assert df.failure_label.tolist() == [1,0]
    assert df.recent_failure_count.iloc[0] == 0
    assert df.recent_failure_count.iloc[1] == 1

def test_no_missing_model_columns():
    df = build_features(sample())
    assert df[["module_failure_rate","coverage_gap","avg_module_execution_time"]].notna().all().all()
