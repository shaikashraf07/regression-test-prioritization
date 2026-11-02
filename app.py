from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.prioritize import prioritize

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Regression Test Intelligence", layout="wide")
st.title("Regression Test Prioritization Dashboardpip install -r requirements.txt")
st.caption("ML-assisted regression test ordering for CI/CD feedback. Demonstration data is synthetic.")
model_path = ROOT / "models" / "failure_predictor.joblib"
if not model_path.exists():
    st.warning("Model not found. Run `python src/generate_data.py` and `python src/train.py` first.")
    st.stop()
model = joblib.load(model_path)
upload = st.file_uploader("Upload test execution history CSV", type=["csv"])
source = pd.read_csv(upload) if upload else ROOT / "data" / "test_execution_history.csv"
if isinstance(source, Path):
    if not source.exists():
        st.error("Demo dataset missing. Generate it using the command in README.")
        st.stop()
    raw = pd.read_csv(source)
else:
    raw = source
ranked = prioritize(raw, model)
c1,c2,c3 = st.columns(3)
c1.metric("Test cases", f"{len(ranked):,}")
c2.metric("High-risk tests (p ≥ 0.5)", int((ranked.predicted_failure_probability >= .5).sum()))
c3.metric("Mean predicted failure risk", f"{ranked.predicted_failure_probability.mean():.1%}")
st.subheader("Prioritized execution queue")
cols = ["priority_rank", "test_case_id", "module_name", "execution_time", "predicted_failure_probability", "priority_score"]
st.dataframe(ranked[cols].round(4), use_container_width=True, hide_index=True)
st.download_button("Download prioritized queue", ranked.to_csv(index=False).encode(),
                   "prioritized_tests.csv", "text/csv")
st.subheader("Risk by module")
st.bar_chart(ranked.groupby("module_name").predicted_failure_probability.mean().sort_values(ascending=False))
