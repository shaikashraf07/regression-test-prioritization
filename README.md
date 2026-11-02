# Machine Learning-Based Regression Test Prioritization and Intelligent CI/CD Optimization

A runnable academic prototype implementing the three project modules:

1. **Test Data Curation & Feature Engineering** – reads test execution history and derives historical module risk, coverage gap, execution-time and change-related features.
2. **Machine Learning-Based Regression Test Prioritization** – trains a Random Forest classifier to estimate failure probability and ranks tests using `probability × impact / execution_time`.
3. **Intelligent CI/CD Optimization & DevOps Analytics** – exports a prioritized queue, exposes a Streamlit dashboard, and demonstrates an automated GitHub Actions workflow.

> The bundled generator creates **synthetic demonstration data**. It is not evidence of real e-commerce production performance. Replace it with your own historical test-run records before reporting experimental results.

## 1. Requirements

- Python 3.10 or newer
- VS Code / terminal
- Internet for first-time package installation

## 2. Setup (Windows PowerShell)

```powershell
cd regression_test_prioritization
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Generate demo test history

```powershell
python src/generate_data.py
```

Creates `data/test_execution_history.csv` with test ID, module, execution time, pass/fail status, prior failures, code-change size, coverage, build ID, timestamp, criticality and change type.

## 4. Train the failure prediction model

```powershell
python src/train.py
```

Saves `models/failure_predictor.joblib` and evaluation metrics in `reports/metrics.json`. Metrics include accuracy, precision, recall, F1, ROC-AUC and average precision.

## 5. Generate the prioritized test queue

```powershell
python src/prioritize.py
```

Output: `reports/prioritized_tests.csv`. Tests with higher estimated failure risk, higher module impact and lower execution time receive higher priority.

## 6. Compare ordering with random baseline

```powershell
python src/evaluate.py
```

Writes an APFD comparison to `reports/apfd_comparison.json`. APFD is a test-ordering metric; it is meaningful only when failure labels are available for the evaluated test set. This demonstration evaluation is not a substitute for a leakage-safe temporal production evaluation.

## 7. Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal. Upload a CSV with the same required columns to rank another test batch.

## 8. Expected CSV schema

Required: `test_case_id`, `module_name`, `execution_time`, `pass_fail_status`, `previous_failures`, `code_change_size`, `code_coverage`, `build_id`.

Optional: `timestamp`, `module_criticality`, `change_type`, `days_since_last_failure`.

Use `pass`/`fail` in `pass_fail_status`. Each row represents one test execution. Historical rows should be ordered chronologically when timestamps are available. Do not include the current run's outcome or any post-execution information in predictors.

## 9. GitHub Actions CI integration

The workflow at `.github/workflows/ci.yml` installs dependencies, generates demo data, trains, ranks and runs tests on pushes and pull requests. For a real CI environment, replace the demo-data generation step with an artifact download or export from your test management/CI system. In a production design, train the model offline and load a versioned model artifact rather than retraining on every pull request.

## 10. Module mapping

- Module 1: `src/features.py`, `src/generate_data.py`
- Module 2: `src/train.py`, `src/prioritize.py`, `src/evaluate.py`
- Module 3: `app.py`, `.github/workflows/ci.yml`, `reports/`

## 11. Important implementation notes

- The target is `failure_label` (1 = fail, 0 = pass).
- The model estimates test failure probability; it does not guarantee a failure.
- The impact/time weighting is a transparent prioritization heuristic, not a learned business-impact model.
- For a defensible final evaluation, split by time/build (train on earlier builds, test on later builds), compare against original order and random order, report APFD and runtime, and avoid training/test contamination.
- A true CI optimization should execute the ranked tests in the CI runner. This prototype generates the queue and provides a GitHub Actions demonstration; it does not control a real e-commerce application or provision CI infrastructure automatically.
