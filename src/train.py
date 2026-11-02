from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import build_features, NUMERIC_INPUTS, CATEGORICAL_INPUTS, TARGET

ROOT = Path(__file__).resolve().parents[1]

def main(data_path=None):
    data_path = Path(data_path) if data_path else ROOT / "data" / "test_execution_history.csv"
    raw = pd.read_csv(data_path)
    df = build_features(raw)
    X, y = df[NUMERIC_INPUTS + CATEGORICAL_INPUTS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_INPUTS),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_INPUTS),
    ])
    model = Pipeline([("features", prep), ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "test_rows": int(len(y_test)), "positive_rate": float(y_test.mean()),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "failure_predictor.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    main()
