"""Production CLI for real-time credit card fraud risk scoring.

Examples
--------
Score a JSON payload using a trained pipeline artifact::

    python -m src.inference --model models/fraud_pipeline.joblib --json '{"Time": 406.0, "Amount": 149.62, "V1": -2.31, ...}'

Run an interactive demo check (uses ``models/fraud_pipeline.joblib`` if present, or trains
a lightweight demo pipeline if not yet saved)::

    python -m src.inference --demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FraudFeatureEngineer
from .threshold import ThresholdClassifier

EXPECTED_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def normalize_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a single transaction dict or list of dicts into an ordered DataFrame."""
    records = [payload] if isinstance(payload, dict) else list(payload)
    if not records:
        raise ValueError("Payload must contain at least one transaction record.")

    normalized_rows: list[dict[str, float]] = []
    for record in records:
        row = {col: float(record.get(col, 0.0)) for col in EXPECTED_COLUMNS}
        normalized_rows.append(row)
    return pd.DataFrame(normalized_rows, columns=EXPECTED_COLUMNS)


def build_demo_model(threshold: float = 0.30, random_state: int = 42) -> ThresholdClassifier:
    """Build a self-contained synthetic demo model when no serialized artifact exists."""
    rng = np.random.default_rng(random_state)
    n_legit, n_fraud = 950, 50

    legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    fraud = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    # Mimic the real ULB fraud signatures on V14, V17, V12, V10
    fraud[:, 13] -= 6.0  # V14
    fraud[:, 16] -= 5.5  # V17
    fraud[:, 11] -= 5.0  # V12
    fraud[:, 9] -= 4.5   # V10

    X_v = np.vstack([legit, fraud])
    time_col = np.concatenate(
        [rng.uniform(0, 172800, size=n_legit), rng.uniform(0, 172800, size=n_fraud)]
    )
    amount_col = np.concatenate(
        [rng.exponential(60.0, size=n_legit), rng.exponential(180.0, size=n_fraud)]
    )
    y = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(X_v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time_col)
    df["Amount"] = amount_col

    base_pipeline = Pipeline(
        steps=[
            ("features", FraudFeatureEngineer()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=60,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model = ThresholdClassifier(estimator=base_pipeline, threshold=threshold)
    model.fit(df, y)
    return model


def score_transactions(
    model: ThresholdClassifier,
    payload: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score one or more transaction dicts and return serializable risk alert dicts."""
    df = normalize_payload(payload)
    decisions = model.predict_decisions(df)
    results: list[dict[str, Any]] = []
    for idx, decision in enumerate(decisions):
        item = asdict(decision)
        item["transaction_index"] = idx
        item["amount"] = float(df.iloc[idx]["Amount"])
        results.append(item)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time credit card fraud risk scoring CLI."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fraud_pipeline.joblib"),
        help="Path to a serialized joblib ThresholdClassifier / Pipeline artifact.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Decision threshold override (default: 0.30).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON file containing a transaction object or list of transactions.",
    )
    parser.add_argument(
        "--json",
        dest="json_str",
        type=str,
        help="Inline JSON string containing a transaction object or list of transactions.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in demonstration comparing one legitimate and one high-risk transaction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.model.exists():
        loaded = joblib.load(args.model)
        if isinstance(loaded, ThresholdClassifier):
            loaded.threshold = args.threshold
            model = loaded
        else:
            model = ThresholdClassifier(estimator=loaded, threshold=args.threshold)
            model.estimator_ = loaded
            model.classes_ = getattr(loaded, "classes_", np.array([0, 1]))
    else:
        model = build_demo_model(threshold=args.threshold)

    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    elif args.json_str:
        payload = json.loads(args.json_str)
    elif args.demo:
        payload = [
            {"Time": 42000.0, "Amount": 24.90, "V14": 0.45, "V17": 0.22, "V12": 0.35, "V10": 0.15},
            {"Time": 7250.0, "Amount": 489.00, "V14": -8.20, "V17": -7.40, "V12": -6.30, "V10": -5.80},
        ]
    else:
        raise SystemExit("Provide --input <file.json>, --json '<payload>', or --demo.")

    alerts = score_transactions(model, payload)
    print(json.dumps(alerts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
