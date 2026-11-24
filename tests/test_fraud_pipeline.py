"""Unit test suite for the credit card fraud detection pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FraudFeatureEngineer
from src.inference import build_demo_model, normalize_payload, score_transactions
from src.threshold import ThresholdClassifier


def _make_synthetic_transactions(n_rows: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_fraud = max(12, int(n_rows * 0.05))
    n_legit = n_rows - n_fraud

    v_legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    v_fraud = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    v_fraud[:, 13] -= 5.5  # V14
    v_fraud[:, 16] -= 5.0  # V17
    v_fraud[:, 11] -= 4.2  # V12

    X_v = np.vstack([v_legit, v_fraud])
    df = pd.DataFrame(X_v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0.0, 172800.0, size=n_rows)))
    df["Amount"] = np.concatenate(
        [rng.exponential(45.0, size=n_legit), rng.exponential(220.0, size=n_fraud)]
    )
    y = np.array([0] * n_legit + [1] * n_fraud, dtype=int)
    return df, y


class TestFraudFeatureEngineer(unittest.TestCase):
    def test_amount_log_and_temporal_features(self) -> None:
        df, _ = _make_synthetic_transactions(40, seed=7)
        engineer = FraudFeatureEngineer(velocity_window=5, drop_raw_time_amount=True)
        transformed = engineer.fit_transform(df)

        self.assertIn("Amount_log", transformed.columns)
        self.assertIn("Hour_of_day", transformed.columns)
        self.assertIn("Hour_sin", transformed.columns)
        self.assertIn("Hour_cos", transformed.columns)
        self.assertIn("Time_diff_prev", transformed.columns)
        self.assertIn("Tx_velocity_window", transformed.columns)
        self.assertNotIn("Time", transformed.columns)
        self.assertNotIn("Amount", transformed.columns)

        expected_log = np.log1p(df["Amount"].to_numpy())
        np.testing.assert_allclose(transformed["Amount_log"].to_numpy(), expected_log, rtol=1e-6)
        self.assertTrue(((transformed["Hour_of_day"] >= 0.0) & (transformed["Hour_of_day"] < 24.0)).all())

    def test_single_row_inference_uses_train_fallback_without_nan(self) -> None:
        df, _ = _make_synthetic_transactions(50, seed=11)
        engineer = FraudFeatureEngineer()
        engineer.fit(df)

        single = df.iloc[[0]].copy()
        transformed = engineer.transform(single)
        self.assertEqual(len(transformed), 1)
        self.assertFalse(transformed.isna().any().any())
        self.assertEqual(list(transformed.columns), list(engineer.get_feature_names_out()))


class TestLeakagePreventionAndThresholding(unittest.TestCase):
    def test_scaler_fits_strictly_on_train_split(self) -> None:
        df, y = _make_synthetic_transactions(240, seed=42)
        X_train, X_test, y_train, _ = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )

        # Inject extreme outlier exclusively in X_test to verify scaler never observes test stats
        X_test_shifted = X_test.copy()
        X_test_shifted["Amount"] = X_test_shifted["Amount"] + 1_000_000.0

        pipe = Pipeline(
            steps=[
                ("features", FraudFeatureEngineer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)

        train_engineered = pipe.named_steps["features"].transform(X_train)
        expected_means = train_engineered.mean(axis=0).to_numpy()
        np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, expected_means, rtol=1e-6)

        # Verify prediction succeeds on shifted test without altering fitted scaler stats
        _ = pipe.predict_proba(X_test_shifted)
        np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, expected_means, rtol=1e-6)

    def test_threshold_classifier_lowers_cutoff_to_0_30(self) -> None:
        df, y = _make_synthetic_transactions(300, seed=19)
        X_train, X_test, y_train, _ = train_test_split(
            df, y, test_size=0.25, random_state=19, stratify=y
        )

        base_pipe = Pipeline(
            steps=[
                ("features", FraudFeatureEngineer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=19)),
            ]
        )

        clf_50 = ThresholdClassifier(estimator=base_pipe, threshold=0.50).fit(X_train, y_train)
        clf_30 = ThresholdClassifier(estimator=base_pipe, threshold=0.30).fit(X_train, y_train)

        probs = clf_30.predict_proba(X_test)[:, 1]
        preds_50 = clf_50.predict(X_test)
        preds_30 = clf_30.predict(X_test)

        np.testing.assert_array_equal(preds_30, (probs >= 0.30).astype(int))
        self.assertGreaterEqual(int(preds_30.sum()), int(preds_50.sum()))

        decisions = clf_30.predict_decisions(X_test.iloc[:5])
        self.assertEqual(len(decisions), 5)
        for dec, p in zip(decisions, probs[:5]):
            self.assertEqual(dec.is_fraud, bool(p >= 0.30))
            self.assertEqual(dec.threshold, 0.30)

    def test_invalid_threshold_raises_value_error(self) -> None:
        df, y = _make_synthetic_transactions(60, seed=3)
        bad_clf = ThresholdClassifier(estimator=LogisticRegression(), threshold=1.2)
        with self.assertRaises(ValueError):
            bad_clf.fit(df, y)

    def test_pipeline_serialization_roundtrip_and_inference(self) -> None:
        model = build_demo_model(threshold=0.30, random_state=42)
        payload = [
            {"Time": 42000.0, "Amount": 19.90, "V14": 0.4, "V17": 0.2, "V12": 0.3, "V10": 0.1},
            {"Time": 7250.0, "Amount": 499.00, "V14": -8.5, "V17": -7.8, "V12": -6.5, "V10": -6.0},
        ]
        df_in = normalize_payload(payload)
        probs_before = model.predict_proba(df_in)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "fraud_pipeline.joblib"
            joblib.dump(model, artifact_path)
            reloaded = joblib.load(artifact_path)

        probs_after = reloaded.predict_proba(df_in)
        np.testing.assert_allclose(probs_before, probs_after, rtol=1e-7)

        alerts = score_transactions(reloaded, payload)
        self.assertEqual(len(alerts), 2)
        self.assertFalse(alerts[0]["is_fraud"])
        self.assertTrue(alerts[1]["is_fraud"])
        self.assertIn(alerts[1]["risk_tier"], {"HIGH", "CRITICAL"})


if __name__ == "__main__":
    unittest.main()
