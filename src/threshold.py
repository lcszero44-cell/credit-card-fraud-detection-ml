"""Configurable probability threshold wrapper for fraud risk decisioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class FraudDecision:
    """Structured fraud risk verdict for an individual transaction."""

    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_tier: str
    recommended_action: str


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wraps any probabilistic classifier/pipeline with a configurable decision threshold.

    In highly imbalanced credit card fraud detection (~0.17% positive rate), the default
    0.50 cutoff is rarely optimal: lowering the threshold (e.g., to ``0.30``) significantly
    increases fraud recall while keeping false positives operationally manageable.

    Parameters
    ----------
    estimator : sklearn-compatible estimator or Pipeline
        Base estimator exposing ``fit`` and ``predict_proba``.
    threshold : float, default=0.30
        Probability cutoff above or equal to which a transaction is flagged as fraud (1).
    """

    def __init__(self, estimator: Any, threshold: float = 0.30) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def _validate_threshold(self) -> None:
        if not (0.0 < float(self.threshold) < 1.0):
            raise ValueError(f"threshold must lie strictly in (0, 1); got {self.threshold!r}")

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "ThresholdClassifier":
        """Fit a fresh clone of the underlying estimator."""
        self._validate_threshold()
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Delegate class probability estimation to the fitted estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary fraud labels using ``P(Class=1) >= threshold``."""
        self._validate_threshold()
        probs = self.predict_proba(X)[:, 1]
        return (probs >= float(self.threshold)).astype(int)

    def predict_decisions(self, X: Any) -> list[FraudDecision]:
        """Generate structured risk verdicts with tiered operational actions."""
        self._validate_threshold()
        probs = self.predict_proba(X)[:, 1]
        decisions: list[FraudDecision] = []
        thr = float(self.threshold)

        for prob in probs:
            p = float(prob)
            is_fraud = p >= thr
            if p >= max(0.80, thr):
                tier, action = "CRITICAL", "BLOCK_AND_NOTIFY_CARDHOLDER"
            elif p >= thr:
                tier, action = "HIGH", "BLOCK_OR_STEP_UP_CHALLENGE"
            elif p >= thr * 0.5:
                tier, action = "MEDIUM", "MONITOR_AND_LOG"
            else:
                tier, action = "LOW", "APPROVE"
            decisions.append(
                FraudDecision(
                    fraud_probability=round(p, 6),
                    is_fraud=is_fraud,
                    threshold=thr,
                    risk_tier=tier,
                    recommended_action=action,
                )
            )
        return decisions
