"""Custom scikit-learn feature engineering transformer for credit card fraud detection.

Applies leak-free deterministic feature transformations:
- ``Amount_log = np.log1p(Amount)`` to compress heavy-tailed monetary distributions.
- Temporal behavioral features derived strictly from ``Time`` (elapsed seconds):
  * ``Hour_of_day``: continuous hour in ``[0, 24)``.
  * ``Hour_sin`` / ``Hour_cos``: cyclical encodings for smooth midnight wrap-around.
  * ``Time_diff_prev``: strictly causal elapsed seconds since the previous transaction.
  * ``Tx_velocity_window``: strictly causal rolling transaction rate over a trailing
    window of ``velocity_window`` transactions (captures fraud bursts without future leakage).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leak-free, sklearn-compatible feature engineer for credit card transactions.

    Parameters
    ----------
    velocity_window : int, default=10
        Number of preceding transactions used to compute the strictly causal
        trailing transaction velocity (transactions per minute).
    drop_raw_time_amount : bool, default=True
        Whether to drop raw ``Time`` and ``Amount`` columns after creating
        ``Amount_log`` and temporal behavioral features.
    """

    def __init__(
        self,
        velocity_window: int = 10,
        drop_raw_time_amount: bool = True,
    ) -> None:
        self.velocity_window = velocity_window
        self.drop_raw_time_amount = drop_raw_time_amount

    def fit(self, X: pd.DataFrame, y: Sequence | None = None) -> "FraudFeatureEngineer":
        """Learn input column metadata and reference median inter-arrival gap on train only.

        Using only the training fold to compute ``median_time_diff_`` guarantees zero
        data leakage when handling the very first transaction or single-row inference.
        """
        df = self._ensure_dataframe(X)
        self.input_features_: list[str] = list(df.columns)

        if "Time" in df.columns and len(df) > 1:
            sorted_time = np.sort(df["Time"].to_numpy(dtype=float))
            diffs = np.diff(sorted_time)
            positive_diffs = diffs[diffs >= 0]
            self.median_time_diff_: float = (
                float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 1.0
            )
        else:
            self.median_time_diff_ = 1.0

        # Precompute output feature names deterministically
        sample_out = self._transform_dataframe(df.iloc[: min(len(df), 2)].copy())
        self.feature_names_out_: list[str] = list(sample_out.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transaction features into engineered model-ready features."""
        df = self._ensure_dataframe(X)
        return self._transform_dataframe(df)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return output feature names for sklearn Pipeline & SHAP interoperability."""
        if hasattr(self, "feature_names_out_"):
            return np.asarray(self.feature_names_out_, dtype=object)
        raise AttributeError("FraudFeatureEngineer must be fitted before calling get_feature_names_out.")

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "input_features_"):
            return pd.DataFrame(X, columns=self.input_features_)
        return pd.DataFrame(X)

    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        fallback_diff = getattr(self, "median_time_diff_", 1.0)

        if "Amount" in out.columns:
            amount_clean = np.clip(out["Amount"].astype(float), a_min=0.0, a_max=None)
            out["Amount_log"] = np.log1p(amount_clean)

        if "Time" in out.columns:
            time_sec = out["Time"].astype(float)
            hour_of_day = (time_sec / 3600.0) % 24.0
            out["Hour_of_day"] = hour_of_day
            out["Hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
            out["Hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)

            # Strictly causal inter-arrival time & burst velocity computed in chronological order
            # and mapped back to the original index without looking at future transactions.
            if len(out) > 1:
                order = np.argsort(time_sec.to_numpy(), kind="mergesort")
                sorted_t = time_sec.to_numpy()[order]

                raw_diff = np.empty_like(sorted_t, dtype=float)
                raw_diff[0] = fallback_diff
                raw_diff[1:] = np.maximum(np.diff(sorted_t), 0.0)

                window = max(int(self.velocity_window), 1)
                velocity = np.empty_like(sorted_t, dtype=float)
                for idx in range(len(sorted_t)):
                    start_idx = max(0, idx - window)
                    count = idx - start_idx
                    if count == 0:
                        elapsed = max(fallback_diff, 1.0)
                        velocity[idx] = 60.0 / elapsed
                    else:
                        elapsed = max(sorted_t[idx] - sorted_t[start_idx], 1.0)
                        velocity[idx] = (count * 60.0) / elapsed

                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))
                out["Time_diff_prev"] = np.log1p(raw_diff[inv_order])
                out["Tx_velocity_window"] = np.log1p(velocity[inv_order])
            else:
                out["Time_diff_prev"] = np.log1p(max(fallback_diff, 0.0))
                out["Tx_velocity_window"] = np.log1p(60.0 / max(fallback_diff, 1.0))

        if self.drop_raw_time_amount:
            cols_to_drop = [c for c in ("Time", "Amount") if c in out.columns]
            if cols_to_drop:
                out = out.drop(columns=cols_to_drop)

        return out
