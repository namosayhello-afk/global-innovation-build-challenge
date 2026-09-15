"""Small supervised candidate-ranking model for research experiments.

The model ranks peaks proposed by the signal-processing baseline.  It does not
diagnose conditions, and it intentionally falls back to the baseline if no
trained model is bundled with the app.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fetal_candidate_ranker.joblib"


def candidate_features(signal: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Extract local, scale-independent features around proposed candidate peaks."""
    signal = np.asarray(signal, dtype=float)
    if peaks.size == 0:
        return np.empty((0, 6))
    scale = np.median(np.abs(signal - np.median(signal))) * 1.4826 + 1e-9
    normalized = (signal - np.median(signal)) / scale
    half_window = max(2, int(sample_rate * 0.035))
    features = []
    for index, peak in enumerate(peaks):
        start, end = max(0, peak - half_window), min(signal.size, peak + half_window + 1)
        local = normalized[start:end]
        before = (peak - peaks[index - 1]) / sample_rate if index else 0.0
        after = (peaks[index + 1] - peak) / sample_rate if index + 1 < peaks.size else 0.0
        center = normalized[peak]
        slope_left = center - normalized[max(0, peak - 2)]
        slope_right = normalized[min(signal.size - 1, peak + 2)] - center
        features.append([center, abs(center), np.std(local), np.mean(np.abs(local)), before, after + slope_left - slope_right])
    return np.asarray(features, dtype=float)


def score_candidates(signal: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray | None:
    """Return fetal-candidate probabilities, or None when no bundled model exists."""
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    return model.predict_proba(candidate_features(signal, peaks, sample_rate))[:, 1]


def select_model_candidates(signal: np.ndarray, peaks: np.ndarray, sample_rate: float, threshold: float = 0.2) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep model-supported candidates while preserving baseline safety fallback."""
    probabilities = score_candidates(signal, peaks, sample_rate)
    if probabilities is None:
        return peaks, None
    selected = peaks[probabilities >= threshold]
    # A small experimental model should never erase most of a recording's
    # baseline candidates when it encounters out-of-distribution noise.
    minimum_retained = max(3, int(np.ceil(peaks.size * 0.75)))
    return (selected if selected.size >= minimum_retained else peaks), probabilities
