"""Train the FetalSignal research candidate-ranker from ADFECGDB EDF records.

Example:
    python -m scripts.train_candidate_ranker --records data/adfecgdb/r01.edf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pyedflib
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fetalsignal.ml import MODEL_PATH, candidate_features
from fetalsignal.signal_processing import extract_fetal_signal


def labels_for_candidates(candidates: np.ndarray, reference: np.ndarray, sample_rate: float) -> np.ndarray:
    tolerance = int(sample_rate * 0.08)
    return np.array([np.any(np.abs(reference - peak) <= tolerance) for peak in candidates], dtype=int)


def load_record(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    reader = pyedflib.EdfReader(str(path))
    try:
        signal = reader.readSignal(1)  # Abdomen_1; direct fetal ECG is never input to the model.
        sample_rate = float(reader.getSampleFrequency(1))
    finally:
        reader.close()
    annotation = wfdb.rdann(str(path), "qrs")
    return signal, annotation.sample, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    feature_sets, label_sets, record_names = [], [], []
    for path in args.records:
        signal, reference, sample_rate = load_record(path)
        result = extract_fetal_signal(signal, sample_rate)
        feature_sets.append(candidate_features(result.residual, result.fetal_peaks, sample_rate))
        label_sets.append(labels_for_candidates(result.fetal_peaks, reference, sample_rate))
        record_names.append(path.name)
    features, labels = np.vstack(feature_sets), np.concatenate(label_sets)
    if np.unique(labels).size < 2:
        raise RuntimeError("Training candidates need both reference-matched and unmatched examples.")
    split = max(1, int(features.shape[0] * 0.8))
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    model.fit(features[:split], labels[:split])
    prediction = model.predict(features[split:])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print({"model": str(MODEL_PATH), "records": record_names, "candidates": int(features.shape[0]), "positives": int(labels.sum()), "validation_precision": round(float(precision_score(labels[split:], prediction, zero_division=0)), 3), "validation_recall": round(float(recall_score(labels[split:], prediction, zero_division=0)), 3), "validation_f1": round(float(f1_score(labels[split:], prediction, zero_division=0)), 3)})


if __name__ == "__main__":
    main()
