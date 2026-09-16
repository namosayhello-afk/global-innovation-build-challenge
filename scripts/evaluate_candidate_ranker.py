"""Run leave-one-record-out evaluation on public ADFECGDB records.

Example:
    python -m scripts.evaluate_candidate_ranker --records data/adfecgdb/r01.edf data/adfecgdb/r04.edf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fetalsignal.ml import MODEL_PATH, candidate_features
from fetalsignal.signal_processing import heart_rate_bpm, match_peaks
from scripts.train_candidate_ranker import labels_for_candidates, load_record
from fetalsignal.signal_processing import extract_fetal_signal


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def choose_candidates(peaks: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    selected = peaks[probabilities >= 0.2]
    minimum_retained = max(3, int(np.ceil(peaks.size * 0.75)))
    return selected if selected.size >= minimum_retained else peaks


def metric_summary(predicted: np.ndarray, reference: np.ndarray, sample_rate: float) -> dict[str, float | int | None]:
    metrics = match_peaks(predicted, reference, sample_rate)
    predicted_bpm = heart_rate_bpm(predicted, sample_rate)
    reference_bpm = heart_rate_bpm(reference, sample_rate)
    return {
        "candidate_beats": int(predicted.size),
        "reference_beats": int(reference.size),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "f1": round(float(metrics["f1"]), 4),
        "peak_error_ms": None if np.isnan(metrics["peak_error_ms"]) else round(float(metrics["peak_error_ms"]), 2),
        "candidate_bpm": None if predicted_bpm is None else round(predicted_bpm, 2),
        "reference_bpm": None if reference_bpm is None else round(reference_bpm, 2),
        "bpm_absolute_error": None if predicted_bpm is None or reference_bpm is None else round(abs(predicted_bpm - reference_bpm), 2),
    }


def mean_metric(records: list[dict], key: str) -> float | None:
    values = [record["ml_assisted"][key] for record in records if record["ml_assisted"][key] is not None]
    return None if not values else round(float(np.mean(values)), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/adfecgdb_leave_one_record_out.json"))
    args = parser.parse_args()
    if len(args.records) < 2:
        raise ValueError("Use at least two records for leave-one-record-out evaluation.")

    records = []
    for path in args.records:
        signal, reference, sample_rate = load_record(path)
        result = extract_fetal_signal(signal, sample_rate)
        records.append(
            {
                "name": path.name,
                "reference": reference,
                "sample_rate": sample_rate,
                "peaks": result.fetal_peaks,
                "features": candidate_features(result.residual, result.fetal_peaks, sample_rate),
                "labels": labels_for_candidates(result.fetal_peaks, reference, sample_rate),
            }
        )

    evaluations = []
    for test_index, test in enumerate(records):
        train_features = np.vstack([record["features"] for index, record in enumerate(records) if index != test_index])
        train_labels = np.concatenate([record["labels"] for index, record in enumerate(records) if index != test_index])
        model = make_model().fit(train_features, train_labels)
        probabilities = model.predict_proba(test["features"])[:, 1]
        ml_peaks = choose_candidates(test["peaks"], probabilities)
        evaluations.append(
            {
                "held_out_record": test["name"],
                "training_records": [record["name"] for index, record in enumerate(records) if index != test_index],
                "baseline": metric_summary(test["peaks"], test["reference"], test["sample_rate"]),
                "ml_assisted": metric_summary(ml_peaks, test["reference"], test["sample_rate"]),
            }
        )

    final_model = make_model().fit(np.vstack([record["features"] for record in records]), np.concatenate([record["labels"] for record in records]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    report = {
        "dataset": "PhysioNet Abdominal and Direct Fetal ECG Database v1.0.0 (ADFECDGB)",
        "dataset_doi": "10.13026/C2RP4B",
        "license": "Open Data Commons Attribution License v1.0",
        "evaluation": "leave-one-record-out; candidate peaks matched to verified reference fetal QRS annotations within 80 ms",
        "records": evaluations,
        "mean_ml_assisted_f1": mean_metric(evaluations, "f1"),
        "mean_ml_assisted_precision": mean_metric(evaluations, "precision"),
        "mean_ml_assisted_recall": mean_metric(evaluations, "recall"),
        "mean_ml_assisted_bpm_absolute_error": mean_metric(evaluations, "bpm_absolute_error"),
        "limitation": "Two public records are currently included. These results are exploratory and must not be described as clinical performance or broad generalization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
