"""Transparent baseline processing for abdominal ECG research data.

This module is intentionally a research baseline, not clinical-grade fetal monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch


@dataclass
class ExtractionResult:
    cleaned: np.ndarray
    maternal_component: np.ndarray
    residual: np.ndarray
    maternal_peaks: np.ndarray
    fetal_peaks: np.ndarray
    quality_score: float


def _safe_filter(b: np.ndarray, a: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Use zero-phase filtering, falling back to demeaned input for short signals."""
    minimum = 3 * max(len(a), len(b))
    if signal.size <= minimum:
        return signal - np.mean(signal)
    return filtfilt(b, a, signal)


def bandpass(signal: np.ndarray, sample_rate: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyquist = sample_rate / 2
    high_hz = min(high_hz, nyquist * 0.94)
    if low_hz <= 0 or high_hz <= low_hz:
        return signal - np.mean(signal)
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
    return _safe_filter(b, a, signal)


def clean_abdominal_ecg(signal: np.ndarray, sample_rate: float, powerline_hz: float = 50.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.median(signal)
    if 0 < powerline_hz < sample_rate / 2 - 1:
        b, a = iirnotch(powerline_hz / (sample_rate / 2), Q=30)
        signal = _safe_filter(b, a, signal)
    return bandpass(signal, sample_rate, low_hz=0.7, high_hz=90.0, order=3)


def _detect_peaks(signal: np.ndarray, sample_rate: float, min_bpm: float, max_bpm: float, prominence_scale: float) -> np.ndarray:
    minimum_distance = max(1, int(sample_rate * 60 / max_bpm))
    scale = np.median(np.abs(signal - np.median(signal))) * 1.4826
    prominence = max(scale * prominence_scale, np.std(signal) * 0.13, 1e-7)
    positive, _ = find_peaks(signal, distance=minimum_distance, prominence=prominence)
    negative, _ = find_peaks(-signal, distance=minimum_distance, prominence=prominence)
    candidates = positive if positive.size >= negative.size else negative
    if candidates.size < 2:
        return candidates
    intervals = np.diff(candidates) / sample_rate
    valid = (intervals >= 60 / max_bpm) & (intervals <= 60 / min_bpm)
    keep = np.r_[True, valid]
    return candidates[keep]


def estimate_maternal_component(cleaned: np.ndarray, maternal_peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Estimate maternal ECG with a median beat template and subtractable overlap-add."""
    half_window = int(sample_rate * 0.24)
    width = 2 * half_window + 1
    usable = maternal_peaks[(maternal_peaks >= half_window) & (maternal_peaks < cleaned.size - half_window)]
    if usable.size < 3:
        return np.zeros_like(cleaned)
    beats = np.vstack([cleaned[p - half_window : p + half_window + 1] for p in usable])
    template = np.median(beats, axis=0)
    component = np.zeros_like(cleaned)
    weights = np.zeros_like(cleaned)
    for peak in usable:
        start, end = peak - half_window, peak + half_window + 1
        local = cleaned[start:end]
        amplitude = np.dot(local, template) / (np.dot(template, template) + 1e-12)
        component[start:end] += template * amplitude
        weights[start:end] += 1
    return np.divide(component, weights, out=np.zeros_like(component), where=weights > 0)


def signal_quality(cleaned: np.ndarray, residual: np.ndarray, fetal_peaks: np.ndarray, sample_rate: float) -> float:
    """Heuristic quality indicator, not a calibrated medical confidence value."""
    if fetal_peaks.size < 3:
        return 0.0
    intervals = np.diff(fetal_peaks) / sample_rate
    physiologic = np.mean((intervals >= 60 / 220) & (intervals <= 60 / 90))
    regularity = max(0.0, 1 - np.std(intervals) / (np.mean(intervals) + 1e-12))
    residual_ratio = np.std(residual) / (np.std(cleaned) + 1e-12)
    separation = min(1.0, residual_ratio / 0.30)
    return float(np.clip(100 * (0.48 * physiologic + 0.30 * regularity + 0.22 * separation), 0, 100))


def extract_fetal_signal(signal: np.ndarray, sample_rate: float, powerline_hz: float = 50.0) -> ExtractionResult:
    cleaned = clean_abdominal_ecg(signal, sample_rate, powerline_hz)
    maternal_view = bandpass(cleaned, sample_rate, low_hz=5.0, high_hz=35.0)
    maternal_peaks = _detect_peaks(maternal_view, sample_rate, min_bpm=40, max_bpm=130, prominence_scale=0.8)
    maternal_component = estimate_maternal_component(cleaned, maternal_peaks, sample_rate)
    residual = cleaned - maternal_component
    fetal_view = bandpass(residual, sample_rate, low_hz=12.0, high_hz=80.0)
    fetal_peaks = _detect_peaks(fetal_view, sample_rate, min_bpm=90, max_bpm=220, prominence_scale=0.45)
    quality = signal_quality(cleaned, fetal_view, fetal_peaks, sample_rate)
    return ExtractionResult(cleaned, maternal_component, fetal_view, maternal_peaks, fetal_peaks, quality)


def heart_rate_bpm(peaks: np.ndarray, sample_rate: float) -> float | None:
    if peaks.size < 2:
        return None
    intervals = np.diff(peaks) / sample_rate
    intervals = intervals[(intervals > 0.27) & (intervals < 0.8)]
    return None if intervals.size == 0 else float(60 / np.median(intervals))


def match_peaks(predicted: np.ndarray, reference: np.ndarray, sample_rate: float, tolerance_ms: float = 80.0) -> dict[str, float | int]:
    """One-to-one annotation matching for transparent evaluation."""
    tolerance = int(sample_rate * tolerance_ms / 1000)
    used = np.zeros(reference.size, dtype=bool)
    matches = 0
    errors: list[float] = []
    for peak in predicted:
        candidates = np.where((np.abs(reference - peak) <= tolerance) & ~used)[0]
        if candidates.size:
            nearest = candidates[np.argmin(np.abs(reference[candidates] - peak))]
            used[nearest] = True
            matches += 1
            errors.append(abs(reference[nearest] - peak) / sample_rate * 1000)
    precision = matches / predicted.size if predicted.size else 0.0
    recall = matches / reference.size if reference.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "peak_error_ms": float(np.mean(errors)) if errors else float("nan"),
    }
