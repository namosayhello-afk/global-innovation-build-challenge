import numpy as np

from src.demo_data import make_demo_recording
from src.signal_processing import extract_fetal_signal, heart_rate_bpm, match_peaks


def test_demo_pipeline_returns_finite_signals():
    demo = make_demo_recording(duration_seconds=12, seed=5)
    result = extract_fetal_signal(demo["abdominal"], demo["sample_rate"])
    assert result.cleaned.shape == demo["abdominal"].shape
    assert result.residual.shape == demo["abdominal"].shape
    assert np.isfinite(result.cleaned).all()
    assert 0 <= result.quality_score <= 100


def test_peak_matcher_matches_identical_annotations():
    peaks = np.array([100, 350, 600])
    metrics = match_peaks(peaks, peaks, sample_rate=500)
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["f1"] == 1
    assert heart_rate_bpm(peaks, 500) == 120
