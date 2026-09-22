import numpy as np
import unittest

from fetalsignal.demo_data import make_demo_recording
from fetalsignal.signal_processing import extract_fetal_signal, fuse_multichannel_peaks, heart_rate_bpm, match_peaks, rolling_heart_rate


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


def test_rolling_rate_returns_expected_rate_for_regular_beats():
    peaks = np.arange(100, 5_100, 250)
    centers, rates = rolling_heart_rate(peaks, sample_rate=500, duration_seconds=12)
    assert centers.size
    assert np.allclose(rates[~np.isnan(rates)], 120)


def test_multichannel_fusion_requires_consensus():
    fused = fuse_multichannel_peaks(
        [np.array([100, 300, 500]), np.array([103, 301, 700]), np.array([98, 302, 500])],
        sample_rate=1_000,
        minimum_channels=3,
        tolerance_ms=8,
    )
    assert np.array_equal(fused, np.array([100, 301]))


def load_tests(loader, suite, pattern):
    """Run the existing function tests with the standard-library test runner too."""
    for name, check in sorted(globals().items()):
        if name.startswith("test_") and callable(check):
            suite.addTest(unittest.FunctionTestCase(check))
    return suite
