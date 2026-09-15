"""Synthetic, labelled abdominal-ECG-like data for safe dashboard demonstrations."""

from __future__ import annotations

import numpy as np


def _pulse_train(time: np.ndarray, beat_times: np.ndarray, amplitude: float, width: float) -> np.ndarray:
    """Build a simple QRS-like pulse train from Gaussian components."""
    signal = np.zeros_like(time)
    for beat in beat_times:
        qrs = amplitude * np.exp(-0.5 * ((time - beat) / width) ** 2)
        q = -0.16 * amplitude * np.exp(-0.5 * ((time - (beat - 0.025)) / (width * 0.55)) ** 2)
        s = -0.24 * amplitude * np.exp(-0.5 * ((time - (beat + 0.035)) / (width * 0.70)) ** 2)
        t_wave = 0.20 * amplitude * np.exp(-0.5 * ((time - (beat + 0.20)) / (width * 3.7)) ** 2)
        signal += qrs + q + s + t_wave
    return signal


def make_demo_recording(
    duration_seconds: float = 30.0,
    sample_rate: int = 500,
    maternal_bpm: float = 77.0,
    fetal_bpm: float = 143.0,
    noise_level: float = 0.075,
    seed: int = 17,
) -> dict[str, np.ndarray | float | int]:
    """Return a repeatable synthetic mixture with hidden reference beat locations.

    This is visual demonstration data, not patient data and not a physiological simulator.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration_seconds, 1 / sample_rate)
    maternal_interval = 60 / maternal_bpm
    fetal_interval = 60 / fetal_bpm
    maternal_beats = np.arange(0.7, duration_seconds - 0.4, maternal_interval)
    fetal_beats = np.arange(0.42, duration_seconds - 0.2, fetal_interval)
    fetal_beats = fetal_beats + rng.normal(0, 0.009, size=fetal_beats.size)

    maternal = _pulse_train(time, maternal_beats, amplitude=1.0, width=0.016)
    fetal = _pulse_train(time, fetal_beats, amplitude=0.16, width=0.009)
    baseline = 0.10 * np.sin(2 * np.pi * 0.23 * time) + 0.035 * np.sin(2 * np.pi * 0.06 * time)
    interference = 0.018 * np.sin(2 * np.pi * 50 * time)
    motion = np.zeros_like(time)
    for center in (8.3, 19.1, 25.4):
        motion += 0.11 * np.exp(-0.5 * ((time - center) / 0.14) ** 2) * rng.normal(0, 1, time.size)
    abdominal = maternal + fetal + baseline + interference + motion + rng.normal(0, noise_level, time.size)

    return {
        "time": time,
        "abdominal": abdominal,
        "maternal_reference_seconds": maternal_beats,
        "fetal_reference_seconds": fetal_beats,
        "sample_rate": sample_rate,
    }
