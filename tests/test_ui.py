import numpy as np
import unittest

from fetalsignal.ui import quality_diagnostics


class PresentationDiagnosticsTests(unittest.TestCase):
    def test_quality_diagnostics_are_bounded_and_named(self):
        sample_rate = 500
        time = np.arange(0, 8, 1 / sample_rate)
        cleaned = np.sin(2 * np.pi * 3 * time)
        residual = 0.2 * np.sin(2 * np.pi * 7 * time)
        peaks = np.arange(200, len(time) - 1, 210)

        diagnostics = quality_diagnostics(cleaned, residual, peaks, sample_rate)

        self.assertEqual(
            set(diagnostics),
            {"Plausible spacing", "Rhythm regularity", "Residual energy"},
        )
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertTrue(all(0 <= value <= 100 for value in diagnostics.values()))

    def test_sparse_candidates_produce_zero_interval_diagnostics(self):
        diagnostics = quality_diagnostics(
            np.arange(1000, dtype=float),
            np.arange(1000, dtype=float),
            np.array([100, 300]),
            500,
        )
        self.assertEqual(diagnostics["Plausible spacing"], 0)
        self.assertEqual(diagnostics["Rhythm regularity"], 0)


if __name__ == "__main__":
    unittest.main()
