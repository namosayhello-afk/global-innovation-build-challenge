import io
import unittest

import numpy as np
import pandas as pd

from fetalsignal.uploads import read_table, extract_column, infer_time, load_reference, TIME_COLUMNS
from fetalsignal.ml import select_model_candidates
from fetalsignal.signal_processing import heart_rate_bpm


class UploadTests(unittest.TestCase):
    def test_reference_single_column_parses_without_delimiter_guessing(self):
        frame = read_table(io.BytesIO(b"fetal_beat_time_seconds\n0.1\n0.8\n"))
        self.assertEqual(frame.shape, (2, 1))

    def test_empty_binary_and_header_only_files_fail_cleanly(self):
        for payload in (b"", b"   ", b"\xff\xfe", b"ecg\n"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_table(io.BytesIO(payload))

    def test_invalid_samples_are_not_dropped_and_shifted(self):
        for invalid in (np.nan, np.inf, "bad"):
            frame = pd.DataFrame({"lead": list(np.sin(np.arange(1600))) + [invalid]})
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "missing or non-numeric"):
                extract_column(frame, "lead", 500)

    def test_short_and_flat_waveforms_are_rejected(self):
        for data in (np.sin(np.arange(30)), np.ones(1600)):
            with self.assertRaises(ValueError):
                extract_column(pd.DataFrame({"lead": data}), "lead", 500)

    def test_time_seconds_is_not_a_signal_lead(self):
        self.assertIn("time_seconds", TIME_COLUMNS)
        time = np.arange(1600) / 500
        result, _ = infer_time(pd.DataFrame({"time_seconds": time}), len(time), 500)
        np.testing.assert_allclose(result, time)

    def test_time_rate_mismatch_and_gaps_are_rejected(self):
        time = np.arange(1600) / 500
        with self.assertRaisesRegex(ValueError, "500 Hz"):
            infer_time(pd.DataFrame({"time": time}), len(time), 250)
        time[800:] += .01
        with self.assertRaisesRegex(ValueError, "not evenly spaced"):
            infer_time(pd.DataFrame({"time": time}), len(time), 500)

    def test_units_are_explicit_and_small_indices_are_not_seconds(self):
        file = io.BytesIO(b"beats\n1\n2\n")
        np.testing.assert_array_equal(load_reference(file, 500, 2000, "Seconds"), [500, 1000])
        np.testing.assert_array_equal(load_reference(file, 500, 2000, "Sample indices (start at 0)"), [1, 2])

    def test_outside_duplicate_and_invalid_reference_values_are_rejected(self):
        for payload in (b"beat\n-1\n", b"beat\n8\n", b"beat\n0.5\n0.5\n", b"beat\n0.5\nbad\n"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                load_reference(io.BytesIO(payload), 500, 2000)

    def test_one_reference_beat_is_accepted(self):
        np.testing.assert_array_equal(load_reference(io.BytesIO(b"beat\n0.5\n"), 500, 2000), [250])

    def test_empty_candidates_do_not_crash_model(self):
        peaks, probabilities = select_model_candidates(np.zeros(1500), np.array([], dtype=int), 500)
        self.assertEqual(peaks.size, 0)
        self.assertIsNone(probabilities)

    def test_maternal_rate_can_be_below_fetal_interval_range(self):
        self.assertEqual(heart_rate_bpm(np.arange(0, 2500, 500), 500, 39, 131), 60)


if __name__ == "__main__":
    unittest.main()
