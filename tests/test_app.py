import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def start(self):
        return AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)

    def assert_healthy(self, app):
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertFalse(app.error, [item.value for item in app.error])

    def test_guide_next_previous_and_replay(self):
        app = self.start()
        self.assert_healthy(app)
        for _ in range(3):
            next(b for b in app.button if b.label == "Next step →").click().run()
        self.assertEqual(app.session_state.guide_step, 3)
        next(b for b in app.button if b.label == "← Previous").click().run()
        self.assertEqual(app.session_state.guide_step, 2)
        next(b for b in app.button if b.label == "Next step →").click().run()
        next(b for b in app.button if b.label == "Replay guide").click().run()
        self.assertEqual(app.session_state.guide_step, 0)
        self.assert_healthy(app)

    def test_sample_has_references_and_upload_empty_state_has_no_stale_results(self):
        app = self.start()
        app.radio(key="source_mode").set_value("Use sample recording").run()
        self.assert_healthy(app)
        self.assertEqual(app.multiselect[0].value, ["abdominal_ecg"])
        self.assertTrue(any(m.label == "F1 score" for m in app.metric))
        app.radio(key="source_mode").set_value("Upload signal data").run()
        self.assert_healthy(app)
        self.assertEqual(len(app.metric), 0)

    def test_bad_waveform_does_not_leave_partial_results(self):
        app = self.start()
        file = io.BytesIO(b"time_seconds,ecg\n0,1\n0.002,2\n")
        file.name = "short.csv"
        with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: file if label.startswith("A.") else None):
            app.radio(key="source_mode").set_value("Upload signal data").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)

    def test_bad_reference_disables_validation_but_preserves_waveform(self):
        app = self.start()
        waveform = io.BytesIO((ROOT / "sample_data/fetalsignal_sample_recording.csv").read_bytes())
        waveform.name = "synthetic.csv"
        reference = io.BytesIO(b"beat\n9999\n")
        with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: waveform if label.startswith("A.") else reference):
            app.radio(key="source_mode").set_value("Upload signal data").run()
        self.assert_healthy(app)
        self.assertTrue(any(m.label == "Possible fetal rate" for m in app.metric))
        self.assertFalse(any(m.label == "F1 score" for m in app.metric))
        self.assertTrue(any("Reference file needs attention" in w.value for w in app.warning))

    def test_multilead_consensus_and_empty_selection(self):
        app = self.start()
        frame = pd.read_csv(ROOT / "sample_data/fetalsignal_sample_recording.csv")
        # Identical synthetic channels exercise consensus plumbing, not accuracy.
        frame["lead_2"] = frame.abdominal_ecg
        frame["lead_3"] = frame.abdominal_ecg
        waveform = io.BytesIO(frame.to_csv(index=False).encode())
        waveform.name = "synthetic_three_leads.csv"
        with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: waveform if label.startswith("A.") else None):
            app.radio(key="source_mode").set_value("Upload signal data").run()
            self.assert_healthy(app)
            self.assertTrue(any("3-lead consensus" in c.value for c in app.caption))
            app.multiselect[0].set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)

    def test_sample_rate_mismatch_removes_results(self):
        app = self.start()
        waveform = io.BytesIO((ROOT / "sample_data/fetalsignal_sample_recording.csv").read_bytes())
        waveform.name = "synthetic.csv"
        with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: waveform if label.startswith("A.") else None):
            app.radio(key="source_mode").set_value("Upload signal data").run()
            app.number_input[0].set_value(250).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("500 Hz" in e.value for e in app.error))
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
