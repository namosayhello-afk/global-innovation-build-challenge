import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2] / "Global Innovation Build Challenge V2"


class RecordingSourceTests(unittest.TestCase):
    def test_interactive_scenarios_change_the_generated_recording(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.radio(key="source_mode").set_value("Try the interactive demo").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("target fetal-candidate rhythm 132 BPM" in item.value for item in app.caption))

        scenario = next(item for item in app.selectbox if item.label == "Scenario")
        scenario.set_value("Motion challenge").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("target fetal-candidate rhythm 158 BPM" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
