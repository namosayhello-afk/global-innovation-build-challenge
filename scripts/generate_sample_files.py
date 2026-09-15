"""Generate safe, synthetic files for testing the upload flow."""

from pathlib import Path

import pandas as pd

from fetalsignal.demo_data import make_demo_recording


def main() -> None:
    destination = Path("sample_data")
    destination.mkdir(exist_ok=True)
    recording = make_demo_recording(duration_seconds=30, sample_rate=500, noise_level=0.045)
    pd.DataFrame(
        {"time_seconds": recording["time"], "abdominal_ecg": recording["abdominal"]}
    ).to_csv(destination / "fetalsignal_sample_recording.csv", index=False)
    pd.DataFrame(
        {"fetal_beat_time_seconds": recording["fetal_reference_seconds"]}
    ).to_csv(destination / "fetalsignal_sample_reference_beats.csv", index=False)


if __name__ == "__main__":
    main()
