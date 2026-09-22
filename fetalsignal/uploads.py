"""Validate research uploads without changing sample alignment or guessing units."""

import io

import numpy as np
import pandas as pd

TIME_COLUMNS = {"time", "timestamp", "time_s", "time_seconds", "seconds", "t"}
MAX_ROWS = 600_000
MAX_BYTES = 25 * 1024 * 1024


def read_table(uploaded) -> pd.DataFrame:
    content = uploaded.getvalue()
    if len(content) > MAX_BYTES:
        raise ValueError("This file is too large. Use a CSV/TXT file under 25 MB and at most 600,000 samples.")
    if not content.strip():
        raise ValueError("This file is empty. Choose a CSV/TXT file with a header row and numeric samples.")
    try:
        # Single-column files need an explicit delimiter: automatic sniffing can
        # mistake characters in the header for separators.
        first_line = content.decode("utf-8-sig").splitlines()[0]
        delimiter = next((value for value in (",", "\t", ";") if value in first_line), None)
        frame = pd.read_csv(io.BytesIO(content), sep=delimiter or r"\s+", engine="python",
                            nrows=MAX_ROWS + 1, skip_blank_lines=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ValueError("Could not read this table. Use UTF-8 CSV/TXT with headers and comma, tab, semicolon, or space separators.") from error
    if frame.empty:
        raise ValueError("This table has no samples. Add numeric rows below the headers.")
    if len(frame) > MAX_ROWS:
        raise ValueError("Use at most 600,000 samples per recording. Split longer research recordings before uploading.")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Give each column a different name so each ECG lead can be selected unambiguously.")
    return frame


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if pd.to_numeric(frame[column], errors="coerce").notna().any()]


def extract_column(frame: pd.DataFrame, column: str, sample_rate: float) -> np.ndarray:
    signal = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError(f"Column '{column}' contains missing or non-numeric samples. Repair the source file; dropping rows would change beat timing and lead alignment.")
    if signal.size < int(sample_rate * 3):
        raise ValueError("Use at least 3 seconds of ECG samples at the selected sampling rate.")
    if np.max(np.abs(signal)) > 1e100:
        raise ValueError(f"Column '{column}' contains extreme values that cannot be processed reliably. Check the waveform units and source file.")
    if np.std(signal) < 1e-10:
        raise ValueError(f"Column '{column}' is flat. Select a changing abdominal ECG waveform.")
    return signal


def infer_time(frame: pd.DataFrame, expected_length: int, sample_rate: float) -> tuple[np.ndarray, str]:
    candidates = [column for column in frame if column.lower() in TIME_COLUMNS]
    if candidates:
        values = pd.to_numeric(frame[candidates[0]], errors="coerce").to_numpy(dtype=float)
        if values.size != expected_length or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
            raise ValueError("The time column must contain one increasing, finite time in seconds for every sample.")
        intervals = np.diff(values)
        interval = float(np.median(intervals))
        if not np.allclose(intervals, interval, rtol=0.02, atol=1e-6):
            raise ValueError("Samples are not evenly spaced. Use a uniformly sampled ECG recording without missing rows.")
        inferred_rate = 1 / interval
        if not np.isclose(inferred_rate, sample_rate, rtol=0.01):
            raise ValueError(f"The time column indicates approximately {inferred_rate:g} Hz. Set Sampling rate (Hz) to that value; time values must be in seconds.")
        return values - values[0], f"Time from {candidates[0]} (relative to recording start)"
    return np.arange(expected_length) / sample_rate, "Time calculated from the selected sampling rate"


def load_reference(uploaded, sample_rate: float, total_samples: int,
                   units: str = "Seconds", column: str | None = None) -> np.ndarray:
    frame = read_table(uploaded)
    candidates = numeric_columns(frame)
    if not candidates:
        raise ValueError("Reference beats need a numeric column of beat times or zero-based sample indices.")
    values = pd.to_numeric(frame[column or candidates[0]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Reference beats contain missing or invalid values. Use only finite numbers.")
    if units == "Seconds":
        samples = values * sample_rate
    elif units == "Sample indices (start at 0)":
        if not np.all(values == np.floor(values)):
            raise ValueError("Sample indices must be whole numbers starting at 0. Choose Seconds for beat times.")
        samples = values
    else:
        raise ValueError("Choose the reference units: seconds or sample indices.")
    if np.any(samples < 0) or np.any(samples >= total_samples):
        raise ValueError("Some reference beats lie outside this recording. Use the matching reference file, correct units, and times relative to the recording start.")
    reference = np.rint(samples).astype(int)
    if np.any(reference >= total_samples):
        raise ValueError("A reference time rounds beyond the last sample. Check the reference file and sampling rate.")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("Reference beats include duplicate sample locations. Keep one annotation per beat.")
    return np.sort(reference)
