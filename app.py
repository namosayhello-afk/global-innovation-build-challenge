import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.demo_data import make_demo_recording
from src.signal_processing import extract_fetal_signal, heart_rate_bpm, match_peaks


st.set_page_config(page_title="FetalSignal AI", page_icon="♥", layout="wide")


def ecg_chart(time, signal, title, color, peaks=None):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time, y=signal, mode="lines", name=title, line=dict(color=color, width=1.4)))
    if peaks is not None and len(peaks):
        figure.add_trace(go.Scatter(x=time[peaks], y=signal[peaks], mode="markers", name="Detected peaks", marker=dict(color="#ffbf69", size=7)))
    figure.update_layout(
        template="plotly_dark", height=245, margin=dict(l=12, r=12, t=42, b=12),
        title=title, xaxis_title="Time (seconds)", yaxis_title="Amplitude (a.u.)", legend=dict(orientation="h", y=1.15),
    )
    return figure


def read_csv(uploaded, column, sample_rate):
    frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' was not found. Available columns: {', '.join(frame.columns)}")
    signal = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    if signal.size < int(sample_rate * 3):
        raise ValueError("Use a recording at least 3 seconds long.")
    return signal


st.markdown("""
<style>
    .stApp { background: #07131f; }
    [data-testid="stMetric"] { background: #102434; padding: 14px; border-radius: 12px; }
    .eyebrow { color: #77d9c4; font-weight: 700; letter-spacing: .11em; font-size: .75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="eyebrow">GLOBAL INNOVATION BUILD CHALLENGE V2 · MEDICAL TECHNOLOGY</p>', unsafe_allow_html=True)
st.title("FetalSignal AI")
st.subheader("Research prototype for separating fetal cardiac activity from abdominal ECG mixtures")
st.warning("Research and education only — not a medical device, not validated for clinical use, and not for diagnosis or patient monitoring.")

with st.sidebar:
    st.header("Recording")
    source = st.radio("Data source", ["Synthetic labelled demo", "Upload abdominal ECG CSV"])
    sample_rate = st.number_input("Sample rate (Hz)", min_value=100, max_value=2_000, value=500, step=50)
    powerline = st.selectbox("Power-line frequency", [50, 60])
    st.caption("CSV upload: one row per sample. Select the signal column below.")

reference_seconds = None
if source == "Synthetic labelled demo":
    demo = make_demo_recording(sample_rate=int(sample_rate))
    raw_signal = demo["abdominal"]
    time = demo["time"]
    reference_seconds = demo["fetal_reference_seconds"]
    st.info("Showing synthetic, labelled demonstration data. Its reference beats are included only to demonstrate evaluation—not a clinical result.")
else:
    uploaded = st.sidebar.file_uploader("Abdominal ECG CSV", type="csv")
    raw_signal = None
    time = None
    if uploaded is not None:
        preview = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        column = st.sidebar.selectbox("ECG column", list(preview.columns))
        try:
            raw_signal = read_csv(uploaded, column, sample_rate)
            time = np.arange(raw_signal.size) / sample_rate
        except ValueError as error:
            st.error(str(error))
    else:
        st.info("Upload a CSV to analyze your own de-identified research recording.")

if raw_signal is not None:
    result = extract_fetal_signal(raw_signal, sample_rate, powerline)
    fetal_bpm = heart_rate_bpm(result.fetal_peaks, sample_rate)
    maternal_bpm = heart_rate_bpm(result.maternal_peaks, sample_rate)
    visible = min(raw_signal.size, int(sample_rate * 12))
    display_time = time[:visible]

    st.markdown("### Extraction results")
    a, b, c, d = st.columns(4)
    a.metric("Estimated fetal rate", "—" if fetal_bpm is None else f"{fetal_bpm:.0f} BPM")
    b.metric("Detected fetal beats", int(result.fetal_peaks.size))
    c.metric("Maternal rate", "—" if maternal_bpm is None else f"{maternal_bpm:.0f} BPM")
    d.metric("Signal quality heuristic", f"{result.quality_score:.0f}/100")
    st.caption("The quality score is a transparent signal heuristic; it is not a calibrated probability or clinical confidence measure.")

    raw_peaks = result.maternal_peaks[result.maternal_peaks < visible]
    fetal_peaks = result.fetal_peaks[result.fetal_peaks < visible]
    left, right = st.columns(2)
    left.plotly_chart(ecg_chart(display_time, raw_signal[:visible], "1. Raw abdominal ECG", "#7dd3fc", raw_peaks), use_container_width=True)
    right.plotly_chart(ecg_chart(display_time, result.cleaned[:visible], "2. Filtered abdominal ECG", "#a7f3d0"), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(ecg_chart(display_time, result.maternal_component[:visible], "3. Maternal template estimate", "#f9a8d4"), use_container_width=True)
    right.plotly_chart(ecg_chart(display_time, result.residual[:visible], "4. Extracted fetal candidate signal", "#c4b5fd", fetal_peaks), use_container_width=True)

    with st.expander("Pipeline and limitations"):
        st.markdown("""
        1. Remove baseline drift and power-line interference.
        2. Detect maternal candidate R-peaks and form a median maternal beat template.
        3. Subtract the template estimate, band-pass the residual, and detect fetal candidate peaks.
        4. Report a quality heuristic based on peak regularity and residual energy.

        This is a baseline for reproducible experiments. Abdominal ECG varies substantially by electrode placement, gestational age, motion, and noise. It must be evaluated only against an appropriate public, de-identified reference dataset before making performance claims.
        """)

    if reference_seconds is not None:
        reference = np.rint(reference_seconds * sample_rate).astype(int)
        reference = reference[(reference >= 0) & (reference < raw_signal.size)]
        metrics = match_peaks(result.fetal_peaks, reference, sample_rate)
        reference_bpm = heart_rate_bpm(reference, sample_rate)
        predicted_bpm = heart_rate_bpm(result.fetal_peaks, sample_rate)
        st.markdown("### Demo-only validation")
        e, f, g, h = st.columns(4)
        e.metric("Precision", f"{metrics['precision']:.1%}")
        f.metric("Recall", f"{metrics['recall']:.1%}")
        g.metric("F1 score", f"{metrics['f1']:.1%}")
        h.metric("Rate error", "—" if not predicted_bpm else f"{abs(predicted_bpm - reference_bpm):.1f} BPM")
        st.caption("These values are generated from the synthetic demo recording and are not a claim of real-world performance.")

    export = pd.DataFrame({"time_seconds": time, "raw_abdominal_ecg": raw_signal, "filtered_ecg": result.cleaned, "maternal_estimate": result.maternal_component, "fetal_candidate_signal": result.residual})
    st.download_button("Download processed signal CSV", export.to_csv(index=False).encode("utf-8"), "fetalsignal_processed.csv", "text/csv")
