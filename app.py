"""FetalSignal AI Streamlit interface.

An explorable research dashboard for abdominal ECG signal separation.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fetalsignal.demo_data import make_demo_recording
from fetalsignal.ml import select_model_candidates
from fetalsignal.signal_processing import extract_fetal_signal, fuse_multichannel_peaks, heart_rate_bpm, match_peaks, rolling_heart_rate
from fetalsignal.uploads import TIME_COLUMNS, read_table, numeric_columns, extract_column, infer_time, load_reference


st.set_page_config(page_title="FetalSignal AI", page_icon="✦", layout="wide", initial_sidebar_state="auto")


def inject_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
            :root { --ink: #eaf3ff; --muted: #95a7b9; --line: rgba(174, 202, 226, .15); --mint: #76f0cf; --lilac: #a99cff; }
            .stApp { background: radial-gradient(circle at 78% 2%, #1b3650 0, #0b1724 36%, #08111c 82%); color: var(--ink); font-family: 'Manrope', sans-serif; }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] { background: #0b1724; border-right: 1px solid var(--line); }
            [data-testid="stSidebar"] * { color: #dbe9f5; }
            .block-container { max-width: 1370px; padding-top: 2.2rem; padding-bottom: 3rem; }
            h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.045em; }
            h1 { font-size: clamp(2rem, 4vw, 3.4rem) !important; font-weight: 800 !important; line-height: 1.15 !important; margin-bottom: .5rem !important; }
            h2 { font-weight: 750 !important; }
            .brand-row { display: flex; align-items: center; gap: 9px; color: var(--mint); font: 500 .73rem 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
            .spark { display: inline-flex; width: 29px; height: 29px; border-radius: 9px; align-items: center; justify-content: center; background: linear-gradient(135deg, #77efd0, #9c95ff); color: #09131f; font-weight: 900; font-size: 1rem; }
            .hero-copy { max-width: 715px; color: #b9c9d8; font-size: 1.12rem; line-height: 1.7; margin: 0 0 1.4rem; }
            .hero-copy strong { color: #f4fbff; }
            .safe-note { background: rgba(118, 240, 207, .08); border: 1px solid rgba(118, 240, 207, .22); border-radius: 14px; padding: .82rem 1rem; color: #b9e5d9; font-size: .86rem; line-height: 1.5; }
            .hero-orb { width: min(290px, 100%); aspect-ratio: 1; margin: 1.2rem auto; border-radius: 50%; display: grid; place-items: center; background: radial-gradient(circle at 35% 30%, #62ddca 0, #2c6574 25%, #182c52 55%, transparent 71%); box-shadow: 0 0 90px rgba(96, 209, 214, .22); position: relative; overflow: hidden; }
            .hero-orb:before { content: ''; position:absolute; inset: 15%; border: 1px solid rgba(234, 243, 255, .34); border-radius: 50%; }
            .orb-line { color: #f2ffff; font: 500 .76rem 'DM Mono', monospace; letter-spacing: .12em; text-align:center; z-index: 1; line-height: 1.8; }
            .step-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin: 1.8rem 0 2rem; }
            .step-card { background: rgba(255, 255, 255, .045); border: 1px solid var(--line); border-radius: 14px; padding: 1rem; min-height: 105px; }
            .step-number { color: var(--mint); font: 500 .72rem 'DM Mono', monospace; letter-spacing: .08em; }
            .step-card b { display:block; margin: .42rem 0 .22rem; font-size: .95rem; }
            .step-card span { color: var(--muted); font-size: .79rem; line-height: 1.4; }
            .section-label { color: var(--mint); font: 500 .74rem 'DM Mono', monospace; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .3rem; }
            [data-testid="stMetric"] { background: rgba(255, 255, 255, .055); border: 1px solid var(--line); padding: 1rem 1.08rem; border-radius: 15px; }
            [data-testid="stMetricLabel"] { color: #9fb2c5; font-size: .77rem; }
            [data-testid="stMetricValue"] { color: #f6fbff; font-size: 1.55rem; font-weight: 750; }
            .status-pill { display: inline-block; max-width: 100%; box-sizing: border-box; border-radius: 99px; padding: .42rem .7rem; font: 500 .7rem 'DM Mono', monospace; letter-spacing: .04em; line-height: 1.35; text-align: center; white-space: normal; }
            .status-good { background: rgba(118, 240, 207, .13); color: #76f0cf; border: 1px solid rgba(118, 240, 207, .3); }
            .status-watch { background: rgba(255, 207, 112, .12); color: #ffd98a; border: 1px solid rgba(255, 207, 112, .28); }
            .status-low { background: rgba(255, 139, 159, .12); color: #ff9caf; border: 1px solid rgba(255, 139, 159, .25); }
            .panel { background: rgba(255,255,255,.032); border: 1px solid var(--line); border-radius: 17px; padding: 1.05rem 1.2rem; }
            .caption { color: var(--muted); font-size: .84rem; line-height: 1.55; }
            .stTabs [data-baseweb="tab-list"] { overflow-x: auto; border-bottom: 1px solid var(--line); }
            .stTabs [aria-selected="true"] { color: #76f0cf !important; }
            .stButton > button, .stDownloadButton > button { min-height: 2.75rem; white-space: normal; line-height: 1.25; border-radius: 10px; font-weight: 700; border: 1px solid rgba(118, 240, 207, .42); background: linear-gradient(135deg, #75e8cb, #a29aff); color: #07111d; }
            .stButton > button:hover, .stDownloadButton > button:hover { border-color: #eaffff; color: #07111d; }
            .stAlert { border-radius: 13px; }
            [data-testid="stMetric"], .panel, .step-card { overflow-wrap: anywhere; }
            [data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; }
            .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid #eaffff; outline-offset: 3px; }
            @media (max-width: 760px) { .step-row { grid-template-columns: 1fr; } h1 { font-size: 2rem !important; } .block-container { padding: 1.5rem 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ecg_chart(time: np.ndarray, signal: np.ndarray, title: str, color: str, peaks: np.ndarray | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time, y=signal, mode="lines", name="Signal", line=dict(color=color, width=1.45), hovertemplate="%{x:.3f} s<br>%{y:.3f} a.u.<extra></extra>"))
    if peaks is not None and peaks.size:
        figure.add_trace(go.Scatter(x=time[peaks], y=signal[peaks], mode="markers", name="Detected beat", marker=dict(color="#ffd68a", size=7, line=dict(color="#07111d", width=1)), hovertemplate="Beat at %{x:.3f} s<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=295, margin=dict(l=6, r=6, t=12, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", xaxis=dict(title="Time (seconds)", gridcolor="rgba(189,210,226,.09)", zeroline=False), yaxis=dict(title="Amplitude", gridcolor="rgba(189,210,226,.09)", zeroline=False), legend=dict(orientation="h", y=-.3, x=0, font=dict(size=10)), hovermode="x unified")
    return figure


def rhythm_chart(centers: np.ndarray, rates: np.ndarray) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=centers, y=rates, mode="lines+markers", line=dict(color="#76f0cf", width=2), marker=dict(size=6, color="#b7abff"), connectgaps=False, hovertemplate="%{x:.1f} s<br>%{y:.0f} BPM<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=220, margin=dict(l=6, r=6, t=15, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", xaxis=dict(title="Window center (seconds)", gridcolor="rgba(189,210,226,.09)"), yaxis=dict(title="Candidate BPM", gridcolor="rgba(189,210,226,.09)"), showlegend=False)
    return figure


def quality_label(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Clear candidate signal", "status-good"
    if score >= 45:
        return "Review signal quality", "status-watch"
    return "Low-quality candidate", "status-low"


def metric_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} BPM"


def make_export(time: np.ndarray, raw: np.ndarray, result) -> bytes:
    return pd.DataFrame({"time_seconds": time, "raw_abdominal_ecg": raw, "filtered_ecg": result.cleaned, "maternal_template_estimate": result.maternal_component, "fetal_candidate_signal": result.residual}).to_csv(index=False).encode("utf-8")


def make_annotation_export(time: np.ndarray, peaks: np.ndarray) -> bytes:
    fetal_times = time[peaks]
    return pd.DataFrame({"candidate_beat_sample": peaks, "candidate_beat_time_seconds": fetal_times}).to_csv(index=False).encode("utf-8")


def make_report(data_note: str, sample_rate: float, fetal_bpm: float | None, maternal_bpm: float | None, result, fetal_peaks: np.ndarray, analysis_method: str) -> bytes:
    report = f"""FetalSignal AI — exploratory analysis report

Input: {data_note}
Sampling rate: {sample_rate} Hz
Candidate fetal rate: {metric_value(fetal_bpm)}
Candidate maternal rate: {metric_value(maternal_bpm)}
Detected fetal candidate beats: {fetal_peaks.size}
Detection method: {analysis_method}
Primary lead signal-quality heuristic: {result.quality_score:.0f} / 100

Interpretation: This is a signal-processing output for de-identified research data. It is not a diagnosis, a clinical confidence value, or a medical-device reading. Candidate beats should be validated against an appropriate reference annotation set before reporting performance.
"""
    return report.encode("utf-8")


def load_evaluation_report() -> dict | None:
    """Read the bundled, reproducible public-data evaluation if it is present."""
    report_path = Path(__file__).parent / "results" / "adfecgdb_leave_one_record_out.json"
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def explain_result(fetal_bpm: float | None, quality_score: float, candidate_count: int, model_enabled: bool, consensus: bool = False) -> tuple[str, str, str]:
    """Translate research output into plain language without clinical claims."""
    model_note = " An experimental ML candidate-ranker reviewed the detected peaks." if model_enabled else " The baseline signal-processing detector selected the peaks."
    if consensus:
        model_note = " These peaks were supported by multiple aligned ECG leads."
    if fetal_bpm is None or candidate_count < 3:
        return (
            "No stable candidate rhythm was found",
            "The app could not find enough repeating peaks in the residual signal to estimate a candidate rate. This can happen when the recording is short, noisy, or the maternal pattern is not cleanly separable.",
            "Try a longer, cleaner research waveform; confirm the sampling rate; then inspect the raw and residual charts before drawing any conclusion.",
        )
    if quality_score >= 75:
        return (
            "A regular candidate pattern was found",
            f"The app found {candidate_count} repeating candidate peaks after reducing the maternal-pattern estimate. Their median interval corresponds to an estimated candidate rate of {fetal_bpm:.0f} BPM.{model_note}",
            "Inspect the markers against the residual waveform. For research reporting, upload separate reference annotations and use the Validation tab rather than treating this as a medical reading.",
        )
    return (
        "A candidate pattern was found, but it needs review",
        f"The app found {candidate_count} candidate peaks and estimates {fetal_bpm:.0f} BPM, but the signal-quality heuristic is only {quality_score:.0f}/100. Noise or imperfect maternal-pattern removal may be influencing the result.{model_note}",
        "Use the Signal lab to inspect the markers, then validate against separate reference annotations. Do not interpret this result as a patient assessment.",
    )


def choose_source(source: str) -> None:
    st.session_state.source_mode = source


def move_guide(direction: int) -> None:
    st.session_state.guide_step = max(0, min(3, st.session_state.get("guide_step", 0) + direction))


def show_guide() -> None:
    """A replayable walkthrough with concrete actions, separate from analysis state."""
    with st.expander("New here? Follow the guided walkthrough", expanded=True):
        step = st.session_state.get("guide_step", 0)
        titles = ["Choose a recording", "Understand the result", "Check the possible beats", "Save and explore"]
        instructions = [
            "Choose **Use sample recording** below for a complete example. It loads a synthetic waveform and matching reference beats automatically. Or choose **Upload signal data** and follow the numbered instructions. CSV/TXT waveforms are supported; photos, audio, and BPM screenshots are not.",
            "Scroll to **Your results**. Read **What this result means** first. BPM means beats per minute. A candidate is a possible beat the algorithm found; the quality score is a rough signal check, not the chance that a result is medically correct.",
            "Open **Overview** and move the time slider. Gold dots mark possible beats. Open **Validation** to compare them with reference beats. The sample already includes references; your own recording needs its matching annotation file.",
            "Open **Signal lab** to see the processing layers and download a report or CSV. **Evidence** shows a separate public-data experiment. To try another recording, change the choice below; you can replay this guide at any time.",
        ]
        st.caption(f"WALKTHROUGH · STEP {step + 1} OF 4")
        st.markdown(f"#### {titles[step]}")
        st.markdown(instructions[step])
        st.progress((step + 1) / 4)
        back, forward = st.columns(2)
        back.button("← Previous", disabled=step == 0, on_click=move_guide, args=(-1,), width="stretch")
        if step < 3:
            forward.button("Next step →", on_click=move_guide, args=(1,), width="stretch")
        else:
            forward.button("Replay guide", on_click=move_guide, args=(-3,), width="stretch")


def sample_downloads() -> None:
    root = Path(__file__).parent / "sample_data"
    with st.expander("Need a file to practice uploading?"):
        st.caption("Download these synthetic files. Use 500 Hz, the abdominal_ecg column, and reference units Seconds.")
        for name, label in [("fetalsignal_sample_recording.csv", "Download example waveform"),
                            ("fetalsignal_sample_reference_beats.csv", "Download matching reference beats")]:
            path = root / name
            if path.exists():
                st.download_button(label, path.read_bytes(), name, "text/csv")
            else:
                st.info("Example downloads are unavailable. Use Try the interactive demo instead.")


inject_style()

st.markdown("<div class='brand-row'><span class='spark'>✦</span> FETALSIGNAL AI · RESEARCH WORKSPACE</div>", unsafe_allow_html=True)
st.title("Explore the heartbeat in the signal.")
st.markdown("Upload an abdominal ECG waveform, explore possible fetal beats, and check them against reference data. **First visit? Start with the sample below.**")
st.markdown("<div class='safe-note'>Research and education only. This prototype is not a medical device or diagnostic tool and cannot be used for patient decisions.</div>", unsafe_allow_html=True)
show_guide()
st.markdown("### 1. Choose your recording")
source = st.radio("Recording source", ["Try the interactive demo", "Use sample recording", "Upload signal data"],
                  key="source_mode", horizontal=True,
                  help="Demo: adjustable synthetic signal. Sample: a ready-made file with references. Upload: your permitted research waveform.")

with st.sidebar:
    st.markdown("<div class='brand-row'><span class='spark'>✦</span> FetalSignal AI</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Recording settings")
    st.caption("Choose a source in the main workspace. Results update when you change a setting.")
    if source == "Try the interactive demo":
        st.markdown("#### Demo controls")
        demo_length = st.select_slider("Recording length", options=[15, 30, 45, 60], value=30, format_func=lambda seconds: f"{seconds} seconds")
        demo_noise = st.select_slider("Noise level", options=["Low", "Standard", "High"], value="Standard")
        sample_rate = st.selectbox("Sample rate", [250, 500, 1_000], index=1, format_func=lambda value: f"{value} Hz")
        powerline, reference_upload = 50, None
    elif source == "Use sample recording":
        sample_rate, powerline, reference_upload = 500, 50, None
        st.info("Example settings: 500 Hz, one ECG lead, matching reference beats. Everything is set for you.")
    else:
        sample_rate, powerline, reference_upload = 500, 50, None
        st.info("Configure your file in the main workspace, beside the upload instructions.")
    st.divider()
    st.markdown("#### Quick glossary")
    st.markdown("**ECG:** electrical heartbeat recording.\n\n**Lead:** one recording channel.\n\n**BPM:** beats per minute.\n\n**Candidate:** a possible beat.\n\n**Residual:** the signal left after reducing the maternal pattern.")
    st.caption("No API key or login is needed to analyze a permitted research waveform.")

uploaded = None
reference_units = "Seconds"
reference_column = None
if source == "Upload signal data":
    st.markdown("<div class='section-label'>Start an analysis</div>", unsafe_allow_html=True)
    st.markdown("#### Upload a research waveform")
    st.caption("Use only public or fully de-identified recordings you have permission to use. Files are processed in this server session and are not saved by the app.")
    sample_downloads()
    uploaded = st.file_uploader("A. Choose your waveform file", type=["csv", "txt"], help="Header row, one row per sample; comma, tab, semicolon, or space separated. Maximum 25 MB / 600,000 samples.")
    setting_one, setting_two = st.columns(2)
    sample_rate = setting_one.number_input("B. Sampling rate (Hz)", min_value=100, max_value=2_000, value=500, step=1, help="Samples recorded per second. Use the dataset's stated rate; the supplied sample uses 500 Hz.")
    powerline = setting_two.selectbox("Electrical interference", [50, 60], format_func=lambda value: f"{value} Hz", help="Use the power-line frequency documented for the recording.")
    with st.expander("C. Add matching reference beats (optional)"):
        st.caption("References are known beat locations for this exact recording. They let you test accuracy; analysis also works without them.")
        reference_upload = st.file_uploader("Reference-beat file", type=["csv", "txt"])
        reference_units = st.selectbox("Reference units", ["Seconds", "Sample indices (start at 0)"], help="Seconds must be measured from recording start. Choose the units documented for your annotations.")
        if reference_upload is not None:
            try:
                reference_choices = numeric_columns(read_table(reference_upload))
                if not reference_choices:
                    raise ValueError("No numeric reference-beat column was found.")
                reference_column = st.selectbox("Reference-beat column", reference_choices)
            except ValueError as error:
                st.warning(str(error))
elif source == "Use sample recording":
    sample_root = Path(__file__).parent / "sample_data"
    try:
        uploaded = io.BytesIO((sample_root / "fetalsignal_sample_recording.csv").read_bytes())
        uploaded.name = "Synthetic sample recording"
        reference_upload = io.BytesIO((sample_root / "fetalsignal_sample_reference_beats.csv").read_bytes())
        st.info("Sample loaded: a synthetic ECG with matching reference beats at 500 Hz. Scroll down to read your results, then open Validation.")
    except OSError:
        uploaded = None
        st.warning("The sample files are unavailable. Choose Try the interactive demo to continue.")
else:
    st.info("You are exploring a synthetic recording. Adjust length or noise in Recording settings, then compare the charts below. These demo results are illustrative.")
    st.button("Upload an ECG recording", on_click=choose_source, args=("Upload signal data",))

raw_signal: np.ndarray | None = None
time: np.ndarray | None = None
reference: np.ndarray | None = None
data_note = ""
precomputed_result = None
precomputed_fetal_peaks: np.ndarray | None = None
analysis_method = ""
reference_error = None

if source == "Try the interactive demo":
    noise_by_label = {"Low": 0.025, "Standard": 0.045, "High": 0.115}
    demo = make_demo_recording(duration_seconds=demo_length, sample_rate=int(sample_rate), noise_level=noise_by_label[demo_noise])
    raw_signal, time = demo["abdominal"], demo["time"]
    reference = np.rint(demo["fetal_reference_seconds"] * sample_rate).astype(int)
    data_note = "Synthetic labelled demo · no patient data"
else:
    if uploaded is None:
        st.info("Your next step: choose a waveform above, or select Use sample recording for a ready-to-run example. Results will appear here after a valid file is loaded.")
    else:
        try:
            uploaded_frame = read_table(uploaded)
            choices = [column for column in numeric_columns(uploaded_frame) if column.lower() not in TIME_COLUMNS]
            if not choices:
                raise ValueError("No numeric signal columns found. Add a header row and one column of ECG samples.")
            selected_columns = st.multiselect("D. Select abdominal ECG lead(s)", choices, default=choices[: min(4, len(choices))], max_selections=8, help="A lead is one ECG recording channel. Select abdominal leads only; exclude direct fetal, reference, and annotation columns. Three or more aligned leads enable consensus detection.")
            if not selected_columns:
                raise ValueError("Choose at least one abdominal ECG column to analyze.")
            signals = [extract_column(uploaded_frame, column, sample_rate) for column in selected_columns]
            time, time_note = infer_time(uploaded_frame, signals[0].size, sample_rate)
            with st.spinner("Cleaning the recording and finding possible beats…"):
                results = [extract_fetal_signal(signal, sample_rate, powerline) for signal in signals]
            primary_index = int(np.argmax([result.quality_score for result in results]))
            raw_signal, precomputed_result = signals[primary_index], results[primary_index]
            if len(results) >= 3:
                required_leads = 3
                consensus = fuse_multichannel_peaks([result.fetal_peaks for result in results], sample_rate, minimum_channels=required_leads)
                if consensus.size >= 3:
                    precomputed_fetal_peaks = consensus
                    analysis_method = f"{len(results)}-lead consensus (a beat needs support from {required_leads} leads)"
                else:
                    analysis_method = f"Best single lead selected from {len(results)} uploaded leads"
            data_note = f"{uploaded.name} · Displayed lead: {selected_columns[primary_index]} · {len(results)} lead(s) selected · {time_note}"
            if reference_upload is not None:
                try:
                    reference = load_reference(reference_upload, sample_rate, raw_signal.size, reference_units, reference_column)
                except ValueError as error:
                    reference_error = str(error)
                    st.warning(f"Reference file needs attention: {reference_error} Waveform analysis is still available; validation is not.")
        except ValueError as error:
            raw_signal, time, precomputed_result, precomputed_fetal_peaks = None, None, None, None
            st.error(str(error))
            st.caption("Fix the highlighted file or setting above. No analysis is shown for an invalid waveform.")

if raw_signal is not None and time is not None:
    result = precomputed_result or extract_fetal_signal(raw_signal, sample_rate, powerline)
    if precomputed_fetal_peaks is not None:
        fetal_peaks, model_probabilities = precomputed_fetal_peaks, None
    else:
        fetal_peaks, model_probabilities = select_model_candidates(result.residual, result.fetal_peaks, sample_rate)
        analysis_method = "Single-lead ML-assisted candidate ranking" if model_probabilities is not None else "Single-lead signal-processing baseline"
    fetal_bpm = heart_rate_bpm(fetal_peaks, sample_rate)
    maternal_bpm = heart_rate_bpm(result.maternal_peaks, sample_rate, min_bpm=39, max_bpm=131)
    label, label_class = quality_label(result.quality_score)
    st.markdown("<div class='section-label'>Analysis workspace</div>", unsafe_allow_html=True)
    title_col, status_col = st.columns([3.8, 2.2], vertical_alignment="center")
    title_col.markdown("## 2. Your results")
    status_col.markdown(f"<span class='status-pill {label_class}'>{label}</span>", unsafe_allow_html=True)
    st.caption(data_note)
    st.caption(f"{raw_signal.size / sample_rate:.1f} seconds · {sample_rate:g} samples/second · {analysis_method}")
    if source in {"Try the interactive demo", "Use sample recording"}:
        st.caption("SYNTHETIC EXAMPLE · no patient data · validation here demonstrates the workflow, not real-world performance")
    if precomputed_fetal_peaks is not None:
        st.caption("Multi-lead consensus is enabled: a candidate beat is kept only when it appears in several aligned abdominal ECG leads. This is an experimental research feature.")
    elif model_probabilities is not None:
        st.caption("ML-assisted candidate ranking is enabled. It is an experimental research model and falls back to the signal-processing baseline when its filtering would be too aggressive.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Possible fetal rate", metric_value(fetal_bpm), help="BPM means beats per minute. Estimated from the median spacing of candidate beats; it is not a confirmed fetal heart rate.")
    metric_columns[1].metric("Possible beats found", f"{fetal_peaks.size}", help="Algorithmic candidates in the full recording. Compare them with reference annotations in Validation.")
    metric_columns[2].metric("Maternal candidate rate", metric_value(maternal_bpm), help="Estimated from the dominant-pattern candidates; also an experimental estimate.")
    metric_columns[3].metric("Primary lead quality", f"{result.quality_score:.0f} / 100", help="A rough heuristic for the displayed lead before ML/consensus selection. It is not accuracy, confidence, or a medical score.")
    explanation_title, explanation_text, next_step = explain_result(fetal_bpm, result.quality_score, fetal_peaks.size, model_probabilities is not None, precomputed_fetal_peaks is not None)
    st.markdown("### What this result means")
    explanation_col, next_col = st.columns(2)
    with explanation_col:
        st.markdown(f"<div class='panel'><b>{explanation_title}</b><p class='caption'>{explanation_text}</p></div>", unsafe_allow_html=True)
    with next_col:
        st.markdown(f"<div class='panel'><b>What to do next</b><p class='caption'>{next_step}</p></div>", unsafe_allow_html=True)

    st.markdown("### 3. Explore, check, and save")
    st.caption("Overview: inspect the beats · Signal lab: see layers and download · Validation: compare references · Method: learn the steps · Evidence: separate public-data results")
    overview_tab, signals_tab, validation_tab, method_tab, evidence_tab = st.tabs(["Overview", "Signal lab", "Validation", "Method", "Evidence"])
    with overview_tab:
        preview_seconds = min(15, int(np.ceil(time[-1] - time[0])))
        range_limit = float(time[-1] - preview_seconds)
        if range_limit > 0.5:
            start_second = st.slider("Inspect this section of the recording", min_value=0.0, max_value=range_limit, value=0.0, step=0.5, format="%.1f s")
        else:
            start_second = 0.0
            st.caption("Showing the full short recording.")
        start_index = int(np.searchsorted(time, start_second, side="left"))
        end_index = min(raw_signal.size, int(np.searchsorted(time, start_second + preview_seconds, side="right")))
        local_time = time[start_index:end_index]
        raw_peaks = result.maternal_peaks[(result.maternal_peaks >= start_index) & (result.maternal_peaks < end_index)] - start_index
        displayed_fetal_peaks = fetal_peaks[(fetal_peaks >= start_index) & (fetal_peaks < end_index)] - start_index
        left, right = st.columns(2)
        left.markdown("#### Original abdominal recording")
        left.caption("The mixed signal. Gold dots mark maternal candidates.")
        left.plotly_chart(ecg_chart(local_time, raw_signal[start_index:end_index], "Abdominal ECG mixture", "#75d9ff", raw_peaks), width="stretch")
        right.markdown("#### Possible fetal signal")
        right.caption("After reducing the maternal pattern. Gold dots mark fetal candidates.")
        right.plotly_chart(ecg_chart(local_time, result.residual[start_index:end_index], "Fetal cardiac-signal candidate", "#aa9dff", displayed_fetal_peaks), width="stretch")
        st.markdown("<p class='caption'>Markers indicate algorithmic candidate peaks. They are not confirmed fetal beats unless compared with an appropriate reference annotation set.</p>", unsafe_allow_html=True)
        centers, rates = rolling_heart_rate(fetal_peaks, sample_rate, float(time[-1] - time[0]))
        if centers.size:
            st.markdown("### Candidate rhythm over time")
            st.plotly_chart(rhythm_chart(centers, rates), width="stretch")
            st.caption("Each point summarizes candidate-peak intervals in a sliding 10-second window. Gaps mean the detector did not find enough usable intervals—not a medical event.")
    with signals_tab:
        st.markdown("### Signal-separation layers")
        st.caption("These charts follow the same time window selected in Overview. Downloads contain the full recording.")
        left, right = st.columns(2)
        left.markdown("#### 1 · Cleaned recording")
        left.caption("Filtering reduces baseline drift and electrical interference.")
        left.plotly_chart(ecg_chart(local_time, result.cleaned[start_index:end_index], "Filtered abdominal ECG", "#74e7c9"), width="stretch")
        right.markdown("#### 2 · Maternal-pattern estimate")
        right.caption("The repeating stronger pattern the algorithm tries to reduce.")
        right.plotly_chart(ecg_chart(local_time, result.maternal_component[start_index:end_index], "Estimated maternal pattern", "#ff9ec3"), width="stretch")
        st.markdown("#### 3 · Remaining signal (residual)")
        st.caption("Possible fetal beats can remain here, along with noise or maternal interference.")
        st.plotly_chart(ecg_chart(local_time, result.residual[start_index:end_index], "Residual fetal candidate signal", "#b1a8ff", displayed_fetal_peaks), width="stretch")
        st.markdown("#### Save your analysis")
        st.caption("Choose the text report for a readable summary, or a CSV to examine numeric samples and beat locations.")
        download_one, download_two, download_three = st.columns(3)
        download_one.download_button("Processed waveform CSV", make_export(time, raw_signal, result), "fetalsignal_processed.csv", "text/csv", width="stretch")
        download_two.download_button("Candidate-beat CSV", make_annotation_export(time, fetal_peaks), "fetalsignal_candidate_beats.csv", "text/csv", width="stretch")
        download_three.download_button("Analysis report", make_report(data_note, sample_rate, fetal_bpm, maternal_bpm, result, fetal_peaks, analysis_method), "fetalsignal_report.txt", "text/plain", width="stretch")
    with validation_tab:
        st.markdown("### Check against known beat locations")
        st.caption("These metrics apply only to the current recording and its supplied references. A predicted beat matches a reference within 80 milliseconds.")
        if reference is not None and reference.size:
            metrics = match_peaks(fetal_peaks, reference, sample_rate)
            reference_bpm = heart_rate_bpm(reference, sample_rate)
            rate_error = abs(fetal_bpm - reference_bpm) if fetal_bpm is not None and reference_bpm is not None else None
            one, two, three, four = st.columns(4)
            one.metric("Precision", f"{metrics['precision']:.1%}", help="Of the predicted beats, how many match a known reference beat?")
            two.metric("Recall", f"{metrics['recall']:.1%}", help="Of the known reference beats, how many did the algorithm find?")
            three.metric("F1 score", f"{metrics['f1']:.1%}", help="A combined measure balancing precision and recall. Higher means better agreement on this recording.")
            four.metric("Rate error", "—" if rate_error is None else f"{rate_error:.1f} BPM", help="Absolute difference between candidate and reference median-interval BPM. This is not a beat-by-beat error measurement.")
            st.markdown("**Reading these numbers:** precision checks false detections; recall checks missed beats; F1 balances both. Rate error compares the two estimated rates. A small rate error alone does not prove every beat is correct.")
            st.caption("Metrics compare algorithmic candidate peaks with the supplied reference times. Demo results are illustrative only; upload a documented public dataset to make research claims.")
        else:
            if reference_error:
                st.warning(f"Validation unavailable: {reference_error}")
            else:
                st.info("No reference beats attached. Above the results, open C. Add matching reference beats, upload the annotations for this recording, and choose their units and column. Or use the sample to see a complete example.")
            st.markdown("<div class='panel'><b>What good validation looks like</b><p class='caption'>Keep reference annotations separate from the input waveform. Define your matching tolerance before testing, document the dataset and preprocessing settings, and report results on recordings not used to tune the pipeline.</p></div>", unsafe_allow_html=True)
    with method_tab:
        st.markdown("### A transparent baseline, step by step")
        st.markdown("""- **Clean** — removes slow baseline drift, the selected power-line frequency, and out-of-band content.
- **Find maternal candidates** — looks for recurring maternal QRS-like peaks in a maternal-frequency view.
- **Estimate and reduce** — builds a median maternal beat template and subtracts the overlap-adjusted estimate.
- **Find fetal candidates** — filters the residual and searches for repeating peaks in a fetal-rate range.
- **Fuse aligned leads when available** — when three or more abdominal ECG columns are selected, keeps a candidate only when several leads agree within 80 ms.
- **Score cautiously** — signal quality combines candidate-interval plausibility, regularity, and residual energy. It is not a calibrated confidence or medical risk score.""")
        st.warning("Limitations: abdominal ECG quality can change with electrode placement, motion, maternal rhythm, gestational age, and noise. A candidate signal can be wrong. This project is for de-identified research data only.")
    with evidence_tab:
        st.markdown("### Public-data evaluation evidence")
        evaluation_report = load_evaluation_report()
        if evaluation_report is None:
            st.info("The bundled evaluation report is unavailable in this deployment. See the repository results folder for the reproducible evaluation command and record-level outputs.")
        else:
            st.markdown("<div class='panel'><b>What these numbers do—and do not—mean</b><p class='caption'>These are exploratory results on five public, de-identified PhysioNet ADFECGDB recordings. They measure the multi-lead consensus pipeline against verified fetal-QRS references using an 80 ms match window. They do not predict performance for an uploaded recording, and they are not clinical-performance claims.</p></div>", unsafe_allow_html=True)
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Macro F1", f"{evaluation_report['mean_multilead_consensus_f1']:.2%}")
            metric_two.metric("Precision", f"{evaluation_report['mean_multilead_consensus_precision']:.2%}")
            metric_three.metric("Recall", f"{evaluation_report['mean_multilead_consensus_recall']:.2%}")
            metric_four.metric("Candidate-rate error", f"{evaluation_report['mean_multilead_consensus_bpm_absolute_error']:.2f} BPM")
            st.caption(f"{evaluation_report['dataset']} · {evaluation_report['license']} · DOI {evaluation_report['dataset_doi']}")
            record_rows = []
            for record in evaluation_report.get("records", []):
                metrics = record.get("multilead_consensus", {})
                record_rows.append({
                    "Public record": record.get("held_out_record", "—"),
                    "F1": f"{metrics.get('f1', 0):.2%}",
                    "Precision": f"{metrics.get('precision', 0):.2%}",
                    "Recall": f"{metrics.get('recall', 0):.2%}",
                    "Rate error": f"{metrics.get('bpm_absolute_error', 0):.2f} BPM",
                })
            st.markdown("#### Public record results")
            st.dataframe(pd.DataFrame(record_rows), hide_index=True, width="stretch")
            st.warning(evaluation_report.get("limitation", "These results are exploratory and not clinical performance claims."))

with st.expander("Common questions & troubleshooting"):
    st.markdown("**What file should I use?** A CSV/TXT with a header and one row per ECG sample. Select abdominal leads only. Use sample recording if you do not have a permitted research file.\n\n**Why does the sampling rate matter?** It converts sample spacing into time. The wrong rate produces the wrong BPM. The app checks it against a recognized time column when present.\n\n**Why are no beats or rates shown?** There may be too few detectable peaks or valid intervals. Check the rate and waveform, and try a longer, cleaner research recording.\n\n**Why are my references rejected?** They must belong to this recording, have the selected units, and fall within its duration. Seconds are relative to recording start; sample indices start at zero.\n\n**Does uploading train the AI?** No. Analysis uses the bundled model; uploads are not used for retraining.\n\n**What is Evidence?** A separate, small public-data evaluation. Those numbers do not describe your upload.")
st.divider()
st.caption("FetalSignal AI · Research prototype · Public, de-identified or synthetic data only")
