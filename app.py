"""FetalSignal AI Streamlit interface.

An explorable research dashboard for abdominal ECG signal separation.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fetalsignal.demo_data import make_demo_recording
from fetalsignal.signal_processing import extract_fetal_signal, heart_rate_bpm, match_peaks, rolling_heart_rate


st.set_page_config(page_title="FetalSignal AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")


def inject_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
            :root { --ink: #eaf3ff; --muted: #95a7b9; --line: rgba(174, 202, 226, .15); --mint: #76f0cf; --lilac: #a99cff; }
            .stApp { background: radial-gradient(circle at 78% 2%, #1b3650 0, #0b1724 36%, #08111c 82%); color: var(--ink); font-family: 'Manrope', sans-serif; }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] { background: rgba(7, 17, 28, .76); border-right: 1px solid var(--line); }
            [data-testid="stSidebar"] * { color: #dbe9f5; }
            .block-container { max-width: 1370px; padding-top: 2.2rem; padding-bottom: 3rem; }
            h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.045em; }
            h1 { font-size: clamp(2.8rem, 5.8vw, 5.6rem) !important; font-weight: 800 !important; line-height: .93 !important; margin-bottom: .75rem !important; }
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
            .status-pill { display: inline-block; border-radius: 99px; padding: .32rem .65rem; font: 500 .7rem 'DM Mono', monospace; letter-spacing: .05em; }
            .status-good { background: rgba(118, 240, 207, .13); color: #76f0cf; border: 1px solid rgba(118, 240, 207, .3); }
            .status-watch { background: rgba(255, 207, 112, .12); color: #ffd98a; border: 1px solid rgba(255, 207, 112, .28); }
            .status-low { background: rgba(255, 139, 159, .12); color: #ff9caf; border: 1px solid rgba(255, 139, 159, .25); }
            .panel { background: rgba(255,255,255,.032); border: 1px solid var(--line); border-radius: 17px; padding: 1.05rem 1.2rem; }
            .caption { color: var(--muted); font-size: .84rem; line-height: 1.55; }
            .stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid var(--line); }
            .stTabs [data-baseweb="tab"] { background: transparent; padding: .8rem 0; color: #93a9bc; font-weight: 700; }
            .stTabs [aria-selected="true"] { color: #76f0cf !important; }
            .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 700; border: 1px solid rgba(118, 240, 207, .42); background: linear-gradient(135deg, #75e8cb, #a29aff); color: #07111d; }
            .stButton > button:hover, .stDownloadButton > button:hover { border-color: #eaffff; color: #07111d; }
            .stAlert { border-radius: 13px; }
            @media (max-width: 760px) { .step-row { grid-template-columns: 1fr; } .hero-orb { width: 190px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ecg_chart(time: np.ndarray, signal: np.ndarray, title: str, color: str, peaks: np.ndarray | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time, y=signal, mode="lines", name="Signal", line=dict(color=color, width=1.45), hovertemplate="%{x:.3f} s<br>%{y:.3f} a.u.<extra></extra>"))
    if peaks is not None and peaks.size:
        figure.add_trace(go.Scatter(x=time[peaks], y=signal[peaks], mode="markers", name="Detected beat", marker=dict(color="#ffd68a", size=7, line=dict(color="#07111d", width=1)), hovertemplate="Beat at %{x:.3f} s<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=295, margin=dict(l=6, r=6, t=44, b=8), title=dict(text=title, font=dict(size=15)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", xaxis=dict(title="Time (seconds)", gridcolor="rgba(189,210,226,.09)", zeroline=False), yaxis=dict(title="Amplitude", gridcolor="rgba(189,210,226,.09)", zeroline=False), legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)), hovermode="x unified")
    return figure


def rhythm_chart(centers: np.ndarray, rates: np.ndarray) -> go.Figure:
    figure = go.Figure()
    figure.add_hrect(y0=110, y1=160, fillcolor="rgba(118, 240, 207, .07)", line_width=0, annotation_text="candidate-rate review band", annotation_position="top left", annotation_font_color="#94c9bb")
    figure.add_trace(go.Scatter(x=centers, y=rates, mode="lines+markers", line=dict(color="#76f0cf", width=2), marker=dict(size=6, color="#b7abff"), connectgaps=False, hovertemplate="%{x:.1f} s<br>%{y:.0f} BPM<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=220, margin=dict(l=6, r=6, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", xaxis=dict(title="Window center (seconds)", gridcolor="rgba(189,210,226,.09)"), yaxis=dict(title="Candidate BPM", gridcolor="rgba(189,210,226,.09)", range=[80, 230]), showlegend=False)
    return figure


def read_table(uploaded) -> pd.DataFrame:
    """Read a CSV or delimiter-separated text signal file without persisting it."""
    try:
        return pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError("We could not read that file. Upload a CSV or plain-text table with a header row.") from error


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 3]


def extract_column(frame: pd.DataFrame, column: str, sample_rate: float) -> np.ndarray:
    signal = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    if signal.size < int(sample_rate * 3):
        raise ValueError("Use a recording with at least 3 seconds of signal data.")
    if not np.isfinite(signal).all() or np.std(signal) < 1e-10:
        raise ValueError("The selected column must contain a changing numeric ECG signal.")
    return signal


def infer_time(frame: pd.DataFrame, expected_length: int, sample_rate: float) -> tuple[np.ndarray, str]:
    time_candidates = [c for c in frame.columns if c.lower() in {"time", "timestamp", "time_s", "seconds", "t"}]
    if time_candidates:
        values = pd.to_numeric(frame[time_candidates[0]], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == expected_length and np.all(np.diff(values) > 0):
            return values - values[0], f"Using `{time_candidates[0]}` from your file"
    return np.arange(expected_length) / sample_rate, "Using the sampling rate selected in the sidebar"


def load_reference(uploaded, sample_rate: float, total_samples: int) -> np.ndarray:
    frame = read_table(uploaded)
    candidates = numeric_columns(frame)
    if not candidates:
        raise ValueError("Reference file needs a numeric time or sample-index column.")
    values = pd.to_numeric(frame[candidates[0]], errors="coerce").dropna().to_numpy(dtype=float)
    if not values.size:
        raise ValueError("No usable reference values found.")
    if np.nanmax(values) <= total_samples / sample_rate * 1.5:
        values = values * sample_rate
    reference = np.rint(values).astype(int)
    return np.unique(reference[(reference >= 0) & (reference < total_samples)])


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


def make_annotation_export(time: np.ndarray, result) -> bytes:
    fetal_times = time[result.fetal_peaks]
    return pd.DataFrame({"candidate_beat_sample": result.fetal_peaks, "candidate_beat_time_seconds": fetal_times}).to_csv(index=False).encode("utf-8")


def make_report(data_note: str, sample_rate: float, fetal_bpm: float | None, maternal_bpm: float | None, result) -> bytes:
    report = f"""FetalSignal AI — exploratory analysis report

Input: {data_note}
Sampling rate: {sample_rate} Hz
Candidate fetal rate: {metric_value(fetal_bpm)}
Candidate maternal rate: {metric_value(maternal_bpm)}
Detected fetal candidate beats: {result.fetal_peaks.size}
Signal-quality heuristic: {result.quality_score:.0f} / 100

Interpretation: This is a signal-processing output for de-identified research data. It is not a diagnosis, a clinical confidence value, or a medical-device reading. Candidate beats should be validated against an appropriate reference annotation set before reporting performance.
"""
    return report.encode("utf-8")


def switch_to_upload_mode() -> None:
    """Switch the radio input before the next Streamlit script run."""
    st.session_state.source_mode = "Upload signal data"


inject_style()

with st.sidebar:
    st.markdown("<div class='brand-row'><span class='spark'>✦</span> FetalSignal AI</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Data lab")
    source = st.radio("Choose a recording", ["Try the interactive demo", "Upload signal data"], key="source_mode", label_visibility="collapsed")
    st.caption("FetalSignal reads waveform data—not a screenshot of an ECG chart.")
    if source == "Try the interactive demo":
        st.markdown("#### Demo controls")
        demo_length = st.select_slider("Recording length", options=[15, 30, 45, 60], value=30, format_func=lambda seconds: f"{seconds} seconds")
        demo_noise = st.select_slider("Noise level", options=["Low", "Standard", "High"], value="Standard")
        sample_rate = st.selectbox("Sample rate", [250, 500, 1_000], index=1, format_func=lambda value: f"{value} Hz")
        powerline, reference_upload = 50, None
    else:
        st.markdown("#### Recording settings")
        sample_rate = st.number_input("Sampling rate (Hz)", min_value=100, max_value=2_000, value=500, step=50)
        powerline = st.selectbox("Electrical interference", [50, 60], format_func=lambda value: f"{value} Hz")
        reference_upload = None
        st.caption("Set the recording rate here, then upload the waveform in the main workspace.")

hero_left, hero_right = st.columns([1.55, .75], vertical_alignment="center")
with hero_left:
    st.markdown("<div class='brand-row'><span class='spark'>✦</span> SIGNAL SEPARATION STUDIO</div>", unsafe_allow_html=True)
    st.title("Find the rhythm\ninside the noise.")
    st.markdown("<p class='hero-copy'>FetalSignal AI helps researchers explore an <strong>abdominal ECG waveform</strong>, reduce the dominant maternal pattern, and inspect a fetal cardiac-signal candidate—visually, transparently, and without overclaiming.</p>", unsafe_allow_html=True)
    st.markdown("<div class='safe-note'>For research and education only. This prototype is not a medical device and cannot diagnose, monitor, or make decisions for a patient.</div>", unsafe_allow_html=True)
    if source == "Try the interactive demo":
        st.button("Upload an ECG recording", type="primary", key="hero_upload", on_click=switch_to_upload_mode)
with hero_right:
    st.markdown("<div class='hero-orb'><div class='orb-line'>ABDOMINAL ECG<br>↓<br>FETAL CANDIDATE</div></div>", unsafe_allow_html=True)

st.markdown("""<div class='step-row'><div class='step-card'><div class='step-number'>01 · INPUT</div><b>Bring a waveform</b><span>Upload a CSV/TXT ECG signal or begin with a synthetic, labelled demo.</span></div><div class='step-card'><div class='step-number'>02 · SEPARATE</div><b>Reduce the dominant pattern</b><span>Filter the mixture, estimate maternal beats, then inspect the residual signal.</span></div><div class='step-card'><div class='step-number'>03 · EXPLORE</div><b>Review, validate, export</b><span>Examine candidate beats, calculate metrics, and download the transformed waveform.</span></div></div>""", unsafe_allow_html=True)

uploaded = None
if source == "Upload signal data":
    st.markdown("<div class='section-label'>Start an analysis</div>", unsafe_allow_html=True)
    st.markdown("## Upload an abdominal ECG recording")
    upload_col, result_col = st.columns([1.15, .85], vertical_alignment="center")
    with upload_col:
        uploaded = st.file_uploader("1. Choose your waveform file", type=["csv", "txt"], help="Use one row per sample and a header for each ECG lead.")
        reference_upload = st.file_uploader("2. Add reference beats (optional)", type=["csv", "txt"], help="First numeric column: fetal-beat times in seconds or sample indices.")
        st.caption("We process only de-identified research data. The uploaded file stays in the current app session.")
    with result_col:
        st.markdown("<div class='panel'><b>What you get back</b><p class='caption'>A cleaned abdominal waveform, a maternal-pattern estimate, a fetal-signal candidate, candidate beat markers, estimated candidate BPM, signal-quality review, and downloadable results.</p><p class='caption'>Use CSV or TXT signal values—not an image or screenshot of an ECG chart.</p></div>", unsafe_allow_html=True)

raw_signal: np.ndarray | None = None
time: np.ndarray | None = None
reference: np.ndarray | None = None
data_note = ""

if source == "Try the interactive demo":
    noise_by_label = {"Low": 0.025, "Standard": 0.045, "High": 0.115}
    demo = make_demo_recording(duration_seconds=demo_length, sample_rate=int(sample_rate), noise_level=noise_by_label[demo_noise])
    raw_signal, time = demo["abdominal"], demo["time"]
    reference = np.rint(demo["fetal_reference_seconds"] * sample_rate).astype(int)
    data_note = "Synthetic labelled demo · no patient data"
else:
    if uploaded is None:
        st.markdown("<div class='panel'><b>Ready when your data is.</b><p class='caption'>Choose an ECG waveform above to start. A screenshot of an ECG cannot provide original sample values, so this app intentionally asks for signal data instead.</p></div>", unsafe_allow_html=True)
    else:
        try:
            uploaded_frame = read_table(uploaded)
            choices = numeric_columns(uploaded_frame)
            if not choices:
                raise ValueError("No numeric signal columns found. Add a header row and one column of ECG samples.")
            with st.sidebar:
                selected_column = st.selectbox("Abdominal ECG column", choices, help="Choose one abdominal ECG lead to analyze.")
            raw_signal = extract_column(uploaded_frame, selected_column, sample_rate)
            time, time_note = infer_time(uploaded_frame, raw_signal.size, sample_rate)
            data_note = f"{uploaded.name} · {selected_column} · {time_note}"
            if reference_upload is not None:
                reference = load_reference(reference_upload, sample_rate, raw_signal.size)
        except ValueError as error:
            st.error(str(error))

if raw_signal is not None and time is not None:
    result = extract_fetal_signal(raw_signal, sample_rate, powerline)
    fetal_bpm, maternal_bpm = heart_rate_bpm(result.fetal_peaks, sample_rate), heart_rate_bpm(result.maternal_peaks, sample_rate)
    label, label_class = quality_label(result.quality_score)
    st.markdown("<div class='section-label'>Analysis workspace</div>", unsafe_allow_html=True)
    title_col, status_col = st.columns([5, 1.2], vertical_alignment="center")
    title_col.markdown("## Your signal, unpacked")
    status_col.markdown(f"<span class='status-pill {label_class}'>{label}</span>", unsafe_allow_html=True)
    st.caption(data_note)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Fetal candidate rate", metric_value(fetal_bpm))
    metric_columns[1].metric("Candidate beats", f"{result.fetal_peaks.size}")
    metric_columns[2].metric("Maternal rate", metric_value(maternal_bpm))
    metric_columns[3].metric("Signal quality", f"{result.quality_score:.0f} / 100")

    overview_tab, signals_tab, validation_tab, method_tab = st.tabs(["Overview", "Signal lab", "Validation", "Method"])
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
        fetal_peaks = result.fetal_peaks[(result.fetal_peaks >= start_index) & (result.fetal_peaks < end_index)] - start_index
        left, right = st.columns(2)
        left.plotly_chart(ecg_chart(local_time, raw_signal[start_index:end_index], "Abdominal ECG mixture", "#75d9ff", raw_peaks), width="stretch")
        right.plotly_chart(ecg_chart(local_time, result.residual[start_index:end_index], "Fetal cardiac-signal candidate", "#aa9dff", fetal_peaks), width="stretch")
        st.markdown("<p class='caption'>Markers indicate algorithmic candidate peaks. They are not confirmed fetal beats unless compared with an appropriate reference annotation set.</p>", unsafe_allow_html=True)
        centers, rates = rolling_heart_rate(result.fetal_peaks, sample_rate, float(time[-1] - time[0]))
        if centers.size:
            st.markdown("### Candidate rhythm over time")
            st.plotly_chart(rhythm_chart(centers, rates), width="stretch")
            st.caption("Each point summarizes candidate-peak intervals in a sliding 10-second window. Gaps mean the detector did not find enough usable intervals—not a medical event.")
    with signals_tab:
        st.markdown("### Signal-separation layers")
        left, right = st.columns(2)
        left.plotly_chart(ecg_chart(time, result.cleaned, "1 · Filtered abdominal ECG", "#74e7c9"), width="stretch")
        right.plotly_chart(ecg_chart(time, result.maternal_component, "2 · Estimated maternal pattern", "#ff9ec3"), width="stretch")
        st.plotly_chart(ecg_chart(time, result.residual, "3 · Residual fetal candidate signal", "#b1a8ff", result.fetal_peaks), width="stretch")
        download_one, download_two, download_three = st.columns(3)
        download_one.download_button("Processed waveform CSV", make_export(time, raw_signal, result), "fetalsignal_processed.csv", "text/csv", width="stretch")
        download_two.download_button("Candidate-beat CSV", make_annotation_export(time, result), "fetalsignal_candidate_beats.csv", "text/csv", width="stretch")
        download_three.download_button("Analysis report", make_report(data_note, sample_rate, fetal_bpm, maternal_bpm, result), "fetalsignal_report.txt", "text/plain", width="stretch")
    with validation_tab:
        st.markdown("### Evidence, not guesses")
        if reference is not None and reference.size:
            metrics = match_peaks(result.fetal_peaks, reference, sample_rate)
            reference_bpm = heart_rate_bpm(reference, sample_rate)
            rate_error = abs(fetal_bpm - reference_bpm) if fetal_bpm is not None and reference_bpm is not None else None
            one, two, three, four = st.columns(4)
            one.metric("Precision", f"{metrics['precision']:.1%}")
            two.metric("Recall", f"{metrics['recall']:.1%}")
            three.metric("F1 score", f"{metrics['f1']:.1%}")
            four.metric("Rate error", "—" if rate_error is None else f"{rate_error:.1f} BPM")
            st.caption("Metrics compare algorithmic candidate peaks with the supplied reference times. Demo results are illustrative only; upload a documented public dataset to make research claims.")
        else:
            st.info("Add a reference-beat CSV/TXT in the sidebar to calculate precision, recall, F1, and rate error. Its first numeric column can contain beat times in seconds or sample indices.")
            st.markdown("<div class='panel'><b>What good validation looks like</b><p class='caption'>Keep reference annotations separate from the input waveform. Define your matching tolerance before testing, document the dataset and preprocessing settings, and report results on recordings not used to tune the pipeline.</p></div>", unsafe_allow_html=True)
    with method_tab:
        st.markdown("### A transparent baseline, step by step")
        st.markdown("""- **Clean** — removes slow baseline drift, the selected power-line frequency, and out-of-band content.
- **Find maternal candidates** — looks for recurring maternal QRS-like peaks in a maternal-frequency view.
- **Estimate and reduce** — builds a median maternal beat template and subtracts the overlap-adjusted estimate.
- **Find fetal candidates** — filters the residual and searches for repeating peaks in a fetal-rate range.
- **Score cautiously** — signal quality combines candidate-interval plausibility, regularity, and residual energy. It is not a calibrated confidence or medical risk score.""")
        st.warning("Limitations: abdominal ECG quality can change with electrode placement, motion, maternal rhythm, gestational age, and noise. A candidate signal can be wrong. This project is for de-identified research data only.")

st.markdown("<br><p class='caption'>FetalSignal AI · Exploratory signal processing for de-identified research recordings.</p>", unsafe_allow_html=True)
