"""FetalSignal AI — explainable abdominal-ECG research workspace."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fetalsignal.demo_data import make_demo_recording
from fetalsignal.ml import select_model_candidates
from fetalsignal.signal_processing import (
    extract_fetal_signal,
    fuse_multichannel_peaks,
    heart_rate_bpm,
    match_peaks,
    rolling_heart_rate,
)
from fetalsignal.ui import (
    COLORS,
    brand,
    diagnostics_chart,
    ecg_chart,
    evidence_chart,
    footer,
    hero,
    inject_theme,
    insight_card,
    pipeline,
    quality_diagnostics,
    result_banner,
    rhythm_chart,
    safety_note,
    section,
    validation_timeline,
)
from fetalsignal.uploads import TIME_COLUMNS, extract_column, infer_time, load_reference, numeric_columns, read_table


ROOT = Path(__file__).parent
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False, "responsive": True}

st.set_page_config(
    page_title="FetalSignal AI · Research workspace",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="auto",
)


def metric_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} BPM"


def quality_label(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Strong candidate pattern", "good"
    if score >= 45:
        return "Review signal quality", "watch"
    return "Weak candidate pattern", "low"


def make_waveform_export(time: np.ndarray, raw: np.ndarray, result) -> bytes:
    return pd.DataFrame(
        {
            "time_seconds": time,
            "raw_abdominal_ecg": raw,
            "filtered_ecg": result.cleaned,
            "maternal_template_estimate": result.maternal_component,
            "fetal_candidate_signal": result.residual,
        }
    ).to_csv(index=False).encode("utf-8")


def make_annotation_export(time: np.ndarray, peaks: np.ndarray) -> bytes:
    return pd.DataFrame(
        {
            "candidate_beat_sample": peaks,
            "candidate_beat_time_seconds": time[peaks],
        }
    ).to_csv(index=False).encode("utf-8")


def make_report(
    data_note: str,
    sample_rate: float,
    fetal_bpm: float | None,
    maternal_bpm: float | None,
    result,
    fetal_peaks: np.ndarray,
    analysis_method: str,
) -> bytes:
    report = f"""FetalSignal AI — exploratory analysis report

INPUT
{data_note}

ANALYSIS
Sampling rate: {sample_rate:g} Hz
Candidate fetal rate: {metric_value(fetal_bpm)}
Candidate maternal rate: {metric_value(maternal_bpm)}
Candidate beats: {fetal_peaks.size}
Detection method: {analysis_method}
Primary-lead quality heuristic: {result.quality_score:.0f} / 100

INTERPRETATION
This is signal-processing output for permitted research data. It is not a diagnosis,
clinical confidence value, or medical-device reading. Candidate beats must be
validated against an appropriate independent annotation set before performance is
reported.
"""
    return report.encode("utf-8")


def make_manifest(
    data_note: str,
    sample_rate: float,
    fetal_bpm: float | None,
    maternal_bpm: float | None,
    quality_score: float,
    candidate_count: int,
    analysis_method: str,
) -> bytes:
    return json.dumps(
        {
            "product": "FetalSignal AI",
            "purpose": "exploratory research",
            "input": data_note,
            "sample_rate_hz": sample_rate,
            "candidate_fetal_bpm": fetal_bpm,
            "candidate_maternal_bpm": maternal_bpm,
            "candidate_beat_count": candidate_count,
            "quality_heuristic_0_to_100": quality_score,
            "analysis_method": analysis_method,
            "medical_device": False,
            "diagnostic_output": False,
        },
        indent=2,
    ).encode("utf-8")


def load_evaluation_report() -> dict | None:
    try:
        return json.loads((ROOT / "results/adfecgdb_leave_one_record_out.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def choose_source(source: str) -> None:
    st.session_state.source_mode = source


def move_guide(direction: int) -> None:
    st.session_state.guide_step = max(0, min(3, st.session_state.get("guide_step", 0) + direction))


def guided_walkthrough() -> None:
    with st.expander("New here? Take the 60-second guided walkthrough", expanded=False):
        step = st.session_state.get("guide_step", 0)
        titles = ["Choose a recording", "Read the snapshot", "Inspect and validate", "Export responsibly"]
        instructions = [
            "Choose **Quick sample** for an instant, labelled example. Choose **Simulation** to change rhythm, noise, and duration. Choose **Upload** to analyze only the permitted CSV/TXT waveform you provide.",
            "Read the four result cards as research diagnostics. Candidate BPM comes from candidate-beat spacing. The quality number is a signal heuristic—not accuracy, medical confidence, or a patient score.",
            "Use **Overview** to inspect peaks, **How it works** to see the separation layers, and **Check accuracy** to compare with independent reference annotations. A visually convincing waveform is not enough on its own.",
            "Download the processed waveform, candidate locations, report, or JSON manifest. Keep the research-only limitation with every exported or presented result.",
        ]
        st.caption(f"GUIDED TOUR · {step + 1} OF 4")
        st.markdown(f"### {titles[step]}")
        st.markdown(instructions[step])
        st.progress((step + 1) / 4)
        back, forward = st.columns(2)
        back.button("← Previous", disabled=step == 0, on_click=move_guide, args=(-1,), width="stretch")
        if step < 3:
            forward.button("Next step →", on_click=move_guide, args=(1,), width="stretch")
        else:
            forward.button("Replay guide", on_click=move_guide, args=(-3,), width="stretch")


def sample_downloads() -> None:
    with st.expander("Download the synthetic practice files"):
        st.caption("Use 500 Hz, select abdominal_ecg, and choose reference units Seconds.")
        options = [
            ("fetalsignal_sample_recording.csv", "Example waveform CSV"),
            ("fetalsignal_sample_reference_beats.csv", "Matching reference beats CSV"),
        ]
        columns = st.columns(2)
        for column, (name, label) in zip(columns, options):
            path = ROOT / "sample_data" / name
            if path.exists():
                column.download_button(label, path.read_bytes(), name, "text/csv", width="stretch")
            else:
                column.info("Example file unavailable")


def explain_result(
    fetal_bpm: float | None,
    quality_score: float,
    candidate_count: int,
    model_enabled: bool,
    consensus: bool,
) -> tuple[str, str, str]:
    method = "Multiple aligned leads supported these candidates." if consensus else (
        "The experimental ML ranker reviewed the baseline candidates." if model_enabled else "The transparent signal-processing baseline selected these candidates."
    )
    if fetal_bpm is None or candidate_count < 3:
        return (
            "No stable candidate rhythm",
            "There were not enough repeating residual peaks to estimate a candidate rate. Short duration, noise, or incomplete maternal-pattern reduction can cause this.",
            "Check the sampling rate and selected lead, inspect the signal layers, and try a longer or cleaner permitted research waveform.",
        )
    if quality_score >= 75:
        return (
            "A repeatable candidate rhythm is visible",
            f"The median spacing of {candidate_count} candidate peaks corresponds to approximately {fetal_bpm:.0f} BPM. {method}",
            "Inspect marker placement, review the diagnostic components, and use independent references before reporting performance.",
        )
    return (
        "A candidate rhythm needs careful review",
        f"The workspace estimates approximately {fetal_bpm:.0f} BPM from {candidate_count} peaks, but the signal heuristic is {quality_score:.0f}/100. Noise or remaining maternal interference may be contributing. {method}",
        "Review the residual markers and validate with independent annotations. Do not treat the estimate as a patient assessment.",
    )


inject_theme()

with st.sidebar:
    brand()
    st.caption("Explainable fetal-ECG research")
    st.divider()
    st.markdown("#### Workspace controls")
    st.caption("Your source and analysis settings stay visible while you explore results.")

st.session_state.setdefault("source_mode", "Use sample recording")
hero()
safety_note()

section("Start here", "How would you like to begin?", "Choose one path. The sample works instantly; no setup is required.")
source_labels = {
    "Use sample recording": "Quick sample · ready now",
    "Try the interactive demo": "Simulation · change the signal",
    "Upload signal data": "Upload · analyze my file",
}
source = st.radio(
    "Choose a starting point",
    ["Use sample recording", "Try the interactive demo", "Upload signal data"],
    key="source_mode",
    horizontal=True,
    format_func=source_labels.get,
    help="Sample: complete labelled example. Demo: adjustable synthetic signal. Upload: your permitted waveform.",
)

if source == "Use sample recording":
    st.success("**Ready.** The labelled sample is loaded and analyzed automatically. Scroll to the result snapshot below.")
elif source == "Try the interactive demo":
    st.info("**Build a simulation.** Choose a preset below or create your own signal experiment.")
else:
    st.info("**Analyze your file.** The app will use only the waveform and settings you provide.")

guided_walkthrough()

with st.sidebar:
    if source == "Use sample recording":
        st.success("Labelled synthetic sample ready")
        st.caption("30 seconds · 500 Hz · one abdominal lead · independent reference beats")
    elif source == "Try the interactive demo":
        st.info("Simulation mode active")
        st.caption("Controls are in the main workspace so they are easy to find on desktop and mobile.")
    else:
        st.info("Upload mode active")
        st.caption("Your data remains separate from the built-in sample.")
    st.divider()
    st.markdown("##### Research glossary")
    with st.expander("ECG and signal terms"):
        st.markdown("**ECG** · electrical heartbeat recording\n\n**Lead** · one recording channel\n\n**BPM** · beats per minute\n\n**Candidate** · a possible algorithmic beat\n\n**Residual** · signal remaining after maternal-pattern reduction")
    with st.expander("Privacy and safety"):
        st.caption("Use public, de-identified, synthetic, or otherwise permitted research data only. Uploaded data is processed during the app session and is not used to retrain the model.")

uploaded = None
reference_units = "Seconds"
reference_column = None
sample_rate, powerline, reference_upload = 500, 50, None

if source == "Try the interactive demo":
    section("Simulation setup", "Choose a scenario", "Every change regenerates the synthetic waveform immediately. Nothing here represents a patient.")
    scenario_col, length_col, rate_col = st.columns(3)
    demo_scenario = scenario_col.selectbox(
        "Scenario",
        ["Clean baseline", "Motion challenge", "Closer rhythms", "Custom experiment"],
        help="Each preset intentionally produces a different waveform and candidate rhythm.",
    )
    demo_length = length_col.select_slider("Duration", options=[15, 30, 45, 60], value=30, format_func=lambda value: f"{value} seconds")
    sample_rate = rate_col.selectbox("Samples per second", [250, 500, 1_000], index=1, format_func=lambda value: f"{value} Hz")
    presets = {
        "Clean baseline": dict(fetal=132.0, maternal=68.0, noise=0.025, seed=3),
        "Motion challenge": dict(fetal=158.0, maternal=88.0, noise=0.125, seed=29),
        "Closer rhythms": dict(fetal=116.0, maternal=94.0, noise=0.065, seed=11),
    }
    if demo_scenario == "Custom experiment":
        custom_one, custom_two = st.columns(2)
        demo_fetal_bpm = custom_one.slider("Candidate rhythm", 100, 190, 150, 1, format="%d BPM")
        demo_maternal_bpm = custom_two.slider("Maternal rhythm", 50, 110, 78, 1, format="%d BPM")
        advanced_one, advanced_two = st.columns(2)
        demo_noise_level = advanced_one.slider("Noise level", 0.01, 0.15, 0.06, 0.01)
        demo_seed = advanced_two.number_input("Repeatable seed", min_value=1, max_value=999, value=17, step=1)
    else:
        preset = presets[demo_scenario]
        demo_fetal_bpm = preset["fetal"]
        demo_maternal_bpm = preset["maternal"]
        demo_noise_level = preset["noise"]
        demo_seed = preset["seed"]
        st.caption(f"Preset: {demo_fetal_bpm:.0f} candidate BPM · {demo_maternal_bpm:.0f} maternal BPM · noise {demo_noise_level:.3f}")
    st.success("Simulation ready. Results below update automatically.")
    st.button("Use my own ECG file instead", on_click=choose_source, args=("Upload signal data",))
elif source == "Upload signal data":
    section("Upload", "Add a permitted waveform", "Only two items are required: the file and its sampling rate. Everything else is optional or detected from the file.")
    sample_downloads()
    uploaded = st.file_uploader(
        "1. Choose a CSV or TXT waveform",
        type=["csv", "txt"],
        help="A header row and one sample per row; comma, tab, semicolon, or space separated. Maximum 25 MB and 600,000 samples.",
    )
    setting_one, setting_two = st.columns(2)
    sample_rate = setting_one.number_input("2. Samples per second (Hz)", min_value=100, max_value=2_000, value=500, step=1, help="Use the sampling rate documented with your recording. If the file includes time_seconds, the app checks this automatically.")
    powerline = setting_two.selectbox("Power-line frequency", [50, 60], format_func=lambda value: f"{value} Hz", help="Most countries use 50 Hz; North America commonly uses 60 Hz.")
    with st.expander("Optional: add reference beats to check accuracy"):
        st.caption("Reference annotations must belong to this exact recording. They are used only for validation.")
        reference_upload = st.file_uploader("Reference-beat file", type=["csv", "txt"])
        reference_units = st.selectbox("Reference units", ["Seconds", "Sample indices (start at 0)"])
        if reference_upload is not None:
            try:
                reference_choices = numeric_columns(read_table(reference_upload))
                if not reference_choices:
                    raise ValueError("No numeric reference-beat column was found.")
                reference_column = st.selectbox("Reference-beat column", reference_choices)
            except ValueError as error:
                st.warning(str(error))
elif source == "Use sample recording":
    try:
        uploaded = io.BytesIO((ROOT / "sample_data/fetalsignal_sample_recording.csv").read_bytes())
        uploaded.name = "Synthetic sample recording"
        reference_upload = io.BytesIO((ROOT / "sample_data/fetalsignal_sample_reference_beats.csv").read_bytes())
    except OSError:
        st.warning("The bundled sample is unavailable. Try the interactive demo instead.")

raw_signal: np.ndarray | None = None
time: np.ndarray | None = None
reference: np.ndarray | None = None
data_note = ""
precomputed_result = None
precomputed_fetal_peaks: np.ndarray | None = None
analysis_method = ""
reference_error = None

if source == "Try the interactive demo":
    demo = make_demo_recording(
        duration_seconds=demo_length,
        sample_rate=int(sample_rate),
        maternal_bpm=float(demo_maternal_bpm),
        fetal_bpm=float(demo_fetal_bpm),
        noise_level=float(demo_noise_level),
        seed=int(demo_seed),
    )
    raw_signal, time = demo["abdominal"], demo["time"]
    reference = np.rint(demo["fetal_reference_seconds"] * sample_rate).astype(int)
    data_note = (
        f"Interactive simulation · {demo_scenario} · target fetal-candidate rhythm {demo_fetal_bpm:.0f} BPM · "
        f"target maternal rhythm {demo_maternal_bpm:.0f} BPM · noise {demo_noise_level:.3f} · seed {int(demo_seed)} · no patient data"
    )
elif source == "Use sample recording":
    if uploaded is not None:
        try:
            uploaded_frame = read_table(uploaded)
            raw_signal = extract_column(uploaded_frame, "abdominal_ecg", sample_rate)
            time, time_note = infer_time(uploaded_frame, raw_signal.size, sample_rate)
            with st.spinner("Preparing the labelled example…"):
                precomputed_result = extract_fetal_signal(raw_signal, sample_rate, powerline)
            reference = load_reference(reference_upload, sample_rate, raw_signal.size, reference_units, reference_column)
            analysis_method = "Single-lead signal-processing baseline"
            data_note = f"Labelled synthetic sample · abdominal_ecg · {time_note}"
        except ValueError as error:
            raw_signal, time, precomputed_result = None, None, None
            st.error(f"The bundled sample could not be prepared: {error}")
else:
    if uploaded is None:
        st.info("Choose a waveform above, or select Quick sample for a complete example.")
    else:
        try:
            uploaded_frame = read_table(uploaded)
            choices = [column for column in numeric_columns(uploaded_frame) if column.lower() not in TIME_COLUMNS]
            if not choices:
                raise ValueError("No numeric signal columns found. Add headers and numeric ECG samples.")
            selected_columns = st.multiselect(
                "3. Choose the ECG channel(s)",
                choices,
                default=choices[: min(4, len(choices))],
                max_selections=8,
                help="Select abdominal leads only. Three or more aligned leads enable consensus detection.",
            )
            if not selected_columns:
                raise ValueError("Choose at least one abdominal ECG column to analyze.")
            signals = [extract_column(uploaded_frame, column, sample_rate) for column in selected_columns]
            time, time_note = infer_time(uploaded_frame, signals[0].size, sample_rate)
            with st.spinner("Separating the signal and ranking candidate beats…"):
                results = [extract_fetal_signal(signal, sample_rate, powerline) for signal in signals]
            primary_index = int(np.argmax([result.quality_score for result in results]))
            raw_signal, precomputed_result = signals[primary_index], results[primary_index]
            if len(results) >= 3:
                consensus = fuse_multichannel_peaks([result.fetal_peaks for result in results], sample_rate, minimum_channels=3)
                if consensus.size >= 3:
                    precomputed_fetal_peaks = consensus
                    analysis_method = f"{len(results)}-lead consensus (support required from 3 leads)"
                else:
                    analysis_method = f"Best single lead selected from {len(results)} uploaded leads"
            data_note = f"{uploaded.name} · displayed lead {selected_columns[primary_index]} · {len(results)} lead(s) · {time_note}"
            if reference_upload is not None:
                try:
                    reference = load_reference(reference_upload, sample_rate, raw_signal.size, reference_units, reference_column)
                except ValueError as error:
                    reference_error = str(error)
                    st.warning(f"Reference file needs attention: {reference_error} Waveform analysis is still available; validation is not.")
        except ValueError as error:
            raw_signal, time, precomputed_result, precomputed_fetal_peaks = None, None, None, None
            st.error(str(error))
            st.caption("Correct the file or setting above. Invalid waveforms are never analyzed partially.")

if raw_signal is not None and time is not None:
    result = precomputed_result or extract_fetal_signal(raw_signal, sample_rate, powerline)
    if precomputed_fetal_peaks is not None:
        fetal_peaks, model_probabilities = precomputed_fetal_peaks, None
    else:
        fetal_peaks, model_probabilities = select_model_candidates(result.residual, result.fetal_peaks, sample_rate)
        analysis_method = "Single-lead ML-assisted candidate ranking" if model_probabilities is not None else "Single-lead signal-processing baseline"

    fetal_bpm = heart_rate_bpm(fetal_peaks, sample_rate)
    maternal_bpm = heart_rate_bpm(result.maternal_peaks, sample_rate, min_bpm=39, max_bpm=131)
    status, status_class = quality_label(result.quality_score)
    explanation_title, explanation_text, next_step = explain_result(
        fetal_bpm,
        result.quality_score,
        fetal_peaks.size,
        model_probabilities is not None,
        precomputed_fetal_peaks is not None,
    )

    section("Your result", "See what the app found", "Start with the summary. Open the detailed views only when you need them.")
    result_banner(explanation_title, explanation_text, status, status_class)
    st.caption(data_note)
    st.caption(f"{raw_signal.size / sample_rate:.1f} seconds · {sample_rate:g} samples/second · {analysis_method}")
    if source in {"Try the interactive demo", "Use sample recording"}:
        st.caption("SYNTHETIC EXAMPLE · no patient data · demonstrates the workflow, not real-world performance")
    if precomputed_fetal_peaks is not None:
        st.caption("3-lead consensus is active: each retained candidate needs support from at least three aligned abdominal leads.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Possible fetal rate", metric_value(fetal_bpm), help="Estimated from median candidate-beat spacing; not a confirmed fetal heart rate.")
    metric_columns[1].metric("Possible beats found", f"{fetal_peaks.size}", help="Algorithmic candidates across the full recording.")
    metric_columns[2].metric("Maternal candidate rate", metric_value(maternal_bpm), help="Experimental estimate of the stronger repeating pattern.")
    metric_columns[3].metric("Primary lead quality", f"{result.quality_score:.0f} / 100", help="A signal heuristic—not accuracy, medical confidence, or risk.")

    insight_one, insight_two = st.columns(2)
    with insight_one:
        insight_card("What the algorithm found", explanation_text)
    with insight_two:
        insight_card("Best next action", next_step)

    section("Explore", "Understand your result", "Start with Overview, then open the other views for more detail.")
    overview_tab, studio_tab, validation_tab, evidence_tab, method_tab = st.tabs(["Overview", "How it works", "Check accuracy", "Public testing", "Technical method"])

    preview_seconds = min(15, int(np.ceil(time[-1] - time[0])))
    range_limit = float(time[-1] - preview_seconds)

    with overview_tab:
        control_one, control_two = st.columns([3, 1])
        with control_one:
            if range_limit > 0.5:
                start_second = st.slider("Inspection window", 0.0, range_limit, 0.0, 0.5, format="%.1f s")
            else:
                start_second = 0.0
                st.caption("Showing the complete short recording")
        with control_two:
            st.metric("Window", f"{preview_seconds} s", help="Charts show this many seconds; exports include the full recording.")
        start_index = int(np.searchsorted(time, start_second, side="left"))
        end_index = min(raw_signal.size, int(np.searchsorted(time, start_second + preview_seconds, side="right")))
        local_time = time[start_index:end_index]
        raw_peaks = result.maternal_peaks[(result.maternal_peaks >= start_index) & (result.maternal_peaks < end_index)] - start_index
        local_fetal_peaks = fetal_peaks[(fetal_peaks >= start_index) & (fetal_peaks < end_index)] - start_index
        mixed, residual = st.columns(2)
        with mixed:
            st.markdown("#### Mixed abdominal ECG")
            st.caption("Stronger repeating peaks plus weaker components and noise")
            st.plotly_chart(ecg_chart(local_time, raw_signal[start_index:end_index], COLORS["cyan"], raw_peaks, "Maternal candidate"), key="overview_mixed", width="stretch", config=CHART_CONFIG)
        with residual:
            st.markdown("#### Possible fetal component")
            st.caption("Residual after maternal-pattern reduction; markers remain candidates")
            st.plotly_chart(ecg_chart(local_time, result.residual[start_index:end_index], COLORS["violet"], local_fetal_peaks), key="overview_residual", width="stretch", config=CHART_CONFIG)

        centers, rates = rolling_heart_rate(fetal_peaks, sample_rate, float(time[-1] - time[0]))
        rhythm_col, diagnostic_col = st.columns([1.35, 1])
        with rhythm_col:
            st.markdown("#### Candidate rhythm over time")
            if centers.size:
                st.plotly_chart(rhythm_chart(centers, rates), key="overview_rhythm", width="stretch", config=CHART_CONFIG)
            else:
                st.info("The recording is too short or sparse for a rhythm-over-time view.")
        with diagnostic_col:
            st.markdown("#### Why the quality score moved")
            diagnostic_values = quality_diagnostics(result.cleaned, result.residual, fetal_peaks, sample_rate)
            st.plotly_chart(diagnostics_chart(diagnostic_values), key="overview_diagnostics", width="stretch", config=CHART_CONFIG)
        st.caption("These diagnostics describe algorithmic plausibility, regularity, and residual energy. They are not calibrated clinical confidence scores.")

    with studio_tab:
        st.markdown("### How the signal is separated")
        st.caption("Follow the same time window through each stage. Downloads still contain the full recording.")
        cleaned_col, maternal_col = st.columns(2)
        with cleaned_col:
            st.markdown("#### 1 · Cleaned mixture")
            st.plotly_chart(ecg_chart(local_time, result.cleaned[start_index:end_index], COLORS["mint"]), key="studio_cleaned", width="stretch", config=CHART_CONFIG)
        with maternal_col:
            st.markdown("#### 2 · Maternal-pattern estimate")
            st.plotly_chart(ecg_chart(local_time, result.maternal_component[start_index:end_index], COLORS["coral"]), key="studio_maternal", width="stretch", config=CHART_CONFIG)
        st.markdown("#### 3 · Residual candidate signal")
        st.plotly_chart(ecg_chart(local_time, result.residual[start_index:end_index], COLORS["violet"], local_fetal_peaks), key="studio_residual", width="stretch", config=CHART_CONFIG)

        st.markdown("### Reproducible exports")
        st.caption("Every export states or preserves what was analyzed. Keep the research-only interpretation with shared results.")
        download_one, download_two, download_three, download_four = st.columns(4)
        download_one.download_button("Processed waveform", make_waveform_export(time, raw_signal, result), "fetalsignal_processed.csv", "text/csv", width="stretch")
        download_two.download_button("Candidate beats", make_annotation_export(time, fetal_peaks), "fetalsignal_candidate_beats.csv", "text/csv", width="stretch")
        download_three.download_button("Readable report", make_report(data_note, sample_rate, fetal_bpm, maternal_bpm, result, fetal_peaks, analysis_method), "fetalsignal_report.txt", "text/plain", width="stretch")
        download_four.download_button("Analysis manifest", make_manifest(data_note, sample_rate, fetal_bpm, maternal_bpm, result.quality_score, fetal_peaks.size, analysis_method), "fetalsignal_manifest.json", "application/json", width="stretch")

    with validation_tab:
        st.markdown("### Check candidate beats against a trusted reference")
        st.caption("A candidate matches a reference when it falls within a pre-declared 80 ms window. Matching is one-to-one.")
        if reference is not None and reference.size:
            validation = match_peaks(fetal_peaks, reference, sample_rate)
            reference_bpm = heart_rate_bpm(reference, sample_rate)
            rate_error = abs(fetal_bpm - reference_bpm) if fetal_bpm is not None and reference_bpm is not None else None
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Precision", f"{validation['precision']:.1%}", help="Of the predicted beats, the share that matched a reference.")
            metric_two.metric("Recall", f"{validation['recall']:.1%}", help="Of the reference beats, the share the algorithm found.")
            metric_three.metric("F1 score", f"{validation['f1']:.1%}", help="Harmonic mean of precision and recall.")
            metric_four.metric("Rate error", "—" if rate_error is None else f"{rate_error:.1f} BPM", help="Absolute difference between candidate and reference median-interval rates.")
            st.plotly_chart(validation_timeline(fetal_peaks, reference, sample_rate), key="validation_timeline", width="stretch", config=CHART_CONFIG)
            st.markdown("**How to read this:** aligned marks indicate agreement in time. Precision reflects extra candidate detections; recall reflects missed reference beats. A small BPM error alone does not prove beat-by-beat agreement.")
            if source in {"Use sample recording", "Try the interactive demo"}:
                st.info("These metrics belong to a labelled synthetic example. They demonstrate the validation workflow and are not evidence of real-world performance.")
        else:
            if reference_error:
                st.warning(f"Validation unavailable: {reference_error}")
            else:
                st.info("No independent reference beats are attached. Analysis remains available, but per-recording performance cannot be calculated.")
            insight_card("What credible validation requires", "Keep references separate from input signals, declare the matching tolerance before testing, document preprocessing, and evaluate recordings that were not used to tune the pipeline.")

    with evidence_tab:
        st.markdown("### Results from separate public-data tests")
        evaluation = load_evaluation_report()
        if evaluation is None:
            st.info("The bundled evaluation report is unavailable in this deployment.")
        else:
            st.markdown("These results come from five public, de-identified PhysioNet ADFECGDB recordings and a separate multi-lead consensus evaluation. They do not predict performance for the current recording.")
            evidence_one, evidence_two, evidence_three, evidence_four = st.columns(4)
            evidence_one.metric("Macro F1", f"{evaluation['mean_multilead_consensus_f1']:.2%}")
            evidence_two.metric("Precision", f"{evaluation['mean_multilead_consensus_precision']:.2%}")
            evidence_three.metric("Recall", f"{evaluation['mean_multilead_consensus_recall']:.2%}")
            evidence_four.metric("Rate error", f"{evaluation['mean_multilead_consensus_bpm_absolute_error']:.2f} BPM")
            records = evaluation.get("records", [])
            st.plotly_chart(evidence_chart(records), key="evidence_records", width="stretch", config=CHART_CONFIG)
            st.caption(f"{evaluation['dataset']} · {evaluation['license']} · DOI {evaluation['dataset_doi']}")
            rows = []
            for record in records:
                metrics = record.get("multilead_consensus", {})
                rows.append({
                    "Held-out record": record.get("held_out_record", "—"),
                    "F1": f"{metrics.get('f1', 0):.2%}",
                    "Precision": f"{metrics.get('precision', 0):.2%}",
                    "Recall": f"{metrics.get('recall', 0):.2%}",
                    "Rate error": f"{metrics.get('bpm_absolute_error', 0):.2f} BPM",
                })
            with st.expander("See record-level evaluation table"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.warning(evaluation.get("limitation", "Exploratory research results are not clinical-performance claims."))

    with method_tab:
        st.markdown("### Transparent by design")
        pipeline()
        method_left, method_right = st.columns(2)
        with method_left:
            st.markdown("#### What the AI does")
            st.markdown("The bundled supervised model **ranks signal-processing candidates** using local waveform and timing features. It does not diagnose a condition, generate patient advice, or retrain on uploads. If it would remove too many baseline candidates, the app falls back to the baseline detector.")
        with method_right:
            st.markdown("#### What multi-lead consensus does")
            st.markdown("With at least three aligned abdominal leads, the pipeline retains candidates supported by multiple channels within 80 ms. This can reject single-channel artifacts, but it still requires independent evaluation.")
        st.warning("Known limitations include electrode placement, movement, maternal rhythm, gestational age, recording hardware, dataset shift, and the small public evaluation cohort. A plausible candidate signal can still be wrong.")

with st.expander("Questions, file requirements, and troubleshooting"):
    st.markdown("""
**What can I upload?** A UTF-8 CSV/TXT table with headers, one uniformly sampled row per time point, and at least three seconds of numeric ECG data. Select abdominal leads only.

**Why does sampling rate matter?** It converts samples into seconds. A wrong rate produces wrong timing and BPM. Recognized time columns are checked against the selected rate.

**Why did the app reject missing rows?** Dropping rows would shift beat timing and misalign leads. The source recording must be repaired instead.

**Why are candidate BPM and quality not a diagnosis?** They summarize algorithmic beat spacing and signal characteristics. They do not establish fetal health, clinical accuracy, or risk.

**Does an upload train the model?** No. The bundled model is fixed; uploads are not used for retraining.

**What is the difference between Check accuracy and Public testing?** Check accuracy evaluates the current recording only when matching references are supplied. Public testing reports a separate five-record public-data experiment.
""")

footer()
