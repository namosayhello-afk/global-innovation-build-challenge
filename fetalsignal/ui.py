"""Presentation helpers for the FetalSignal AI research workspace."""

from __future__ import annotations

from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "bg": "#071018",
    "panel": "#0d1b27",
    "panel_2": "#112331",
    "text": "#f3f8fb",
    "muted": "#95a9b8",
    "mint": "#62e6bd",
    "cyan": "#68c8ff",
    "violet": "#a99bff",
    "coral": "#ff9f91",
    "gold": "#ffd47a",
}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fs-bg:#071018; --fs-panel:#0d1b27; --fs-panel-2:#112331;
            --fs-text:#f3f8fb; --fs-muted:#95a9b8; --fs-mint:#62e6bd;
            --fs-cyan:#68c8ff; --fs-violet:#a99bff; --fs-coral:#ff9f91;
            --fs-line:rgba(180,213,229,.14);
        }
        .stApp {
            background:
                radial-gradient(circle at 82% -8%, rgba(88,128,179,.22), transparent 29rem),
                radial-gradient(circle at 2% 28%, rgba(75,205,174,.09), transparent 25rem),
                var(--fs-bg);
            color:var(--fs-text);
        }
        [data-testid="stHeader"] { background:rgba(7,16,24,.72); backdrop-filter:blur(16px); }
        [data-testid="stSidebar"] { background:#09151f; border-right:1px solid var(--fs-line); }
        [data-testid="stSidebarContent"] { padding-top:1.15rem; }
        .block-container { max-width:1440px; padding-top:1.15rem; padding-bottom:4rem; }
        h1,h2,h3,h4 { letter-spacing:-.035em !important; }
        h1 { font-size:clamp(2.35rem,4.5vw,4.25rem) !important; line-height:.98 !important; }
        h2 { font-size:clamp(1.65rem,3vw,2.45rem) !important; }
        h3 { font-size:1.22rem !important; }
        p,li { line-height:1.62; }
        .fs-brand { display:flex; align-items:center; gap:.7rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; font-size:.72rem; color:#d9fdf3; }
        .fs-mark { width:2rem; height:2rem; border-radius:.72rem; display:grid; place-items:center; color:#071018; background:linear-gradient(135deg,var(--fs-mint),var(--fs-violet)); box-shadow:0 0 26px rgba(98,230,189,.22); }
        .fs-hero { position:relative; overflow:hidden; border:1px solid var(--fs-line); border-radius:24px; padding:clamp(1.35rem,3vw,2.4rem); margin:.55rem 0 1rem; background:linear-gradient(145deg,rgba(16,36,51,.96),rgba(8,19,29,.94)); }
        .fs-hero:after { content:""; position:absolute; width:28rem; height:28rem; border-radius:50%; right:-12rem; top:-16rem; background:radial-gradient(circle,rgba(98,230,189,.22),rgba(169,155,255,.08) 46%,transparent 70%); pointer-events:none; }
        .fs-eyebrow { color:var(--fs-mint); font-weight:750; letter-spacing:.13em; text-transform:uppercase; font-size:.72rem; margin-bottom:.7rem; }
        .fs-title { max-width:870px; margin:0; font-size:clamp(2.35rem,4.5vw,4.25rem); line-height:1; letter-spacing:-.055em; font-weight:800; }
        .fs-gradient { background:linear-gradient(90deg,#f4fbff 0%,#91f2d7 42%,#b7aaff 100%); -webkit-background-clip:text; color:transparent; }
        .fs-subtitle { max-width:780px; color:#b8c9d4; font-size:1rem; line-height:1.6; margin:1rem 0 1.15rem; }
        .fs-badges { display:flex; flex-wrap:wrap; gap:.55rem; }
        .fs-badge { border:1px solid var(--fs-line); border-radius:999px; padding:.45rem .7rem; color:#bad0dc; background:rgba(255,255,255,.035); font-size:.74rem; }
        .fs-badge strong { color:#effffc; }
        .fs-alert { border:1px solid rgba(98,230,189,.24); background:rgba(98,230,189,.065); border-radius:14px; padding:.82rem 1rem; color:#c6e8df; font-size:.84rem; }
        .fs-alert.warning { border-color:rgba(255,212,122,.28); background:rgba(255,212,122,.07); color:#f3dfb4; }
        .fs-section-kicker { color:var(--fs-mint); font-size:.69rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:1.8rem 0 .25rem; }
        .fs-section-title { font-size:clamp(1.55rem,2.4vw,2.2rem); font-weight:780; letter-spacing:-.04em; margin:0 0 .35rem; }
        .fs-section-copy { color:var(--fs-muted); max-width:800px; margin-bottom:1.1rem; }
        .fs-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; margin:.9rem 0 1.2rem; }
        .fs-card { border:1px solid var(--fs-line); background:linear-gradient(145deg,rgba(17,35,49,.9),rgba(10,23,34,.9)); border-radius:17px; padding:1rem 1.05rem; min-height:126px; }
        .fs-card-number { display:inline-grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:.55rem; background:rgba(98,230,189,.12); color:var(--fs-mint); font-weight:800; font-size:.75rem; }
        .fs-card h4 { margin:.72rem 0 .3rem; font-size:.98rem; }
        .fs-card p { color:var(--fs-muted); font-size:.8rem; margin:0; line-height:1.48; }
        .fs-result-banner { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; border:1px solid var(--fs-line); background:linear-gradient(110deg,rgba(98,230,189,.07),rgba(169,155,255,.06)); border-radius:18px; padding:1.05rem 1.15rem; margin:.35rem 0 1rem; }
        .fs-result-banner h3 { margin:.12rem 0 .28rem; }
        .fs-result-banner p { margin:0; color:var(--fs-muted); font-size:.83rem; max-width:850px; }
        .fs-status { white-space:nowrap; border-radius:999px; padding:.48rem .72rem; font-size:.72rem; font-weight:800; }
        .fs-status.good { color:var(--fs-mint); background:rgba(98,230,189,.12); border:1px solid rgba(98,230,189,.28); }
        .fs-status.watch { color:#ffd47a; background:rgba(255,212,122,.1); border:1px solid rgba(255,212,122,.24); }
        .fs-status.low { color:#ffaaa0; background:rgba(255,159,145,.1); border:1px solid rgba(255,159,145,.24); }
        .fs-insight { height:100%; box-sizing:border-box; border:1px solid var(--fs-line); background:rgba(255,255,255,.028); border-radius:16px; padding:1rem 1.05rem; }
        .fs-insight b { display:block; margin-bottom:.38rem; }
        .fs-insight p { margin:0; color:var(--fs-muted); font-size:.82rem; line-height:1.55; }
        .fs-pipeline { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.55rem; margin:.9rem 0; }
        .fs-pipe { position:relative; border:1px solid var(--fs-line); background:rgba(255,255,255,.025); border-radius:14px; padding:.9rem; min-height:112px; }
        .fs-pipe:not(:last-child):after { content:"→"; position:absolute; right:-.48rem; top:42%; color:var(--fs-mint); z-index:2; font-weight:800; }
        .fs-pipe small { color:var(--fs-mint); font-weight:800; letter-spacing:.08em; }
        .fs-pipe b { display:block; margin:.45rem 0 .22rem; font-size:.84rem; }
        .fs-pipe span { color:var(--fs-muted); font-size:.73rem; line-height:1.4; }
        [data-testid="stMetric"] { border:1px solid var(--fs-line); border-radius:16px; background:linear-gradient(145deg,rgba(17,35,49,.9),rgba(10,23,34,.9)); padding:1rem 1.05rem; }
        [data-testid="stMetricLabel"] { color:#9bb0bd; font-size:.77rem; }
        [data-testid="stMetricValue"] { color:#f5fbff; font-size:1.65rem; font-weight:780; }
        [data-testid="stMetricDelta"] { font-size:.7rem; }
        .stTabs [data-baseweb="tab-list"] { gap:.25rem; overflow-x:auto; border-bottom:1px solid var(--fs-line); }
        .stTabs [data-baseweb="tab"] { height:3rem; padding:0 1rem; border-radius:10px 10px 0 0; }
        .stTabs [aria-selected="true"] { color:var(--fs-mint) !important; background:rgba(98,230,189,.06); }
        .stButton>button,.stDownloadButton>button { min-height:2.7rem; border-radius:11px; border:1px solid rgba(98,230,189,.38); background:linear-gradient(135deg,#75ebc9,#ada0ff); color:#071018; font-weight:800; }
        .stButton>button:hover,.stDownloadButton>button:hover { border-color:#fff; color:#071018; filter:brightness(1.06); }
        div[data-baseweb="radio"]>div { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; }
        div[data-baseweb="radio"] label { min-height:3.15rem; align-items:center; border:1px solid var(--fs-line); border-radius:12px; padding:.65rem .8rem; background:rgba(255,255,255,.025); transition:border-color .15s ease,background .15s ease; }
        div[data-baseweb="radio"] label:hover { border-color:rgba(98,230,189,.48); background:rgba(98,230,189,.055); }
        .stAlert { border-radius:13px; }
        [data-testid="stExpander"] { border-color:var(--fs-line); border-radius:14px; overflow:hidden; }
        [data-testid="stDataFrame"] { border:1px solid var(--fs-line); border-radius:14px; overflow:hidden; }
        .fs-footer { border-top:1px solid var(--fs-line); padding-top:1.2rem; margin-top:2.5rem; color:#7f96a5; font-size:.75rem; display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
        @media(max-width:900px) { .fs-grid{grid-template-columns:1fr;} .fs-pipeline{grid-template-columns:1fr;} .fs-pipe:not(:last-child):after{content:"↓";right:50%;top:auto;bottom:-.65rem;} .fs-result-banner{flex-direction:column;} .fs-status{white-space:normal;} div[data-baseweb="radio"]>div{grid-template-columns:1fr;} }
        @media(max-width:700px) { .block-container{padding:.75rem .85rem 3rem;} .fs-hero{border-radius:18px;padding:1rem;} h1{font-size:2.15rem !important;} .fs-subtitle{font-size:.9rem;line-height:1.5;margin:.75rem 0 .2rem;} .fs-badges{display:none;} .fs-alert{font-size:.76rem;padding:.72rem .8rem;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown("<div class='fs-brand'><span class='fs-mark'>✦</span> FetalSignal AI</div>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <section class="fs-hero">
          <div class="fs-eyebrow">Explainable fetal-ECG research workspace</div>
          <h1 class="fs-title">Find the faint signal.<br><span class="fs-gradient">Show every step.</span></h1>
          <p class="fs-subtitle">Explore how a possible fetal heartbeat can be separated from mixed abdominal ECG. See the steps, check candidate beats, and export the result.</p>
          <div class="fs-badges">
            <span class="fs-badge"><strong>Transparent</strong> signal pipeline</span>
            <span class="fs-badge"><strong>Independent</strong> validation</span>
            <span class="fs-badge"><strong>Public-data</strong> evidence</span>
            <span class="fs-badge"><strong>No patient data</strong> in the demo</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def safety_note(compact: bool = False) -> None:
    detail = "" if compact else " It cannot be used for diagnosis, treatment, monitoring, or patient decisions."
    st.markdown(
        f"<div class='fs-alert warning'><strong>Research prototype.</strong> This workspace analyzes public, de-identified, or synthetic data only.{detail}</div>",
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str = "") -> None:
    st.markdown(f"<div class='fs-section-kicker'>{escape(kicker)}</div><div class='fs-section-title'>{escape(title)}</div>", unsafe_allow_html=True)
    if copy:
        st.markdown(f"<div class='fs-section-copy'>{escape(copy)}</div>", unsafe_allow_html=True)


def onboarding_cards() -> None:
    st.markdown(
        """
        <div class="fs-grid">
          <div class="fs-card"><span class="fs-card-number">01</span><h4>Choose a safe recording</h4><p>Use the labelled synthetic sample, tune the interactive demo, or upload a permitted CSV/TXT waveform.</p></div>
          <div class="fs-card"><span class="fs-card-number">02</span><h4>Inspect the separation</h4><p>See the mixed ECG, maternal-pattern estimate, residual signal, candidate beats, and rhythm stability.</p></div>
          <div class="fs-card"><span class="fs-card-number">03</span><h4>Validate before claiming</h4><p>Compare with independent reference annotations and keep public-dataset evidence separate from a new upload.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(title: str, text: str, status: str, status_class: str) -> None:
    st.markdown(
        f"<div class='fs-result-banner'><div><div class='fs-eyebrow'>Analysis snapshot</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div><span class='fs-status {status_class}'>{escape(status)}</span></div>",
        unsafe_allow_html=True,
    )


def insight_card(title: str, text: str) -> None:
    st.markdown(f"<div class='fs-insight'><b>{escape(title)}</b><p>{escape(text)}</p></div>", unsafe_allow_html=True)


def pipeline() -> None:
    steps = [
        ("01", "Clean", "Reduce baseline drift, power-line interference, and out-of-band content."),
        ("02", "Model maternal pattern", "Find recurring stronger peaks and build a median beat template."),
        ("03", "Separate", "Subtract the overlap-adjusted maternal estimate from the cleaned signal."),
        ("04", "Rank candidates", "Find residual peaks and optionally apply the experimental ML ranker."),
        ("05", "Validate", "Compare against independent beat annotations with a fixed tolerance."),
    ]
    html = "".join(f"<div class='fs-pipe'><small>{n}</small><b>{escape(title)}</b><span>{escape(text)}</span></div>" for n, title, text in steps)
    st.markdown(f"<div class='fs-pipeline'>{html}</div>", unsafe_allow_html=True)


def chart_layout(height: int, y_title: str = "Amplitude") -> dict:
    return dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=18, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.018)",
        font=dict(color="#b9cad5", size=11),
        xaxis=dict(title="Time (seconds)", gridcolor="rgba(180,213,229,.08)", zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(title=y_title, gridcolor="rgba(180,213,229,.08)", zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", y=-.23, x=0, font=dict(size=10)),
    )


def ecg_chart(time: np.ndarray, signal: np.ndarray, color: str, peaks: np.ndarray | None = None, peak_name: str = "Candidate beat") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scattergl(x=time, y=signal, mode="lines", name="Signal", line=dict(color=color, width=1.35), hovertemplate="%{x:.3f} s<br>%{y:.4f} a.u.<extra></extra>"))
    if peaks is not None and peaks.size:
        figure.add_trace(go.Scatter(x=time[peaks], y=signal[peaks], mode="markers", name=peak_name, marker=dict(color=COLORS["gold"], size=7, line=dict(color="#071018", width=1)), hovertemplate="%{x:.3f} s<extra></extra>"))
    figure.update_layout(**chart_layout(285))
    return figure


def rhythm_chart(centers: np.ndarray, rates: np.ndarray) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=centers, y=rates, mode="lines+markers", line=dict(color=COLORS["mint"], width=2.6, shape="spline"), marker=dict(color=COLORS["violet"], size=7), connectgaps=False, fill="tozeroy", fillcolor="rgba(98,230,189,.05)", hovertemplate="%{x:.1f} s<br>%{y:.1f} BPM<extra></extra>"))
    layout = chart_layout(245, "Candidate BPM")
    layout["showlegend"] = False
    layout["yaxis"]["rangemode"] = "tozero"
    figure.update_layout(**layout)
    return figure


def validation_timeline(predicted: np.ndarray, reference: np.ndarray, sample_rate: float) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=reference / sample_rate, y=np.ones(reference.size), mode="markers", name="Reference beat", marker=dict(symbol="line-ns", size=18, color=COLORS["cyan"], line=dict(width=2)), hovertemplate="Reference · %{x:.3f} s<extra></extra>"))
    figure.add_trace(go.Scatter(x=predicted / sample_rate, y=np.zeros(predicted.size), mode="markers", name="Candidate beat", marker=dict(symbol="diamond", size=8, color=COLORS["gold"]), hovertemplate="Candidate · %{x:.3f} s<extra></extra>"))
    figure.update_layout(**chart_layout(230, ""))
    figure.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Candidate", "Reference"], range=[-.65, 1.65])
    return figure


def quality_diagnostics(cleaned: np.ndarray, residual: np.ndarray, peaks: np.ndarray, sample_rate: float) -> dict[str, float]:
    if peaks.size >= 3:
        intervals = np.diff(peaks) / sample_rate
        valid = intervals[(intervals >= 60 / 220) & (intervals <= 60 / 90)]
        interval_plausibility = float(np.mean((intervals >= 60 / 220) & (intervals <= 60 / 90)))
        regularity = float(max(0.0, 1 - np.std(valid) / (np.mean(valid) + 1e-12))) if valid.size else 0.0
    else:
        interval_plausibility = regularity = 0.0
    residual_ratio = float(np.std(residual) / (np.std(cleaned) + 1e-12))
    return {
        "Plausible spacing": np.clip(interval_plausibility * 100, 0, 100),
        "Rhythm regularity": np.clip(regularity * 100, 0, 100),
        "Residual energy": np.clip(residual_ratio / 0.30 * 100, 0, 100),
    }


def diagnostics_chart(values: dict[str, float]) -> go.Figure:
    names = list(values)
    scores = list(values.values())
    figure = go.Figure(go.Bar(x=scores, y=names, orientation="h", marker=dict(color=[COLORS["mint"], COLORS["violet"], COLORS["cyan"]], line=dict(width=0)), text=[f"{value:.0f}" for value in scores], textposition="inside", hovertemplate="%{y}: %{x:.0f}/100<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=220, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.018)", xaxis=dict(range=[0, 100], title="Heuristic component (0–100)", gridcolor="rgba(180,213,229,.08)"), yaxis=dict(autorange="reversed"), showlegend=False, font=dict(color="#b9cad5", size=11))
    return figure


def evidence_chart(records: list[dict]) -> go.Figure:
    names = [str(record.get("held_out_record", "—")).replace(".edf", "") for record in records]
    f1 = [float(record.get("multilead_consensus", {}).get("f1", 0)) * 100 for record in records]
    error = [float(record.get("multilead_consensus", {}).get("bpm_absolute_error", 0)) for record in records]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=names, y=f1, name="F1", marker_color=COLORS["mint"], text=[f"{value:.1f}%" for value in f1], textposition="outside", hovertemplate="%{x}<br>F1 %{y:.2f}%<extra></extra>"))
    figure.add_trace(go.Scatter(x=names, y=error, name="Rate error", yaxis="y2", mode="lines+markers", line=dict(color=COLORS["coral"], width=2), marker=dict(size=8), hovertemplate="%{x}<br>%{y:.2f} BPM error<extra></extra>"))
    figure.update_layout(template="plotly_dark", height=320, margin=dict(l=12, r=12, t=24, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.018)", font=dict(color="#b9cad5", size=11), xaxis=dict(title="Held-out PhysioNet record", gridcolor="rgba(180,213,229,.06)"), yaxis=dict(title="F1 (%)", range=[0, 108], gridcolor="rgba(180,213,229,.08)"), yaxis2=dict(title="Rate error (BPM)", overlaying="y", side="right", rangemode="tozero", showgrid=False), legend=dict(orientation="h", y=-.22))
    return figure


def footer() -> None:
    st.markdown("<div class='fs-footer'><span>FetalSignal AI · Explainable research prototype</span><span>Public, de-identified, or synthetic data only · Not a medical device</span></div>", unsafe_allow_html=True)
