# FetalSignal AI

FetalSignal AI is a research prototype for exploring whether a fetal cardiac signal can be recovered from a mixed abdominal ECG recording. It is built for the Global Innovation Build Challenge V2 medical technology track.

> **Safety statement:** This is not a medical device and must not be used for diagnosis, treatment, patient monitoring, or real-world medical decisions. It is designed only for experiments on public, de-identified research data.

## What it does

The dashboard accepts an abdominal ECG waveform (CSV or TXT) and implements an explainable baseline pipeline:

1. Filter baseline drift and electrical interference from an abdominal ECG signal.
2. Detect maternal candidate R-peaks.
3. Build and subtract a median maternal beat template.
4. Filter the residual and detect fetal candidate peaks.
5. Estimate fetal heart rate and report a signal-quality heuristic.
6. Optionally compare predicted candidate beats against a separate reference-annotation file.
7. Export the transformed signals for reproducible analysis.

The default recording is synthetic and has labelled fetal beat times so the evaluation panel can be demonstrated safely. Its results are deliberately labelled **demo-only** and must never be presented as real performance.

## Run it

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Using a public dataset

Use public, de-identified data such as PhysioNet fetal ECG collections. Convert one abdominal ECG lead into a CSV or TXT table with one sample per row and upload it through the app. Enter the recording's actual sampling frequency. If the table includes a `time`, `timestamp`, `time_s`, `seconds`, or `t` column, the dashboard uses it for the x-axis.

This app deliberately does **not** take a screenshot or photograph of an ECG chart as input: an image loses the waveform's original sample values and cannot support reliable signal processing.

For final evaluation, keep fetal reference annotations separate from the input signal. Compare predicted peaks against those references using a fixed matching tolerance, then report precision, recall, F1, and fetal heart-rate MAE from the actual experiment. Record the dataset version, provenance, licence/terms, preprocessing settings, and random seed in your submission.

Useful starting points:

- [Abdominal and Direct Fetal ECG Database](https://physionet.org/content/adfecgdb/)
- [Non-Invasive Fetal ECG Database](https://physionet.org/content/nifecgdb/)

## Repository layout

```
app.py                    Streamlit dashboard
src/demo_data.py          Synthetic labelled demo generator
src/signal_processing.py  Explainable signal-processing baseline
requirements.txt          Dashboard dependencies
```

## AI-use disclosure

> ChatGPT and OpenAI Codex were used as coding and learning assistants for implementation guidance, debugging, signal-processing explanations, and documentation. All project decisions, evaluation results, experiments, and submitted code were reviewed and understood by the team.
