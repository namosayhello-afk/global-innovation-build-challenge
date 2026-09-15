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

The app also includes a small supervised candidate-ranking model trained by `scripts/train_candidate_ranker.py`. It ranks signal-processing candidates; it is an experimental research component, not a diagnosis model. The training script uses public ADFECGDB data and is designed to accept more verified records as they are added.

The default recording is synthetic and has labelled fetal beat times so the evaluation panel can be demonstrated safely. Its results are deliberately labelled **demo-only** and must never be presented as real performance.

## Run it

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To recreate the experimental training model after downloading the public ADFECGDB record(s):

```bash
pip install pyedflib wfdb
python -m scripts.train_candidate_ranker --records data/adfecgdb/r01.edf
```

## Two-minute judge demo

1. Open the app and leave **Try the interactive demo** selected.
2. Point out the raw abdominal mixture and the fetal cardiac-signal candidate in **Overview**.
3. Move the recording-window slider and open **Signal lab** to show each separation layer.
4. Open **Validation** to explain why performance must be measured against separate reference annotations.
5. Switch the demo noise control to **High** to demonstrate that the prototype exposes challenging signal conditions rather than hiding them.
6. Download the waveform, candidate-beat annotations, or analysis report to show reproducibility.

## Test files included

Use these two synthetic files to test the upload flow without downloading any data:

- `sample_data/fetalsignal_sample_recording.csv` — upload this as the abdominal ECG waveform, choose `abdominal_ecg`, and leave the sampling rate at 500 Hz.
- `sample_data/fetalsignal_sample_reference_beats.csv` — optionally upload this as the reference-beat file to populate the validation tab.

They are synthetic demonstration data only, not a performance claim or patient data.

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
fetalsignal/demo_data.py          Synthetic labelled demo generator
fetalsignal/signal_processing.py  Explainable signal-processing baseline
requirements.txt          Dashboard dependencies
```

## AI-use disclosure

> ChatGPT and OpenAI Codex were used as coding and learning assistants for implementation guidance, debugging, signal-processing explanations, and documentation. All project decisions, evaluation results, experiments, and submitted code were reviewed and understood by the team.

## Before submitting

- [ ] Run the pipeline against at least one documented public, de-identified dataset.
- [ ] Record real precision, recall, F1, and fetal-rate error values—do not use the synthetic demo results as project performance.
- [ ] Cite the dataset version and its licence/terms in the submission.
- [ ] Include a short screen recording or screenshots of the app and its signal-separation flow.
- [ ] Keep the research-only / non-medical-device disclaimer in the presentation and submission.
- [ ] Review and understand every reported result and implementation choice.
