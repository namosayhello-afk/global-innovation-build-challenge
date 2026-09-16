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

## Data ethics and provenance

FetalSignal AI accepts and uses only public, properly licensed, or fully de-identified research data. It must never be used with identifiable patient data, personal health records, or recordings without documented permission to use them. Uploaded files are processed only for the current app session and are not written into this repository.

| Data source | Version / licence | Use in this project | Citation |
| --- | --- | --- | --- |
| Abdominal and Direct Fetal ECG Database (ADFECDGB), PhysioNet | Version 1.0.0; Open Data Commons Attribution License v1.0; DOI: [10.13026/C2RP4B](https://doi.org/10.13026/C2RP4B) | The experimental candidate-ranking model and five-record evaluation use the public, de-identified `r01`, `r04`, `r07`, `r08`, and `r10` records. Their direct fetal channels and verified QRS annotations are used only as training/validation references, never as app input. | Jezewski J, Matonia A, Kupka T, Roj D, Czabanski R. *Determination of the fetal heart rate from abdominal signals: evaluation of beat-to-beat accuracy in relation to the direct fetal electrocardiogram.* Biomedical Engineering/Biomedizinische Technik. 2012;57(5):383–394. |
| `sample_data/` files in this repository | Synthetic data generated locally by `fetalsignal/demo_data.py`; no human data or external licence required | Upload-flow demonstration only. They are never reported as model performance. | Not applicable. |

The source ADFECGDB data are intentionally excluded from Git via `.gitignore`; only code, a small derived research-model artifact, and synthetic demo files are in the public repository. Before adding any new dataset, record its source URL, exact version, licence/terms, de-identification status, permitted use, and the experiment that uses it in this table.

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

To run the current record-level evaluation and regenerate the bundled model:

```bash
pip install -r requirements-train.txt
python -m scripts.evaluate_candidate_ranker --records data/adfecgdb/r01.edf data/adfecgdb/r04.edf data/adfecgdb/r07.edf data/adfecgdb/r08.edf data/adfecgdb/r10.edf
```

## Current research evaluation

The current reproducible evaluation is saved in [`results/adfecgdb_leave_one_record_out.json`](results/adfecgdb_leave_one_record_out.json). The ML candidate-ranker uses leave-one-record-out testing: train on four public ADFECGDB records, then evaluate on the fifth. The multi-lead consensus pipeline is evaluated separately on each complete four-abdominal-lead recording. A predicted candidate peak counts as matched when it falls within 80 ms of a verified reference fetal QRS annotation.

| Public record | Multi-lead consensus F1 | Candidate-rate absolute error |
| --- | ---: | ---: |
| `r01.edf` | 99.07% | 0.00 BPM |
| `r04.edf` | 93.47% | 1.87 BPM |
| `r07.edf` | 94.32% | 0.00 BPM |
| `r08.edf` | 98.69% | 0.28 BPM |
| `r10.edf` | 93.29% | 0.87 BPM |
| Macro mean, 5 records | 95.77% | 0.60 BPM |

For comparison, the single-lead ML-assisted candidate-ranker reaches a 82.33% macro F1 and 11.60 BPM macro candidate-rate error on the same five-record leave-one-record-out evaluation. The stronger multi-lead result is expected: it keeps a candidate only when at least three aligned abdominal leads agree. These are real exploratory results from five public, de-identified records—not clinical performance claims. The dataset remains small and drawn from women in labor, so broader validation is still needed.

The live dashboard's **Evidence** tab reads this same version-controlled evaluation file and displays the record-level results and limitations. It never treats those cohort-level research numbers as a prediction for a new upload.

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

- [Abdominal and Direct Fetal ECG Database v1.0.0 — public/de-identified, ODC-By 1.0](https://www.physionet.org/content/adfecgdb/1.0.0/)
- [Non-Invasive Fetal ECG Database](https://physionet.org/content/nifecgdb/)

## Repository layout

```
app.py                    Streamlit dashboard
fetalsignal/demo_data.py          Synthetic labelled demo generator
fetalsignal/signal_processing.py  Explainable signal-processing baseline
requirements.txt          Dashboard dependencies
docs/                     Ready-to-use Devpost, demo-video, screenshot, and submission materials
```

## AI-use disclosure

> **AI tools used:** ChatGPT and OpenAI Codex. They assisted with the Streamlit interface, signal-processing implementation and debugging, experimental-model workflow, test/sample-data scaffolding, signal-processing explanations, and documentation. All project decisions, public-data experiments, reported results, and submitted code were reviewed and understood by the team. The project is original work created during the July 11–October 1, 2026 hackathon period; no pre-existing product or prior hackathon submission was used.

## Submission materials

The repository includes polished materials to paste into the competition submission and use while recording:

- [`docs/DEVPOST_SUBMISSION.md`](docs/DEVPOST_SUBMISSION.md) — project copy, technology list, ethics statement, and AI-use disclosure.
- [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md) — a 2–3 minute screen-recording script with voiceover/caption text.
- [`docs/SCREENSHOT_GUIDE.md`](docs/SCREENSHOT_GUIDE.md) — three required screenshot shots and captions.
- [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md) — requirement-by-requirement final handoff checklist.

## Before submitting

- [x] Run the pipeline against a documented public, de-identified dataset.
- [x] Record real precision, recall, F1, and fetal-rate error values from five public held-out-record evaluations—do not use the synthetic demo results as project performance.
- [ ] Expand validation beyond five records and report all record-level results, including difficult cases.
- [ ] Cite the dataset version and its licence/terms in the submission.
- [ ] Confirm every dataset is public or fully de-identified and that the README data-provenance table is current.
- [ ] Include a short screen recording or screenshots of the app and its signal-separation flow.
- [ ] Keep the research-only / non-medical-device disclaimer in the presentation and submission.
- [ ] Review and understand every reported result and implementation choice.
