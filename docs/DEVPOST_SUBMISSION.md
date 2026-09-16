# FetalSignal AI — Devpost submission copy

Use this as the starting point for the Devpost project page. Replace only the bracketed links and the team section with real information.

## Tagline

An explainable research dashboard that separates fetal cardiac-signal candidates from abdominal ECG mixtures and makes validation visible.

## Inspiration

Abdominal ECG contains a faint fetal cardiac contribution mixed with a much stronger maternal pattern, motion, and electrical noise. That makes it a compelling signal-processing challenge—but also a domain where a polished interface must not overstate what an algorithm knows. We built FetalSignal AI to make the full research workflow understandable: inspect the waveform, reduce the dominant maternal pattern, review candidate beats, and validate against independent references.

## What it does

FetalSignal AI accepts a de-identified abdominal ECG waveform in CSV or TXT form. It filters the recording, estimates and reduces the recurring maternal pattern, finds fetal cardiac-signal candidates in the residual, and displays candidate beats, candidate BPM, a quality heuristic, and downloadable analysis files. A separate validation input calculates precision, recall, F1, and candidate-rate error against reference beat annotations.

When a research file includes at least three aligned abdominal ECG leads, the dashboard can use a multi-lead consensus: a candidate beat is retained only when several leads agree within 80 ms. The dashboard also includes an experimental supervised candidate-ranker that augments the single-lead signal-processing baseline.

## How we built it

The dashboard is built in Python and Streamlit. The processing pipeline uses NumPy and SciPy for filtering, peak detection, maternal-template estimation, residual analysis, and multi-lead consensus. Pandas handles upload/export tables, Plotly provides interactive waveform exploration, and scikit-learn powers the experimental candidate-ranking model. The code, evaluation script, synthetic demo generator, and results are public and reproducible in this repository.

## Validation and evidence

We evaluated the multi-lead consensus pipeline on five held-out records from the public, de-identified PhysioNet Abdominal and Direct Fetal ECG Database (ADFECDGB), version 1.0.0. A predicted candidate peak was counted as a match if it was within 80 ms of a verified fetal-QRS annotation.

- Macro F1: **95.77%**
- Macro precision: **96.50%**
- Macro recall: **95.07%**
- Macro candidate-rate absolute error: **0.60 BPM**

These are exploratory research results from five public records, not clinical-performance claims. The dataset is small and consists of recordings from women in labor; broader validation is required before any generalization. The dashboard exposes this limitation directly in its Evidence tab and labels every output as research-only.

## Data ethics and permissions

The project uses only public, de-identified data and synthetic demonstration files. The evaluation data are from the PhysioNet Abdominal and Direct Fetal ECG Database (ADFECDGB), v1.0.0, under the Open Data Commons Attribution License v1.0, DOI [10.13026/C2RP4B](https://doi.org/10.13026/C2RP4B). The exact source, version, license, use, and citation are documented in the repository README. Raw public records are excluded from the repository; the app's included upload sample is synthetic and contains no patient data.

FetalSignal AI is a research and education prototype only. It is not a medical device and must not be used for diagnosis, treatment, patient monitoring, or clinical decision-making.

## Built with

Python, Streamlit, NumPy, SciPy, Pandas, Plotly, scikit-learn, joblib, pyEDFlib, WFDB, PhysioNet ADFECGDB, and GitHub.

## AI-use disclosure

ChatGPT and OpenAI Codex were used as coding and learning assistants for implementation guidance, debugging, signal-processing explanations, and documentation. The team reviewed and understood the submitted implementation, experiments, and reported results.

## Links to add before publishing

- Live app: [add deployed Streamlit URL]
- Public source repository: https://github.com/namosayhello-afk/global-innovation-build-challenge
- Demo video: [add unlisted video URL]
- Team members: [enter each member's legal name and Devpost account]
