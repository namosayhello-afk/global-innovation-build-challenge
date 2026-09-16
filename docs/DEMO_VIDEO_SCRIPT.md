# FetalSignal AI — 2 minute 30 second demo video

Record a screen capture of the deployed app with your voice **or** these English captions. A face camera is optional; the important part is showing the working system and explaining the approach. Do not call a candidate result a diagnosis.

| Time | Show on screen | Say or caption |
| --- | --- | --- |
| 0:00–0:12 | App home and research-only safety statement | “FetalSignal AI is an explainable research dashboard for exploring fetal cardiac-signal candidates inside mixed abdominal ECG recordings.” |
| 0:12–0:27 | Three-step Input → Separate → Explore cards | “It makes the workflow visible: bring a de-identified waveform, reduce the dominant maternal pattern, then inspect and validate the result.” |
| 0:27–0:45 | Click **Upload an ECG recording**; upload `sample_data/fetalsignal_sample_recording.csv` | “The app accepts CSV or TXT waveform samples, not an image of an ECG chart, because original signal values are needed for reproducible processing.” |
| 0:45–1:00 | Select `abdominal_ecg`, leave 500 Hz, upload `fetalsignal_sample_reference_beats.csv` | “A separate reference-beat file is optional. Keeping it separate lets us measure performance instead of guessing.” |
| 1:00–1:17 | Overview tab; move the time-window slider | “The overview compares the abdominal mixture with the residual fetal cardiac-signal candidate. Markers are algorithmic candidates, not confirmed clinical beats.” |
| 1:17–1:32 | Signal lab | “The Signal lab exposes every layer: the filtered mixture, the maternal-pattern estimate, and the residual. You can download the processed waveform, candidate beats, and analysis report.” |
| 1:32–1:45 | Validation tab | “With references, the app reports precision, recall, F1, and candidate-rate error. That makes the evaluation transparent and repeatable.” |
| 1:45–2:05 | Evidence tab; point to macro metrics and record table | “For real evidence, we evaluated multi-lead consensus on five public, de-identified PhysioNet records. The held-out-record macro F1 is 95.77 percent, with 0.60 BPM candidate-rate error. These are exploratory research results, not clinical performance claims.” |
| 2:05–2:20 | Method tab; point to multi-lead and limitations bullets | “The pipeline filters the signal, estimates and reduces the maternal template, finds residual candidates, and—with multiple leads—keeps only aligned candidates supported by several leads.” |
| 2:20–2:30 | Return to the safety statement / Evidence limitation | “FetalSignal AI uses only public, de-identified or synthetic data. It is a research prototype, not a medical device, and it must not be used for patient decisions.” |

Before recording, run the app once, increase browser zoom only if text becomes too small, and use a clean browser window. Keep the video between 2 and 5 minutes, as required by the competition.
