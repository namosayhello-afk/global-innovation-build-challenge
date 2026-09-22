# FetalSignal AI — understand the app and explain your results

A learning guide for using the dashboard and preparing your demo.

## 1. What the project does

FetalSignal AI helps researchers look for possible fetal heartbeat patterns in an abdominal ECG recording.

An ECG is a recording of electrical activity associated with heartbeats. An abdominal recording can contain the pregnant person's heartbeat, a weaker fetal contribution, and unwanted interference. The app tries to reduce the stronger maternal pattern and find repeating peaks in the remaining signal.

Think of two drummers playing together: one is loud, one is quiet. The app tries to estimate the loud rhythm so the quieter rhythm becomes easier to inspect. The separation is imperfect, so we call the detected beats **candidates**, meaning possible beats.

**A sentence you can say:**

> “My app helps researchers explore possible fetal heartbeat patterns in a mixed ECG recording. It shows how the signal is processed and lets users compare the detected beats with reference data.”

## 2. Start with a complete example

1. Open **New here? Follow the guided walkthrough**.
2. Choose **Use sample recording** under **1. Choose your recording**.
3. The app loads a synthetic waveform, a sampling rate of 500 Hz, and matching reference beats.
4. Scroll to **2. Your results** and read the explanation.
5. Open the five tabs: Overview, Signal lab, Validation, Method, and Evidence.

The sample is artificially generated. There is no patient behind its numbers.

**Try the interactive demo** is another synthetic example with adjustable length and noise. **Upload signal data** lets you provide a permitted research recording.

## 3. What the upload settings mean

| Setting | Meaning | What to do |
| --- | --- | --- |
| Waveform file | A table of electrical-signal samples | Use CSV/TXT with headers and one row per sample. Photos, audio, and screenshots of BPM are not supported. |
| Sampling rate, Hz | How many signal measurements were recorded each second | Use the rate documented for the recording. The supplied sample uses 500 Hz. |
| Electrical interference, 50 or 60 Hz | The power-line frequency the filter attempts to reduce | Use the frequency appropriate to the source recording. This is different from heartbeat BPM. |
| ECG lead | One recording channel, usually from a particular electrode arrangement | Select abdominal ECG columns. Do not select time, direct fetal ECG, or reference annotations as abdominal inputs. |
| Reference-beat file | Known beat locations used to check predictions | Use annotations belonging to the exact waveform being analyzed. |
| Reference units | Whether annotations represent seconds or sample positions | Choose Seconds for times measured from recording start; choose sample indices for zero-based sample positions. |
| Reference-beat column | The column containing the annotations | Select beat locations, rather than a beat label or another measurement. |

**500 Hz does not mean 500 heartbeats per second.** It means 500 measurements of the waveform per second.

For a recorded interval of 250 samples, the elapsed time is 0.5 seconds at 500 Hz. That spacing corresponds to 120 BPM. If you incorrectly enter 1,000 Hz, the same sample spacing becomes 0.25 seconds. That illustrates why the wrong sampling rate distorts timing; the app may also reject intervals outside its internal limits.

The app checks recognized time columns against your chosen rate. Without a time column, you are responsible for using the correct rate.

## 4. What BPM means—and what high or low values tell you

**BPM means beats per minute.** It is a rate, not the number of beats found in the entire file.

The app estimates candidate BPM from the median interval between accepted candidate peaks. For evenly spaced beats:

| Spacing between beats | Corresponding rate |
| --- | ---: |
| 0.50 seconds | 120 BPM |
| Approximately 0.42 seconds | Approximately 144 BPM |
| Approximately 0.33 seconds | Approximately 180 BPM |

Closer spacing produces a higher rate. Wider spacing produces a lower rate.

### A higher candidate BPM

This means the accepted candidate peaks are closer together. Possible explanations include:

- A faster pattern in the source recording.
- Extra noise peaks being counted as beats.
- Parts of the maternal pattern remaining in the signal.
- A sampling-rate error.

**Say:** “The algorithm detected a faster candidate rhythm. I would inspect the peak markers and compare them with the reference annotations before interpreting it.”

### A lower candidate BPM

This means the accepted candidate peaks are farther apart. Possible explanations include:

- A slower pattern in the source recording.
- The algorithm missing some beats.
- The algorithm following the maternal pattern.
- A sampling-rate error.

**Say:** “The candidate rhythm is slower here. That could come from the recording or from missed or misidentified beats, so the result needs checking.”

### Does a value mean the fetus is healthy or in danger?

The app cannot determine that.

For clinical context only, NICE guidance for monitoring during labour describes a usual fetal baseline of 110–160 BPM. Clinicians determine a baseline over a 10-minute period, excluding accelerations and decelerations, and assess the wider trace and clinical situation. FetalSignal's median candidate interval and short demo are different measurements. Do not turn that clinical range into a green/red health classification for this app. [Source: NICE, Fetal monitoring in labour, recommendations 1.4.14–1.4.16](https://www.nice.org.uk/guidance/ng229/chapter/Recommendations#interpreting-cardiotocography-traces)

A displayed 140 does not prove wellbeing. A displayed 180 does not prove a medical problem.

The detector also uses preset beat-spacing limits. It is not designed to reliably detect every extremely fast or slow rhythm.

## 5. Read the four result cards

### Possible fetal rate

The estimated rate of the selected fetal candidate peaks, expressed in BPM.

**Example:** “144 BPM means the accepted peaks are spaced at roughly 0.42 seconds apart. These still need to be checked against references.”

### Possible beats found

The total number of selected candidate peaks in the full recording.

**Example:** 72 detected peaks in a 30-second file roughly corresponds to 144 beats per minute if the rhythm is regular. The displayed BPM can differ from this count-based calculation because the app uses median intervals and excludes some intervals.

### Maternal candidate rate

An estimate from the stronger maternal-pattern candidates. It is also an algorithmic estimate.

**Example:** “The app estimated a maternal candidate rate of 77 BPM and a fetal candidate rate of 144 BPM. Different estimates alone do not prove the two signals were separated correctly.”

### Primary lead quality

A rough score from 0 to 100 for the displayed lead, based on peak-spacing plausibility, regularity, and residual signal energy. It is calculated before final ML or consensus selection.

| Score | App label | Meaning |
| --- | --- | --- |
| 75–100 | Clear candidate signal | The primary lead scored well under this heuristic. Incorrect but regular patterns can still score highly. |
| 45–below 75 | Review signal quality | The heuristic suggests closer inspection. |
| Below 45 | Low-quality candidate | The signal or detected pattern scored poorly and needs careful review. |

These are the app's chosen thresholds, not clinical standards.

**86/100 does not mean 86% accurate or an 86% chance that the fetus is healthy.**

## 6. Read the charts

### Overview

**Original abdominal recording:** the mixed input. Gold dots on this chart show maternal candidate locations.

**Possible fetal signal:** the processed residual. Gold dots here show fetal candidate locations.

- Horizontal axis: time in seconds.
- Vertical axis: waveform amplitude, or signal size.
- A taller spike is a larger signal value. It does not directly mean a stronger or healthier heart.
- A gold dot is a detection made by the algorithm. A dot on a spike is not proof of a correct fetal beat.

Move **Inspect this section of the recording** to look through the waveform. The charts show a window; the result cards summarize the full recording.

**Candidate rhythm over time** summarizes candidate intervals in sliding 10-second windows. A rise means a higher estimated rate in that window. Gaps mean there were not enough usable intervals to calculate a value; they do not demonstrate that a heartbeat stopped. This chart is not a clinical assessment of fetal heart-rate variability.

### Signal lab

1. **Cleaned recording:** filtering reduces some drift and electrical interference.
2. **Maternal-pattern estimate:** a typical repeated maternal shape estimated from detected beats.
3. **Remaining signal (residual):** the result after maternal-pattern reduction and further filtering. Noise and maternal interference can remain.

These charts follow the Overview time window. The download files contain the full recording.

**Say:** “The app estimates the recurring maternal pattern, reduces it, then searches the remaining signal for possible fetal beats.”

## 7. Understand Validation

Reference beats are the known annotation locations used to check the output. Their quality and alignment matter.

The app compares candidate locations with references using an 80-millisecond matching tolerance. That is 0.08 seconds.

| Metric | Plain meaning | Which direction is better for detection? |
| --- | --- | --- |
| Precision | Of the predicted beats, how many match a reference? | Higher |
| Recall | Of the reference beats, how many did the app find? | Higher |
| F1 | A score balancing precision and recall | Higher |
| Rate error | Absolute difference between candidate and reference median-interval BPM | Lower |

### An example with easy numbers

Suppose the reference contains 100 beats. The algorithm predicts 100 beats, but only 90 match the reference:

- 90 correct matches.
- 10 false detections.
- 10 missed reference beats.
- Precision: 90 / 100 = 90%.
- Recall: 90 / 100 = 90%.
- F1: 90%.

**High precision with low recall:** most reported beats match, but many reference beats were missed.

**High recall with low precision:** many reference beats were found, but there were also many false detections.

F1 is the harmonic mean of precision and recall: 2 × precision × recall / (precision + recall). It penalizes imbalance between the two.

### Rate error can be misleading on its own

If candidate BPM is 144 and reference BPM is 143, the rate error is 1 BPM.

A small error means the two summarized rates are close. It does not prove that every beat was detected at the correct time. False detections and missed beats can partially cancel when summarizing a rate.

If reference annotations are missing, detection accuracy has not been measured for that upload.

## 8. What Method and Evidence mean

**Method** describes the processing pipeline.

**Machine-learning candidate ranker:** a small trained model that scores possible beat locations using waveform features. It uses the saved model; uploading a file does not train it.

**Multi-lead consensus:** when at least three abdominal leads are selected, the app looks for candidates supported by three leads within 80 milliseconds. If too few consensus peaks survive, it falls back to single-lead processing. Agreement can reduce some errors, but shared noise can affect several leads.

**Evidence** shows a separate experiment on five public PhysioNet recordings. It is not recalculated from your upload.

The saved multi-lead results are:

- Macro F1: 95.77%.
- Macro precision: 96.50%.
- Macro recall: 95.07%.
- Macro absolute candidate-rate error: approximately 0.60 BPM.

“Macro” means the per-record values were averaged with each record receiving equal weight.

The 95.77% result belongs to the multi-lead pipeline. Do not attribute it solely to the ML ranker: the saved single-lead ML-assisted result is 82.33% macro F1. The ML evaluation trains on four records and tests on the fifth, repeating across the five records. Consensus is evaluated separately and does not itself learn from training labels.

Five recordings are a small research evaluation. These results do not establish clinical reliability or performance across different populations, equipment, or datasets.

## 9. What each download contains

- **Processed waveform CSV:** time, original abdominal signal, cleaned signal, maternal estimate, and residual.
- **Candidate-beat CSV:** selected peak sample positions and their times.
- **Analysis report:** the source, sampling rate, rates, selected beat count, processing method, quality heuristic, and interpretation limits.

The report documents the algorithm's output; it is not a patient assessment.

## 10. A practice explanation you can use

> “I’m using a synthetic sample with reference beats, so these numbers demonstrate the workflow. The app found possible fetal beats after reducing the maternal pattern. The rate card estimates their spacing in beats per minute, while the quality score gives a rough check of the primary signal.
>
> I can inspect the gold markers on the waveform, then open Validation to see how many match the reference beats. Precision measures false detections, recall measures missed beats, and F1 balances the two.
>
> Evidence shows a different experiment on public research recordings. Those results help us evaluate the method, but they do not tell us whether a patient is healthy. This project is for research and education.”

Use the values actually visible in your recording. Do not read memorized numbers if your settings produce different results.

## 11. Check that you understand it

Try answering these before you record:

1. Does 500 Hz mean 500 heartbeats per second? **No; 500 waveform measurements per second.**
2. Does a high candidate BPM prove a medical problem? **No; the source rhythm and detection errors are possible explanations.**
3. Does 90/100 quality mean 90% accuracy? **No; it is a signal heuristic.**
4. What checks whether detected beats match known locations? **Validation against corresponding reference annotations.**
5. Does 95.77% F1 describe every uploaded recording? **No; it summarizes the separate five-record multi-lead evaluation.**
6. Does uploading a waveform train the AI? **No; it uses the existing model.**
