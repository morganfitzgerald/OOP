# ECG_tool_validation

This repository provides a pipeline for processing and analyzing electrocardiogram waveform shapes. The tool is the first to capture several ECG wavform shape features on a beat-to-beat basis.

## Key Features

1. **Signal Processing:** Crop and normalize ECG signals for focused analysis.
2. **Filtering:** Apply high-pass and notch filters to clean the signal and remove noise.
3. **Template Matching:** Identify the most similar signals to a given ECG template.
4. **Signal Cleaning & Feature Extraction:** Use Neurokit to clean ECG signals and extract key waveforms (P, Q, R, S, T peaks).
5. **Epoching:** Align and segment signals based on P-peaks.
6. **Parameterization Loop:** Automatically detect and parameterize ECG peaks across cycles.
7. **HRV Metrics Calculation:** Compute key HRV metrics like SDNN and RMSSD for comprehensive heart rate analysis.

## Usage

This tool automates the complex process of ECG signal analysis, making it accessible for various applications such as clinical research, biomedical engineering, and physiological monitoring. By using this pipeline, you can efficiently process large ECG datasets, extract relevant features, and gain insights into cardiac health.
