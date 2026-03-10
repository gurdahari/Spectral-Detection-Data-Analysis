# Spectral Detection Data Analysis

Python analysis project for controlled and uncontrolled RF spectral detections, prepared for GitHub publishing.

## What is included

- `analysis.py` — main analysis script
- `data/` — the three CSV datasets used by the analysis
- `report/spectral_detection_data_analysis_report_redacted.pdf` — redacted report version without personal name
- `requirements.txt` — Python dependencies
- `.gitignore` — common Python ignores

## Project overview

This repository analyzes RF spectral-snapshot detections produced by an object detector and a tracker.

The workflow covers two main tasks:

1. **Exploratory data analysis** on a controlled dataset and two uncontrolled datasets.
2. **Trigger design** using a Neyman–Pearson threshold with track-level reliability gates.

Main analysis topics include:

- score distributions
- signal power, SINR, bandwidth, and center-frequency summaries
- precision/recall tables on controlled data
- false positive rate behavior on uncontrolled data
- track-based trigger evaluation using `association_id`

## Repository structure

```text
spectral-detection-analysis/
├── analysis.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── controlled.csv
│   ├── uncontrolled_detections_export_config_1.csv
│   └── uncontrolled_detections_export_config_2.csv
└── report/
    └── spectral_detection_data_analysis_report_redacted.pdf
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Interactive mode:

```bash
python analysis.py
```

Headless mode without opening plot windows:

```bash
python analysis.py --no-show
```

Save plots to a folder:

```bash
python analysis.py --no-show --save-plots-dir outputs/plots
```

## Notes

- The script expects the CSV files to be inside `data/` by default.
- `seaborn` is optional and is only used for the score-by-category boxplot.
- The PDF in `report/` is the redacted version prepared for public sharing.

## Suggested GitHub repo name

`spectral-detection-analysis`
