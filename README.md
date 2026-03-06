# Authenticity Detection System

This repository contains a pipeline for detecting anomalous product reviews using feature engineering, anomaly scoring, isolation forest modelling and simple rule-based decisions.

## Adding a Streamlit Frontend

A Streamlit application has been added to the project to make analysis interactive.

### Installation

1. Create or activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the app

Start the Streamlit server from the project root:

```bash
streamlit run app.py
```

The sidebar allows you to upload a CSV of reviews or use the provided sample file (`data/preprocessed/preprocessed_dataset.csv`). After clicking **Run analysis**, results and charts will be displayed. You can download the processed dataset from the UI.

## Existing pipeline

The processing steps are located in the `src` package. The original `main.py` demonstrates a CLI-style run; the Streamlit app uses the same functions internally.

Feel free to extend or modify the UI or model components as needed.
