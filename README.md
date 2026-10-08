# MedPredictor-AI

MedPredictor-AI is an offline Python machine-learning research project for experimenting with diabetes and ten-year cardiovascular-risk classification.

It trains several scikit-learn classifiers, compares their performance, generates diagnostic plots, saves the best trained artifact, and runs a sample or interactive prediction.

> **Medical disclaimer:** This repository is for education and research. Its predictions are not medical diagnoses and must not be used as a substitute for a qualified clinician or for patient-care decisions.

## What it does

### Diabetes
Uses the PIMA Indians Diabetes dataset with predictors for pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.

### Heart disease
Uses the Framingham dataset and its original `TenYearCHD` target, renamed in code to `HeartDiseaseRisk`.

## Models

The pipeline evaluates:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors

Models are ranked by ROC AUC on the held-out test set.

## Requirements

- Python 3.10+
- pip

Install runtime dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

For development/testing:

~~~bash
python -m pip install -r requirements-dev.txt
~~~

## Run

Train and evaluate both datasets:

~~~bash
python main.py
~~~

Only diabetes:

~~~bash
python main.py --disease diabetes
~~~

Only heart-risk:

~~~bash
python main.py --disease heart
~~~

Enter a custom example patient after training:

~~~bash
python main.py --disease diabetes --interactive
python main.py --disease heart --interactive
~~~

Generated plots and trusted local model artifacts are written to `outputs/`.

## Project layout

~~~text
MedPredictor-AI/
├── data/
│   ├── diabetes.csv
│   └── framingham.csv
├── notebooks/
├── Health related project/   # original exploratory material
├── main.py
├── data_preprocessing.py
├── feature_engineering.py
├── models.py
├── predict.py
├── visualization.py
├── requirements.txt
├── requirements-dev.txt
├── tests/
└── outputs/
~~~

## Data handling

For diabetes, zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as missing and replaced with the corresponding dataset median.

For heart disease, rows containing missing values are removed and `TenYearCHD` is renamed to `HeartDiseaseRisk`.

Scaling is fitted only on the training split and then applied to the test split, avoiding test-set leakage.

## Security and reliability

- Dataset and output paths are resolved relative to the repository.
- Generated filenames accept only simple local names; path traversal is rejected.
- Patient inputs must be numeric and finite.
- The project does not load user-supplied model files.
- Local model artifacts are written with joblib only after training.
- Joblib is kept above the historical arbitrary-code-execution threshold; GitHub's advisory database lists versions below 1.2.0 as affected. citeturn922828search3
- scikit-learn is pinned to its currently supported 1.9.x line; its security policy currently lists 1.9.1 as supported and older releases as unsupported. citeturn922828search0

Do not load a .joblib or pickle file from an untrusted source. Python object deserialization is not a safe interchange format.

## Development

~~~bash
python -m pytest
~~~

Tests cover dataset loading, feature-analysis edge cases, input validation, and model-artifact path handling.

## License

MIT
