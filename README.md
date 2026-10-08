# MedPredictor-AI

MedPredictor-AI is an **offline machine-learning research project** for two binary classification tasks:

- **Diabetes classification** using the PIMA Indians Diabetes dataset.
- **Ten-year cardiovascular-risk classification** using the Framingham dataset.

> **Medical disclaimer:** this is research/education software, not a clinical diagnostic system.

## What it does

The project takes structured health measurements and:

1. Loads and validates the repository datasets.
2. Splits each dataset into training data and one untouched 20% test set.
3. Fits median imputation and standardization only on the training split.
4. Benchmarks multiple models with stratified 5-fold cross-validation.
5. Searches model hyperparameters on training folds.
6. Tunes the probability threshold from out-of-fold training predictions while enforcing a minimum recall.
7. Evaluates the selected model once on the untouched test set.
8. Generates ROC curves, confusion matrices, feature-importance plots, and model comparisons.
9. Saves the trained model, imputer, scaler, and decision threshold as one local artifact.
10. Runs a sample prediction or accepts interactive example input.

## Model portfolio

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- Support Vector Machine
- MLP neural network
- **XGBoost** when the optional XGBoost dependency is installed

The current PyPI XGBoost release is 3.4.1 and it requires Python 3.12+. citeturn119920search3

## Accuracy and the 90% target

The benchmark reports both cross-validation accuracy and final held-out test accuracy.

The CLI explicitly prints:

    90% held-out accuracy target: MET / NOT MET

**90% is a target, not a guarantee.** A result above 90% is only legitimate when the untouched test set actually produces it. Repeatedly tuning against the test labels or leaking preprocessing statistics would make the reported accuracy unreliable.

## How prediction works

    raw health values
          ↓
    input validation
          ↓
    training-fitted imputer
          ↓
    training-fitted scaler
          ↓
    selected ML model
          ↓
    positive-class probability
          ↓
    out-of-fold tuned threshold
          ↓
    class + probability

The saved artifact contains the preprocessing state required for reproducible inference.

## Metrics

- Cross-validation accuracy
- Held-out test accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Average precision
- Decision threshold
- Confusion matrix

Accuracy alone is insufficient for medical classification, especially for the imbalanced Framingham target.

## Requirements

- Python 3.11+
- pip

Install the baseline stack:

    python -m pip install -r requirements.txt

Optional XGBoost benchmark:

    python -m pip install -r requirements-boost.txt

Development/test dependencies:

    python -m pip install -r requirements-dev.txt

scikit-learn 1.9.1 is the currently supported release listed by the project's security policy. citeturn119920search0

## Run

Full tuned benchmark:

    python main.py

Only diabetes:

    python main.py --disease diabetes

Only heart-risk:

    python main.py --disease heart

Fast smoke test:

    python main.py --fast --disease diabetes

Disable XGBoost:

    python main.py --no-xgboost

Interactive example patient:

    python main.py --disease diabetes --interactive

Generated plots and trusted local model artifacts are written to outputs/.

## Data preparation

### Diabetes

The PIMA data includes pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age, and the Outcome label.

Zero values in physiological fields where zero is not meaningful are treated as missing. Median imputation is fitted on the training split only.

### Heart disease

The Framingham data uses the original TenYearCHD target, renamed in code to HeartDiseaseRisk. Rows containing missing values are removed.

## Security and reliability

- Dataset and output paths are repository-relative.
- Generated filenames reject path traversal.
- Prediction inputs must be finite numeric values.
- Test-set labels are not used to tune preprocessing, hyperparameters, or thresholds.
- The application does not automatically load arbitrary user-supplied joblib files.
- Never load untrusted joblib/pickle files because Python object deserialization can execute arbitrary code.
- scikit-learn remains on its currently supported 1.9.x line. citeturn119920search0

## Project structure

    MedPredictor-AI/
    ├── data/
    ├── notebooks/
    ├── Health related project/
    ├── main.py
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── models.py
    ├── predict.py
    ├── visualization.py
    ├── requirements.txt
    ├── requirements-boost.txt
    ├── requirements-dev.txt
    ├── tests/
    └── outputs/

## Tests

    python -m pytest

The tests cover dataset loading, training-split preprocessing, input validation, feature-analysis edge cases, artifact isolation, and threshold-aware prediction.

## Medical disclaimer

**Research/education only.** Predictions are statistical model outputs, not diagnoses or treatment recommendations. These historical datasets do not establish clinical safety, fairness, or real-world performance on today's patient population.

## License

MIT
