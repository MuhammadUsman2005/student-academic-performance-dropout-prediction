# Student Academic Performance & Dropout Risk Prediction

A machine learning project that predicts the likelihood of **course withdrawal after the first 30 days** using student demographics, academic history, registration details, early assessment activity, and historical regional unemployment information.

## Live Application

**Streamlit App:** https://student-withdrawal-risk-predictor.streamlit.app/

The application provides:
- A student information input form
- Predicted withdrawal probability
- Higher-risk / lower-risk classification
- A probability comparison chart
- Automatic historical unemployment-rate lookup

## Problem Statement

Early identification of students at risk of course withdrawal may help educational institutions provide timely academic support.

This project builds a binary classification model to estimate whether a student who is still enrolled at Day 30 will subsequently withdraw from their course.

**Important:** The project predicts course withdrawal, not permanent university dropout.

## Data Sources

### 1. Open University Learning Analytics Dataset (OULAD)

https://analyse.kmi.open.ac.uk/open_dataset

Used student demographics, registration information, final course outcomes, and assessment activity.

### 2. UK Regional Unemployment Statistics

https://www.data.gov.uk/dataset/448923c0-6224-492a-b4d8-db239cdc7724/unemployment-rate-region1

Historical regional unemployment rates provide additional socioeconomic context. Only clearly matched regions were joined.

### Additional Reference Dataset

UCI Predict Students' Dropout and Academic Success:

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

This dataset was inspected separately and was not merged with the main training dataset.

## Dataset Summary

| Item | Value |
|---|---|
| Eligible enrollment records | 27,443 |
| Input features | 15 |
| Target | dropout_risk |
| Positive class | Withdrawn |
| Negative class | Not Withdrawn |
| Prediction point | Day 30 |
| Training records | 19,189 |
| Validation records | 4,136 |
| Testing records | 4,118 |

Student identifiers were excluded from model features. Splitting was group-aware to prevent the same student appearing in different sets.

## Machine Learning Methodology

1. Data collection and integration
2. Day-30 eligibility filtering
3. Exploratory data analysis
4. Missing-value imputation
5. Categorical encoding and numerical scaling
6. Student-group-aware train/validation/test splitting
7. Baseline classification
8. Training Logistic Regression, Random Forest, and KNN
9. Five-fold StratifiedGroupKFold cross-validation
10. Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
11. Validation-based threshold selection
12. Final held-out test evaluation

Preprocessing was implemented using scikit-learn pipelines to avoid fitting preprocessing transformations on validation or test data.

## Final Model

**Algorithm:** Random Forest

**Selected decision threshold:** 0.51

### Held-Out Test Performance

| Metric | Score |
|---|---|
| Accuracy | 0.6872 |
| Precision | 0.2884 |
| Recall | 0.5178 |
| F1-Score | 0.3705 |
| ROC-AUC | 0.6733 |

### Confusion Matrix

| Actual / Predicted | Not Withdrawn | Withdrawn |
|---|---:|---:|
| Not Withdrawn | 2451 | 935 |
| Withdrawn | 353 | 379 |

The model identified 379 of 732 withdrawn students, with 935 false-positive predictions. Performance is moderate, so predictions should be interpreted cautiously.

## Deployment

The application is deployed publicly on Streamlit Community Cloud.

**URL:** https://student-withdrawal-risk-predictor.streamlit.app/

### Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Repository Files

- `app.py`: Streamlit web application
- `model.joblib`: Trained preprocessing and Random Forest pipeline
- `model_config.json`: Model threshold and feature configuration
- `category_options.json`: Categorical input choices
- `region_mapping.json`: OULAD-to-ONS regional mapping
- `unemployment_lookup.csv`: Historical regional unemployment lookup
- `requirements.txt`: Python dependencies

## Limitations

- Predicts course withdrawal rather than permanent university dropout.
- Developed using historical UK Open University data.
- External unemployment data are unavailable for some OULAD regions.
- Assessment activity is limited to the first 30 days.
- Prediction probabilities have not been independently calibrated.
- The model is an educational prototype, not a production-ready academic decision system.

## Project Context

NED University of Engineering & Technology
Department of Computer Science & Information Technology
Machine Learning (CT-378) — Course Project
