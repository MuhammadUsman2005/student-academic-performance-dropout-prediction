# Dataset Sources and Documentation

This project combines two public data sources to study early prediction of student course withdrawal.

## 1. Open University Learning Analytics Dataset (OULAD)

- **Original source:** https://analyse.kmi.open.ac.uk/open_dataset
- **Download source used:** https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
- **Original publisher:** The Open University
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Access date:** 21 November 2026
- **Download method:** Public dataset downloaded from Kaggle, which provides a mirror of the original OULAD dataset.

### Files used

- studentInfo.csv
- studentRegistration.csv
- assessments.csv
- studentAssessment.csv

These tables were joined using their relevant student, course and assessment identifiers.

## 2. Regional Unemployment Data

- **Dataset:** Unemployment Rate, Region
- **Source:** https://www.data.gov.uk/dataset/448923c0-6224-492a-b4d8-db239cdc7724/unemployment-rate-region1
- **Related publisher dataset:** https://data.london.gov.uk/dataset/unemployment-rate-region-e5mnw
- **Data producer:** Office for National Statistics (ONS)
- **Maintainer:** Greater London Authority
- **Licence information:** The London Datastore dataset record states Open Government Licence v2. The corresponding data.gov.uk record does not specify a licence. Refer to the source records for their respective terms.
- **Access date:** 21 November 2026
- **Download method:** Public Excel workbook downloaded from the linked government data source.

The unemployment data was cleaned and mapped to eight compatible OULAD regions. Ambiguous or non-equivalent regional categories were excluded from the mapping.

## 3. Data Integration and Preparation

The OULAD student, registration and assessment tables were combined to create student-course records.

The analysis uses students eligible for prediction at Day 30. The target variable is defined as:

- `1`: Student subsequently withdrew from the course.
- `0`: Student did not withdraw from the course.

The regional unemployment rate was added using the available region and historical presentation mapping. A conservative three-month time delay was applied when matching historical unemployment periods. This is an assumption used in the project, not a verified publication schedule.

Unmatched unemployment values were retained as missing values and handled by the model's preprocessing pipeline.

## 4. Final Dataset

- **Records:** 27,443
- **Input features:** 15
- **Target:** `dropout_risk`
- **Target meaning:** Subsequent course withdrawal, not permanent university dropout.

The UCI Predict Students' Dropout and Academic Success dataset was also inspected separately but was not merged into the final modelling dataset because it represents a different population and outcome structure.

## 5. Privacy and Limitations

The project uses publicly available, anonymised or aggregated data. Direct personal identifiers are not used.

Regional unemployment data is not available for every student record because only compatible regional mappings were retained. Missing values are handled by the model pipeline.

The model predicts course withdrawal in the OULAD context and should not be interpreted as a prediction of permanent university dropout.
