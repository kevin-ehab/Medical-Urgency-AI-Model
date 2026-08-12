# Medical-Urgency-AI-Model
### Emergency Triage Classification

A machine learning project that predicts whether an emergency-department
patient should be classified as an **emergency** based on demographic
information, vital signs, pain-related information, mental status,
injury status, and chief complaint.

> **Important:** This is an educational machine learning project. The
> model is not a medical diagnostic or treatment tool and should not be
> used for real clinical decisions.

## Project Overview

The original dataset contains emergency-department triage information,
including the nurse-assigned **KTAS** (Korean Triage and Acuity Scale)
level.

For this project, the original five-level KTAS target was converted into
a binary classification target:

-   `Emergency = 1` -> `KTAS_RN <= 3`
-   `Emergency = 0` -> `KTAS_RN > 3`

The goal is therefore to learn:

**Patient information -> Emergency / Non-Emergency**

The dataset contains **1,233 usable patient records** after
preprocessing.

## Features

### Numerical features

  Feature      Description
  ------------ --------------------------
  `Age`        Patient age
  `SBP`        Systolic blood pressure
  `DBP`        Diastolic blood pressure
  `HR`         Heart rate
  `RR`         Respiratory rate
  `BT`         Body temperature
  `NRS_pain`   Numeric pain rating

### Categorical features

  Feature            Description
  ------------------ ---------------------------
  `Sex`              Patient sex category
  `Injury`           Injury status/category
  `Mental`           Mental-status category
  `Pain`             Pain indicator
  `Chief_complain`   Patient's chief complaint

## Data Preprocessing

The dataset was loaded from `data.csv` using a semicolon separator and
Latin-1 encoding.

Several source-data issues were handled before training:

1.  Rows containing `"??"` in `HR`, `RR`, `BT`, `SBP`, or `DBP` were
    removed.
2.  Invalid `NRS_pain` entries (`#BOÞ!`) were replaced with the median
    of the valid numeric pain scores.
3.  The numerical features were converted to floating-point values.
4.  The target `Emergency` was created from `KTAS_RN`.

The final feature set consists of:

``` python
num_x = ["Age", "SBP", "DBP", "HR", "RR", "BT", "NRS_pain"]
cat_x = ["Sex", "Injury", "Mental", "Pain", "Chief_complain"]
```

## Model

The final model uses an **RBF Support Vector Classifier (SVC)** inside a
scikit-learn Pipeline.

The preprocessing pipeline is:

``` text
Numerical features
        †
StandardScaler
        †
Categorical features
        †
OneHotEncoder
        †
SVC (RBF kernel)
        †
Emergency prediction
```

### Hyperparameter tuning

`GridSearchCV` was used with 5-fold cross-validation.

The searched hyperparameters were:

``` python
param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": ["scale", 0.01, 0.1, 1]
}
```

The grid search optimized **F1 score**.

The selected configuration was:

``` text
C = 100
gamma = 0.01
```

## Results

The final model was evaluated on a held-out test set of **247
patients**.

  Metric                        Score
  --------------------- -------------
  Accuracy                 **81.78%**
  Emergency Recall         **81.25%**
  Emergency F1 Score       **82.21%**
  Emergency Precision     **\~83.2%**

Classification report:

``` text
              precision    recall  f1-score   support

           0       0.80      0.82      0.81       119
           1       0.83      0.81      0.82       128

    accuracy                           0.82       247
   macro avg       0.82      0.82      0.82       247
weighted avg       0.82      0.82      0.82       247
```

Confusion matrix:

``` text
                  Predicted
                  0      1
Actual 0         98     21
Actual 1         24    104
```

This means the model correctly classified 104 of the 128 emergency cases
in the test set.

## Project Structure

A simple version of the project can be organized as:

``` text
.
├── data.csv
├── model_preparation_notebook.ipynb
├── model.pkl
└── README.md
```

-   `data.csv` --- source dataset used by the notebook.
-   `model_preparation_notebook.ipynb` --- data preparation,
    preprocessing, training, tuning, and evaluation.
-   `model.pkl` --- serialized trained model.
-   `README.md` --- project documentation.

## Requirements

The notebook uses Python and the following main libraries:

``` text
pandas
scikit-learn
matplotlib
joblib
```

Install them with:

``` bash
pip install pandas scikit-learn matplotlib joblib
```

## Running the Project

1.  Place `data.csv` in the same directory as the notebook.
2.  Open `model_preparation_notebook.ipynb` in Jupyter Notebook,
    JupyterLab, or Google Colab.
3.  Run the cells in order.
4.  The trained model can be saved with:

``` python
import joblib

joblib.dump(model, "model.pkl")
```

The saved model includes the preprocessing pipeline, so new data can be
passed through the same transformations used during training.

## Limitations

-   The dataset contains only 1,233 usable records after preprocessing.
-   `Chief_complain` contains many distinct categories, so some
    categories may not appear in the training portion of a split.
-   The model was evaluated using one held-out test split
    (`test_size=0.2`, `random_state=42`).
-   The target is derived directly from `KTAS_RN`, so this project
    should be interpreted as a machine learning experiment based on the
    existing triage labels.
-   The results should not be interpreted as evidence of clinical safety
    or real-world medical performance.
-   The model should not be used to determine whether a real patient
    requires emergency treatment.

## Future Improvements

Possible next steps include:

-   Evaluate the model with repeated or stratified cross-validation.
-   Test additional classification algorithms.
-   Investigate the effect of different decision thresholds.
-   Analyze feature importance or model interpretability.
-   Test the model on an independent external dataset.
-   Investigate the effect of the high-cardinality `Chief_complain`
    feature.
-   Perform more comprehensive validation before drawing conclusions
    about generalization.

## License / Dataset

The notebook uses a local file named `data.csv`. The original dataset's
licensing and attribution requirements should be checked before
redistributing the dataset or publishing it with this project.
[original dataset](https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application)

## Disclaimer

This project was created for **educational and machine learning
purposes**. It is not intended to diagnose disease, determine treatment,
replace medical professionals, or make clinical decisions.
