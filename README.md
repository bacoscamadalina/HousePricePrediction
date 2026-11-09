# House Price Prediction

A machine learning project focused on predicting residential sale prices from property characteristics.

## Project overview

The project uses housing data that describes each property through a broad mix of numerical and categorical attributes. The prediction target is `SalePrice`, making this a supervised regression problem.

## Goal

The goal is to predict the sale price of every house in the test set. For each `Id`, the final submission must provide a predicted value for the `SalePrice` variable.

## Evaluation metric

Predictions are evaluated using [Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) between the logarithm of the predicted sale price and the logarithm of the observed sale price:

```text
RMSE(log(predicted SalePrice), log(observed SalePrice))
```

Using logarithms gives errors on lower-priced and higher-priced houses a more comparable influence on the final score.

## Exploratory analysis

The current analysis investigates the target distribution and the relationships between sale price and several candidate predictors, including:

- lot size (`LotArea`)
- above-ground living area (`GrLivArea`)
- overall material and finish quality (`OverallQual`)
- construction and remodeling years (`YearBuilt`, `YearRemodAdd`)

It also explores feature engineering based on property age and remodeling history, together with missing-value patterns such as those found in `LotFrontage`.

## Dataset

The repository includes the tabular data and accompanying feature descriptions:

- `train.csv` — 1,460 properties, 80 non-target columns, and the `SalePrice` target
- `test.csv` — 1,459 properties with the same 80 non-target columns
- `data_description.txt` — definitions and possible values for each feature
- `sample_submission.csv` — expected prediction submission format

The data covers property characteristics such as location, size, quality, condition, construction, amenities, and sale details.

## Approach

The notebook follows an exploratory, evidence-driven workflow:

1. Load and inspect the training and test data.
2. Examine the distribution and summary statistics of `SalePrice`.
3. Visualize relationships between selected predictors and the target.
4. Measure linear associations with correlation analysis.
5. Derive time-based features related to construction and remodeling.
6. Investigate missing values and suitable imputation strategies.

These steps support informed feature selection and establish a basis for preprocessing, model training, and validation.

## Modeling objectives

- Prepare numerical and categorical data for regression modeling.
- Identify useful predictors and engineer relevant features.
- Compare regression approaches using validation aligned with the competition metric.
- Generate one `SalePrice` prediction for each test-set `Id` without data leakage.

## Getting started

Install the core analysis libraries:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Then open `housepriceprediction.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order. Keep the CSV files in the same directory as the notebook.

## Project structure

```text
.
├── housepriceprediction.ipynb
├── train.csv
├── test.csv
├── data_description.txt
├── sample_submission.csv
└── README.md
```

## Tech stack

Python · NumPy · pandas · Matplotlib · Seaborn · Jupyter
