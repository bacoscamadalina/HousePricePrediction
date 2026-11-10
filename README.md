# 🏠 House Price Prediction

This is my project for the Kaggle **House Prices: Advanced Regression Techniques** competition.

The main goal is to predict the `SalePrice` of the houses from the test dataset. Since the value I want to predict is numerical, this is a regression problem.

## 📂 Dataset

The project contains:

- `train.csv` — 1,460 houses and their sale prices;
- `test.csv` — 1,459 houses without sale prices;
- `data_description.txt` — description of all the features;
- `sample_submission.csv` — example of the format required by Kaggle.

The dataset has information about each house, such as its size, quality, neighborhood, construction year, garage, basement, and number of bathrooms.

## 🔍 Exploratory Data Analysis

I started by analyzing `SalePrice`, which is the target variable.

The distribution is right-skewed. The mean price is approximately **$180,921**, while the median is **$163,000**. The mean is higher because there are some houses with much higher prices.

I then checked the relationship between `SalePrice` and several features:

| Feature | Correlation with SalePrice |
|---|---:|
| `LotArea` | 0.264 |
| `GrLivArea` | 0.709 |
| `YearBuilt` | 0.523 |
| `YearRemodAdd` | 0.507 |
| `DiffYear` | -0.218 |
| `YearsSinceRemodel` | -0.509 |

`GrLivArea` has a much stronger relationship with the sale price than `LotArea`. The boxplot for `OverallQual` also shows that houses with higher quality usually have higher prices.

Correlation does not prove that one variable causes another, but it helps me decide which features may be useful for a model.

## 🧩 New features and missing values

I created two new features:

```text
DiffYear = YearRemodAdd - YearBuilt
YearsSinceRemodel = YrSold - YearRemodAdd
```

I also investigated the missing values from `LotFrontage`.

The mean `LotFrontage` is approximately **70.05 ft**, and the median is **69 ft**. Because the distribution is right-skewed, I used median imputation in the preprocessing pipeline.

The median frontage is different from one neighborhood to another. I tested adding `LotFrontage` to the model, but the validation MAE increased from **$21,253.93** to **$21,604.66**, so it did not improve the model in this version.

## 🤖 Models

I used an 80/20 train-validation split with `random_state=42` and tested three types of regression models:

- Linear Regression;
- Decision Tree Regressor;
- Random Forest Regressor.

I used **Mean Absolute Error (MAE)** to compare the local validation results. A lower MAE means that the predictions are closer to the real prices.

| Model | Validation MAE |
|---|---:|
| Linear Regression | $28,028.15 |
| Decision Tree | $29,779.10 |
| Decision Tree (`max_depth=7`) | $25,886.22 |
| Random Forest | $22,945.75 |
| Random Forest + `Neighborhood` | $21,253.93 |
| Random Forest + `LotFrontage` | $21,604.66 |
| Random Forest V2 | **$18,254.15** |

The first Decision Tree had a training MAE of only **$50.50**, but its validation MAE was **$29,779.10**. This showed that the model was overfitting. Limiting the tree depth improved the validation result.

Random Forest performed better than Linear Regression and Decision Tree for the features I tested.

## ⚙️ Preprocessing and model improvements

I added `Neighborhood` as a categorical feature and used one-hot encoding. This reduced the Random Forest validation MAE from **$22,945.75** to **$21,253.93**.

For the next version, I added:

- `TotalBsmtSF`;
- `GarageCars`;
- `FullBath`.

With these features, the validation MAE decreased to **$18,254.15**.

The preprocessing is included in a scikit-learn pipeline. It uses:

- `SimpleImputer` with the median for numerical missing values;
- `OneHotEncoder` for `Neighborhood`;
- `ColumnTransformer` for applying the transformations to the correct columns;
- `RandomForestRegressor` for the final model.

## 📈 Cross-validation and Grid Search

I used five-fold cross-validation to get a more reliable estimate of the model performance.

- Average cross-validation MAE: **$19,462.02**;
- Best Grid Search MAE: **$19,400.78**;
- Best parameters: `n_estimators=200` and `max_depth=None`.

Grid Search produced only a small improvement. Adding useful features had a larger effect on the model than changing these parameters.

## 📏 Evaluation metric

I currently use MAE for the local experiments because it is easy to understand in dollars.

The Kaggle competition uses **RMSLE**, so the local MAE results cannot be compared directly with the Kaggle leaderboard score. Adding RMSLE to the notebook is one of my next steps.

## 📦 Submission files

The notebook creates three submission files:

- `submission.csv` — first Random Forest submission;
- `submission_v2.csv` — model with the additional features;
- `submission_v3.csv` — model selected with Grid Search.

Each submission contains the two columns required by Kaggle:

```text
Id, SalePrice
```

## 🗂️ Project structure

```text
home-data-for-ml-course/
├── housepriceprediction.ipynb
├── train.csv
├── test.csv
├── data_description.txt
├── sample_submission.csv
├── submission.csv
├── submission_v2.csv
├── submission_v3.csv
└── README.md
```

## 🛠️ Libraries used

- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## ▶️ How to run the project

Install the libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `housepriceprediction.ipynb` and run the cells in order. The CSV files need to be in the same folder as the notebook.

## 🚧 Next steps

- add RMSLE as an evaluation metric;
- test neighborhood-based imputation for `LotFrontage`;
- try more features and compare their cross-validation results;
- retrain the final pipeline on all training rows;
- continue improving the Kaggle score.
