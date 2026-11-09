# Methodology: House Price Prediction Using Machine Learning

## 1. Refined research question

How accurately can machine-learning regression models predict residential sale prices in Ames, Iowa, using property characteristics such as overall quality, living area, lot size, age, bathrooms, garage capacity, and neighborhood?

## 2. Dataset description

**Dataset:** Kaggle *House Prices: Advanced Regression Techniques* (`train.csv`).

- **Observations:** 1,460 labeled training properties.
- **Columns:** 81 total columns: an identifier (`Id`), 79 explanatory variables, and the target (`SalePrice`).
- **Target:** `SalePrice`, a positive continuous value measured in US dollars.
- **Feature types:** Numerical, ordinal, and nominal/categorical variables.
- **Examples:** `OverallQual`, `GrLivArea`, `LotArea`, `YearBuilt`, `GarageCars`, `TotalBsmtSF`, and `Neighborhood`.
- **Source:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

### Dataset limitations

The dataset is relatively small and represents historical sales in one city (Ames, Iowa), not the current housing market or other geographic regions. Some fields contain missing values, and the distribution of sale prices is right-skewed. Results therefore should not be assumed to generalize to other locations or present-day prices. The Kaggle `test.csv` file has no observed `SalePrice`, so model evaluation in this experiment will use a hold-out validation split from `train.csv`.

## 3. Data cleaning plan

1. Load `train.csv` and inspect shape, data types, summary statistics, and target distribution.
2. Check duplicate rows and duplicate `Id` values; remove exact duplicate records if any are found.
3. Drop `Id` from the predictors because it is an identifier, not a meaningful property measurement.
4. Separate `SalePrice` from the predictors before modeling.
5. Treat missing values within a scikit-learn preprocessing pipeline: median imputation for numerical columns and a constant `"Missing"` category for categorical columns.
6. One-hot encode categorical variables with unknown categories ignored at validation/prediction time.
7. Fit all imputers and encoders on the training subset only, through a pipeline, to avoid leakage.
8. Inspect unusual values and influential outliers. Do not remove observations automatically; document any exclusion and its rationale.

## 4. Feature engineering plan

Create the following features when their source columns are present:

- `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`.
- `TotalBathrooms` = `FullBath` + `0.5 * HalfBath` + `BsmtFullBath` + `0.5 * BsmtHalfBath`.
- `HouseAge` = `YrSold` - `YearBuilt`.
- `RemodelAge` = `YrSold` - `YearRemodAdd`.
- `TotalPorchSF` = `WoodDeckSF` + `OpenPorchSF` + `EnclosedPorch` + `3SsnPorch` + `ScreenPorch`.

These are derived from information available in the sale record. Negative age values, if encountered, will be inspected and clipped to zero for these age features.

## 5. Models and rationale

Two regression models will be trained:

1. **Ridge Regression (alpha=10):** a regularized linear baseline. It is relatively interpretable and regularization can reduce instability caused by correlated predictors and the many one-hot encoded columns.
2. **Random Forest Regressor (300 trees):** a tree-ensemble model that can learn nonlinear patterns and interactions without requiring a linear relationship between each feature and price.

The target will be transformed using `log1p` during training. Predictions will be transformed back with `expm1` before reporting dollar-based errors. This makes the training objective more consistent with the logarithmic competition metric.

## 6. Train/validation design

Use an 80/20 random train/validation split from the 1,460 labeled rows, with `random_state=42` for reproducibility. All preprocessing is fitted inside each model pipeline using only the training partition. The validation partition is held out until evaluation.

## 7. Evaluation metrics

- **Primary: RMSLE (Root Mean Squared Logarithmic Error).** This is the Kaggle competition's evaluation metric and measures relative/log-scale prediction error. Lower is better. Predictions must be positive for this metric.
- **Secondary: MAE (Mean Absolute Error).** Reports the average absolute error in dollars and is easy to interpret. Lower is better.
- **Secondary: R-squared (R²).** Describes the proportion of validation-target variance explained by the model. Higher is generally better; it can be negative if the model performs worse than a constant-mean baseline.

The model with the lowest validation RMSLE will be considered the best model for this experiment. MAE and R² will also be reported so that the comparison is not based on one metric alone.

## 8. Reproducibility and limitations

Use a fixed random seed, record library versions where practical, and keep the notebook's preprocessing and model code together. The validation score is an estimate based on one split and may vary with a different split. A future version should use cross-validation, tune hyperparameters, investigate outliers more deeply, and test external or more recent housing data before making broader claims.
