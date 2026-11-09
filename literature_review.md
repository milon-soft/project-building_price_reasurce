# Literature Review: House Price Prediction

## 1. De Cock (2011) — *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*

De Cock presents a dataset describing 2,930 residential property sales in Ames, Iowa, from 2006 to 2010. The paper explains how a broad set of property characteristics—including numerical measurements and categorical descriptions—can support regression analysis and make the dataset useful for teaching and studying house-price prediction.

Source: https://doi.org/10.1080/10691898.2011.11889627

## 2. Kaggle — *House Prices: Advanced Regression Techniques*

The Kaggle competition uses the Ames housing data to predict each property's sale price from 79 explanatory variables. Its overview highlights feature engineering and regression methods, and its evaluation uses Root Mean Squared Logarithmic Error (RMSLE), making it a practical reference for the target problem and evaluation design in this project.

Source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## 3. scikit-learn — *ColumnTransformer and Pipelines*

The scikit-learn documentation describes tools for applying different preprocessing operations to different column types and combining preprocessing with a predictive estimator. This supports the project design of imputing numerical and categorical values separately, one-hot encoding categories, and fitting preprocessing only on training folds to reduce data leakage.

Sources:
- https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- https://scikit-learn.org/stable/modules/compose.html

## 4. scikit-learn — *Regression Model Evaluation*

The scikit-learn model-evaluation guide documents regression metrics such as mean absolute error, root mean squared error, root mean squared logarithmic error, and R-squared. It helps justify using RMSLE as the primary metric to compare with the Kaggle task, while MAE and R-squared provide additional interpretations of error size and explained variation.

Source: https://scikit-learn.org/stable/modules/model_evaluation.html

---
