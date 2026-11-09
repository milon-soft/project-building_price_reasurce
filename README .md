# House Price Prediction

Machine-learning regression experiment using the Kaggle Ames Housing competition dataset.

## Files

- `literature_review.md` — summaries of four research and technical sources.
- `methodology.md` — research question, dataset, cleaning, feature engineering, models, and metrics.
- `experiment.ipynb` — end-to-end data preparation, model training, evaluation, and findings.
- `dataset/train.csv` — download this file yourself from Kaggle; it is intentionally not included in this repository.

## Dataset setup

1. Download `train.csv` from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. Create a folder named `dataset` in the project root.
3. Place `train.csv` inside `dataset/`.
4. Install the packages listed in `requirements.txt`.
5. Open `experiment.ipynb` in Jupyter Notebook or VS Code and run all cells.

The notebook uses an 80/20 validation split and compares Ridge Regression with Random Forest. Do not report metrics until the notebook has run successfully.

## Reproducibility

Random seed: 42. See `methodology.md` for the complete experimental design.
