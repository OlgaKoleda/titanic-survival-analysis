# Titanic Survival Analysis

Exploratory data analysis and a machine-learning model that predicts which passengers survived the Titanic disaster, using the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data).

![Survival by sex and class](images/survival_by_sex_class.png)

## Key findings

- **Sex** was the strongest predictor: 74% of women survived versus 19% of men.
- **Ticket class** mattered: 63% of 1st-class passengers survived, compared with 24% in 3rd class.
- **Children** and passengers travelling in **small families** (2–4 people) had better chances.
- A **Random Forest** model reaches ~82% cross-validated accuracy, beating a simple "all women survive" baseline (78%).

![Feature importance](images/feature_importance.png)

## Model tuning and Kaggle submission

Random Forest, scikit-learn Gradient Boosting and XGBoost were tuned with grid and randomized search and compared with the same 5-fold stratified cross-validation:

| Model | CV accuracy |
|---|---|
| Gradient Boosting (tuned) | 0.837 |
| Random Forest (tuned) | 0.836 |
| XGBoost (tuned) | 0.834 |
| Logistic Regression | 0.831 |
| Random Forest (default) | 0.824 |

All models land within 1–2 percentage points of each other, so the engineered features (title, sex, class, family size) matter more than the algorithm. The best model was retrained on all 891 passengers to predict the 418 test passengers, and the predictions were saved to [`submissions/submission.csv`](submissions/submission.csv) for the [Kaggle competition](https://www.kaggle.com/competitions/titanic).

**Kaggle public score:** 0.77511 (Gradient Boosting, tuned)

![Model comparison](images/model_comparison.png)

## Project structure

```
├── data/                  # Kaggle CSVs (not committed – see below)
├── images/                # Charts saved by the notebook
├── notebooks/
│   └── titanic_analysis.ipynb
├── submissions/
│   └── submission.csv     # Kaggle predictions for test.csv
├── requirements.txt
└── README.md
```

## How to run

1. Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/titanic/data) into the `data/` folder.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `notebooks/titanic_analysis.ipynb` in VS Code or Jupyter and run all cells.

## Tools

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · Jupyter
