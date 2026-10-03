# Titanic Survival Analysis

Exploratory data analysis and a machine-learning model that predicts which passengers survived the Titanic disaster, using the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data).

![Survival by sex and class](images/survival_by_sex_class.png)

## Key findings

- **Sex** was the strongest predictor: 74% of women survived versus 19% of men.
- **Ticket class** mattered: 63% of 1st-class passengers survived, compared with 24% in 3rd class.
- **Children** and passengers travelling in **small families** (2–4 people) had better chances.
- A **Random Forest** model reaches ~82% cross-validated accuracy, beating a simple "all women survive" baseline (78%).

![Feature importance](images/feature_importance.png)

## Project structure

```
├── data/                  # Kaggle CSVs (not committed – see below)
├── images/                # Charts saved by the notebook
├── notebooks/
│   └── titanic_analysis.ipynb
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

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · Jupyter
