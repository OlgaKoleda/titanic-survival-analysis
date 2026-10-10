"""Predict survival for the passengers in a CSV file, using the saved pipeline.

Run from the project folder:
    python -m src.predict data/test.csv
    python -m src.predict data/test.csv --output submissions/my_predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "titanic_pipeline.joblib"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", help="CSV with the same columns as Kaggle's test.csv")
    parser.add_argument("--output", help="where to save the predictions (optional)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Train one first with: python -m src.train")

    pipeline = joblib.load(MODEL_PATH)
    passengers = pd.read_csv(args.csv_path)

    result = pd.DataFrame({
        "PassengerId": passengers["PassengerId"],
        "Survived": pipeline.predict(passengers).astype(int),
        "SurvivalChance": pipeline.predict_proba(passengers)[:, 1].round(3),
    })

    print(f"Predicted {len(result)} passengers, {result['Survived'].mean():.1%} survive")
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
