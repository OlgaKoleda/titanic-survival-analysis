"""Train the pipeline on train.csv and save it to models/titanic_pipeline.joblib.

Run from the project folder:
    python -m src.train
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.pipeline import make_titanic_pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_PATH = PROJECT_DIR / "models" / "titanic_pipeline.joblib"


def main():
    df = pd.read_csv(TRAIN_PATH)
    X, y = df.drop(columns="Survived"), df["Survived"]

    pipeline = make_titanic_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained on {len(X)} passengers and saved to {MODEL_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
