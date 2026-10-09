"""Train a small survival model and export it for the "Would you have survived?" page.

The page only asks for sex, ticket class, age and family size, so the model is
trained on exactly those four features. Its predictions for every possible
combination are written into index.html, together with the real passengers,
so the page is a single static file with no Python behind it.

Run from the repository root:  python survival-calculator/build_model.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "index.html"

MAX_AGE = 80
MAX_FAMILY = 8  # "8 or more"; the largest families (8 and 11 people) are grouped together

# Load and clean, the same way as section 2 of the notebook
df = pd.read_csv(ROOT / "data" / "train.csv")
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()[lambda s: s < 10].index
df["Title"] = df["Title"].replace(rare, "Rare")
df["AgeKnown"] = df["Age"].notna()
df["Age"] = df.groupby("Title")["Age"].transform(lambda s: s.fillna(s.median()))
df["FamilySize"] = (df["SibSp"] + df["Parch"] + 1).clip(upper=MAX_FAMILY)
df["Female"] = (df["Sex"] == "female").astype(int)

features = ["Female", "Pclass", "Age", "FamilySize"]
X, y = df[features], df["Survived"]

# Shallow, slow-learning boosting keeps the predicted curves smooth across ages
model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=3, min_samples_leaf=10, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(model, X, y, cv=cv).mean()
model.fit(X, y)
print(f"5-fold CV accuracy with 4 features: {cv_accuracy:.3f}")

# Predict every combination: grid[female][class-1][age][family-1]
ages = np.arange(MAX_AGE + 1)
grid = []
for female in (0, 1):
    by_class = []
    for pclass in (1, 2, 3):
        rows = pd.DataFrame(
            [(female, pclass, age, fam) for age in ages for fam in range(1, MAX_FAMILY + 1)],
            columns=features,
        )
        probs = model.predict_proba(rows)[:, 1].reshape(len(ages), MAX_FAMILY)
        # Trees jump between neighbouring ages; a 7-year rolling mean removes that noise
        smooth = pd.DataFrame(probs).rolling(7, center=True, min_periods=1).mean().to_numpy()
        by_class.append(smooth.round(3).tolist())
    grid.append(by_class)

# Real passengers, for the "people like you" list (only those with a recorded age)
known = df[df["AgeKnown"]]
passengers = [
    [r.Name, r.Female, r.Pclass, float(r.Age), int(r.SibSp + r.Parch + 1), r.Survived]
    for r in known.itertuples()
]

payload = {
    "cvAccuracy": round(cv_accuracy, 3),
    "overallRate": round(y.mean(), 3),
    "maxAge": MAX_AGE,
    "maxFamily": MAX_FAMILY,
    "grid": grid,
    "passengers": passengers,
}
# Replace the data block between the markers in index.html
data = "window.MODEL = " + json.dumps(payload, separators=(",", ":")) + ";"
html = PAGE.read_text()
html, found = re.subn(
    r"(// BEGIN MODEL DATA.*?\n).*?(// END MODEL DATA)",
    lambda m: m.group(1) + data + "\n" + m.group(2),
    html,
    flags=re.S,
)
if not found:
    raise SystemExit("Could not find the MODEL DATA markers in index.html")
PAGE.write_text(html)
print(f"Updated {PAGE.relative_to(ROOT)} ({PAGE.stat().st_size / 1024:.0f} KB)")
