"""Build the full Titanic pipeline: features -> prepare -> model."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import TitanicFeatures

NUMERIC_FEATURES = ["Pclass", "Age", "Fare", "FamilySize", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]


def default_model():
    """Gradient Boosting with the settings chosen in notebooks 02 and 04."""
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
    )


def make_titanic_pipeline(model=None):
    """Return an unfitted pipeline that goes from the raw Kaggle columns to a prediction."""
    return Pipeline([
        ("features", TitanicFeatures()),
        ("prepare", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ], verbose_feature_names_out=False)),
        ("model", model if model is not None else default_model()),
    ])
