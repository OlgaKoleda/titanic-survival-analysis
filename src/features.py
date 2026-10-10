"""Custom feature engineering step for the Titanic pipeline."""

from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Add Title, FamilySize, HasCabin and Deck, and fill missing Age using the passenger's title.

    fit() learns from the training data which titles are common and the median age
    for each title. transform() applies what was learned to any data.
    """

    def __init__(self, min_title_count=10):
        self.min_title_count = min_title_count  # titles rarer than this become "Rare"

    @staticmethod
    def _extract_title(X):
        title = X["Name"].str.extract(r",\s*([^\.]+)\.")[0]
        return title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    def _group_rare(self, title):
        return title.where(title.isin(self.common_titles_), "Rare")

    def fit(self, X, y=None):
        title = self._extract_title(X)
        counts = title.value_counts()
        self.common_titles_ = set(counts[counts >= self.min_title_count].index)
        self.age_by_title_ = X["Age"].groupby(self._group_rare(title)).median()
        self.age_overall_ = X["Age"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["Title"] = self._group_rare(self._extract_title(X))
        X["Age"] = X["Age"].fillna(X["Title"].map(self.age_by_title_)).fillna(self.age_overall_)
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        X["Deck"] = X["Cabin"].str[0].fillna("U").replace({"G": "U", "T": "U"})
        return X
