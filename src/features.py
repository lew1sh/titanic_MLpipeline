import numpy as np
import pandas as pd


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create Titanic features used by the training pipeline."""
    target = config["project"]["target"]
    min_title_count = config["features"]["min_title_count"]

    train = train.copy()
    test = test.copy()

    train["HasCabin"] = train["Cabin"].notna().astype(int)
    test["HasCabin"] = test["Cabin"].notna().astype(int)

    embarked_mode = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(embarked_mode)
    test["Embarked"] = test["Embarked"].fillna(embarked_mode)

    mean_age = train["Age"].mean()
    train["Age"] = train["Age"].fillna(mean_age)
    test["Age"] = test["Age"].fillna(mean_age)

    train["Title"] = train["Name"].str.extract(r",\s*([^\.]+)\.")
    title_counts = train["Title"].value_counts()
    popular_titles = title_counts[title_counts > min_title_count].index
    train["Title"] = train["Title"].where(train["Title"].isin(popular_titles), "Other")

    test["Title"] = test["Name"].str.extract(r",\s*([^\.]+)\.")
    test["Title"] = test["Title"].where(test["Title"].isin(popular_titles), "Other")

    train = train.drop(columns=["Name", "Ticket", "Cabin"])
    test = test.drop(columns=["Name", "Ticket", "Cabin"])

    median_fare = train["Fare"].median()
    train["Fare"] = np.log1p(train["Fare"].fillna(median_fare))
    test["Fare"] = np.log1p(test["Fare"].fillna(median_fare))

    X = train.drop(columns=target)
    y = train[target]
    return X, y, test
