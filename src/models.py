import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.evaluation import summarize_scores
from src.preprocessing import encode_categoricals


def _with_random_state(params: dict, random_state: int) -> dict:
    """Add random_state to model params when the estimator supports it."""
    return {**params, "random_state": random_state}


def _tuple_params(params: dict, keys: list[str]) -> dict:
    """Convert YAML lists to tuples for sklearn parameters that expect tuples."""
    prepared = params.copy()
    for key in keys:
        if isinstance(prepared.get(key), list):
            prepared[key] = tuple(prepared[key])
    return prepared


def build_model_specs(config: dict) -> dict[str, dict]:
    """Create model objects and preprocessing type from config parameters."""
    params = config["model_params"]
    random_state = config["project"]["random_state"]

    voting_rf = _with_random_state(params["voting_rf"], random_state)
    stacking_rf = _with_random_state(params["stacking_rf"], random_state)
    stacking_tree = _with_random_state(params["stacking_tree"], random_state)

    base_estimators = [
        ("logreg", LogisticRegression(max_iter=1000)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(**voting_rf)),
    ]
    stack_estimators = [
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(**stacking_rf)),
        ("tree", DecisionTreeClassifier(**stacking_tree)),
    ]

    return {
        "logreg_none": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": LogisticRegression(**params["logreg_none"]),
        },
        "logreg_l1": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": LogisticRegression(**params["logreg_l1"]),
        },
        "logreg_l2": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": LogisticRegression(**params["logreg_l2"]),
        },
        "logreg_elastic": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": LogisticRegression(**_with_random_state(params["logreg_elastic"], random_state)),
        },
        "knn_3": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": KNeighborsClassifier(**params["knn_3"]),
        },
        "knn_5": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": KNeighborsClassifier(**params["knn_5"]),
        },
        "knn_7_distance": {
            "group": "baselines",
            "preprocessing": "scaled",
            "model": KNeighborsClassifier(**params["knn_7_distance"]),
        },
        "tree_depth_3": {
            "group": "baselines",
            "preprocessing": "encoded",
            "model": DecisionTreeClassifier(**_with_random_state(params["tree_depth_3"], random_state)),
        },
        "tree_depth_5": {
            "group": "baselines",
            "preprocessing": "encoded",
            "model": DecisionTreeClassifier(**_with_random_state(params["tree_depth_5"], random_state)),
        },
        "rf_100": {
            "group": "baselines",
            "preprocessing": "encoded",
            "model": RandomForestClassifier(**_with_random_state(params["rf_100"], random_state)),
        },
        "rf_300_depth_5": {
            "group": "baselines",
            "preprocessing": "encoded",
            "model": RandomForestClassifier(**_with_random_state(params["rf_300_depth_5"], random_state)),
        },
        "catboost": {
            "group": "boosting",
            "preprocessing": "raw_catboost",
            "model": CatBoostClassifier(**_with_random_state(params["catboost"], random_state)),
        },
        "lightgbm": {
            "group": "boosting",
            "preprocessing": "encoded",
            "model": LGBMClassifier(**_with_random_state(params["lightgbm"], random_state)),
        },
        "xgboost": {
            "group": "boosting",
            "preprocessing": "encoded",
            "model": XGBClassifier(**_with_random_state(params["xgboost"], random_state)),
        },
        "voting_hard": {
            "group": "ensembles",
            "preprocessing": "scaled",
            "model": VotingClassifier(base_estimators, voting="hard"),
        },
        "voting_soft": {
            "group": "ensembles",
            "preprocessing": "scaled",
            "model": VotingClassifier(base_estimators, voting="soft"),
        },
        "stacking_logreg": {
            "group": "ensembles",
            "preprocessing": "scaled",
            "model": StackingClassifier(
                estimators=stack_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
            ),
        },
        "stacking_ridge": {
            "group": "ensembles",
            "preprocessing": "scaled",
            "model": StackingClassifier(
                estimators=stack_estimators,
                final_estimator=RidgeClassifier(),
            ),
        },
        "mlp": {
            "group": "mlp",
            "preprocessing": "scaled",
            "model": MLPClassifier(
                **_with_random_state(_tuple_params(params["mlp"], ["hidden_layer_sizes"]), random_state)
            ),
        },
    }


def _evaluate_specs(
    specs: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Evaluate model specs with the preprocessing each model requires."""
    scores = {name: [] for name in specs}
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    for train_idx, valid_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_train_encoded, X_valid_encoded = encode_categoricals(X_train, X_valid)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_valid_scaled = scaler.transform(X_valid_encoded)

        for name, spec in specs.items():
            model = clone(spec["model"])

            if spec["preprocessing"] == "raw_catboost":
                model.fit(X_train, y_train, cat_features=cat_cols)
                predictions = model.predict(X_valid)
            elif spec["preprocessing"] == "scaled":
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_valid_scaled)
            else:
                model.fit(X_train_encoded, y_train)
                predictions = model.predict(X_valid_encoded)

            scores[name].append(accuracy_score(y_valid, predictions))

    return summarize_scores(scores)


def evaluate_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    config: dict,
) -> pd.DataFrame:
    """Evaluate simple baseline models and tree models."""
    specs = {
        name: spec
        for name, spec in build_model_specs(config).items()
        if spec["group"] == "baselines"
    }
    return _evaluate_specs(specs, X, y, cv)


def evaluate_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    config: dict,
) -> pd.DataFrame:
    """Evaluate CatBoost, LightGBM, and XGBoost."""
    specs = {
        name: spec
        for name, spec in build_model_specs(config).items()
        if spec["group"] == "boosting"
    }
    return _evaluate_specs(specs, X, y, cv)


def evaluate_ensembles(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    config: dict,
) -> pd.DataFrame:
    """Evaluate voting and stacking ensembles."""
    specs = {
        name: spec
        for name, spec in build_model_specs(config).items()
        if spec["group"] == "ensembles"
    }
    return _evaluate_specs(specs, X, y, cv)


def evaluate_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    config: dict,
) -> pd.DataFrame:
    """Evaluate a small neural-network baseline for tabular data."""
    specs = {
        name: spec
        for name, spec in build_model_specs(config).items()
        if spec["group"] == "mlp"
    }
    return _evaluate_specs(specs, X, y, cv)
