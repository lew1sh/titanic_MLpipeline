from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.models import build_model_specs
from src.preprocessing import encode_categoricals


def save_best_submission(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    config: dict,
) -> Path:
    """Train the best CV model on full data and save a single submission file."""
    specs = build_model_specs(config)
    if model_name not in specs:
        raise ValueError(f"Unknown model for submission: {model_name}")

    spec = specs[model_name]
    model = clone(spec["model"])

    if spec["preprocessing"] == "raw_catboost":
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        model.fit(X, y, cat_features=cat_cols)
        predictions = model.predict(X_test)
    elif spec["preprocessing"] == "scaled":
        X_encoded, X_test_encoded = encode_categoricals(X, X_test)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        model.fit(X_scaled, y)
        predictions = model.predict(X_test_scaled)
    else:
        X_encoded, X_test_encoded = encode_categoricals(X, X_test)
        model.fit(X_encoded, y)
        predictions = model.predict(X_test_encoded)

    output_dir = Path(config["data"]["submission_dir"])
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"submission_{model_name}.csv"

    submission = pd.DataFrame(
        {
            "PassengerId": X_test.index,
            "Survived": predictions.astype(int),
        }
    )
    submission.to_csv(output_path, index=False)
    return output_path
