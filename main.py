from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.data import load_data
from src.features import prepare_features
from src.models import (
    evaluate_baselines,
    evaluate_boosting,
    evaluate_ensembles,
    evaluate_mlp,
)
from src.submission import save_best_submission


def main() -> None:
    """Run the full Titanic training pipeline."""
    config = load_config()
    Path("outputs").mkdir(exist_ok=True)

    train, test = load_data(config)
    X, y, X_test = prepare_features(train, test, config)

    cv = StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=config["validation"]["shuffle"],
        random_state=config["project"]["random_state"],
    )

    result_tables = []
    if config["models"]["baselines"]["enabled"]:
        result_tables.append(evaluate_baselines(X, y, cv, config))
    if config["models"]["boosting"]["enabled"]:
        result_tables.append(evaluate_boosting(X, y, cv, config))
    if config["models"]["ensembles"]["enabled"]:
        result_tables.append(evaluate_ensembles(X, y, cv, config))
    if config["models"]["mlp"]["enabled"]:
        result_tables.append(evaluate_mlp(X, y, cv, config))

    results = pd.concat(result_tables, ignore_index=True).sort_values(
        "mean_accuracy",
        ascending=False,
    )
    results.to_csv(config["data"]["results_path"], index=False)
    print(results.to_string(index=False))

    best_model_name = results.iloc[0]["model"]
    submission_path = save_best_submission(best_model_name, X, y, X_test, config)
    print(f"\nBest model: {best_model_name}")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
