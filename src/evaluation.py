import numpy as np
import pandas as pd


def summarize_scores(scores: dict[str, list[float]]) -> pd.DataFrame:
    """Convert fold scores to a sorted results table."""
    return pd.DataFrame(
        {
            "model": scores.keys(),
            "mean_accuracy": [np.mean(values) for values in scores.values()],
            "std_accuracy": [np.std(values) for values in scores.values()],
        }
    ).sort_values("mean_accuracy", ascending=False)
