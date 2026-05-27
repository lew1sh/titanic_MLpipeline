import pandas as pd


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data with PassengerId as index."""
    train = pd.read_csv(config["data"]["train_path"], index_col="PassengerId")
    test = pd.read_csv(config["data"]["test_path"], index_col="PassengerId")
    return train, test
