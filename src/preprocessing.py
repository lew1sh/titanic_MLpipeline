import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_categoricals(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical columns without leaking validation data."""
    cat_cols = X_train.select_dtypes(include="object").columns

    encoder = OneHotEncoder(
        sparse_output=False,
        drop="first",
        handle_unknown="ignore",
    )

    encoded_train = encoder.fit_transform(X_train[cat_cols])
    encoded_valid = encoder.transform(X_valid[cat_cols])

    encoded_train = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index,
    )
    encoded_valid = pd.DataFrame(
        encoded_valid,
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_valid.index,
    )

    X_train_encoded = pd.concat([X_train.drop(columns=cat_cols), encoded_train], axis=1)
    X_valid_encoded = pd.concat([X_valid.drop(columns=cat_cols), encoded_valid], axis=1)
    return X_train_encoded, X_valid_encoded
