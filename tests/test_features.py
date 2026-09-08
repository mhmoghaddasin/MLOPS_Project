import pandas as pd

from src.features.build_features import scale_features, select_features


def test_feature_selection_shape():
    X_train = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 3, 4, 5, 6, 7],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )
    X_test = pd.DataFrame({"a": [7, 8], "b": [8, 9], "c": [0, 1]})
    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
        method="standard",
    )
    X_train_selected, X_test_selected, selector = select_features(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=2,
    )

    assert scaler is not None
    assert selector is not None
    assert X_train_selected.shape == (6, 2)
    assert X_test_selected.shape == (2, 2)
