import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def scale_features(X_train, X_test, method="standard"):
    """Scale features using the specified method."""
    method = (method or "none").lower()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "none":
        return X_train.copy(), X_test.copy(), None
    else:
        raise ValueError('method must be "standard", "minmax", or "none"')

    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="pandas")

    train_values = scaler.fit_transform(X_train)
    test_values = scaler.transform(X_test)

    X_train_scaled = (
        train_values.copy()
        if isinstance(train_values, pd.DataFrame)
        else pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index)
    )
    X_test_scaled = (
        test_values.copy()
        if isinstance(test_values, pd.DataFrame)
        else pd.DataFrame(test_values, columns=X_test.columns, index=X_test.index)
    )
    X_train_scaled.index = X_train.index
    X_test_scaled.index = X_test.index
    return X_train_scaled, X_test_scaled, scaler


def select_features(X_train, y_train, X_test, k=10):
    """Select top-k features using ANOVA F-value."""
    if k in (None, "all"):
        k = "all"
    else:
        k = min(int(k), X_train.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)
    if hasattr(selector, "set_output"):
        selector.set_output(transform="pandas")

    train_values = selector.fit_transform(X_train, y_train)
    test_values = selector.transform(X_test)

    selected_columns = X_train.columns[selector.get_support()].tolist()
    X_train_selected = (
        train_values.copy()
        if isinstance(train_values, pd.DataFrame)
        else pd.DataFrame(train_values, columns=selected_columns, index=X_train.index)
    )
    X_test_selected = (
        test_values.copy()
        if isinstance(test_values, pd.DataFrame)
        else pd.DataFrame(test_values, columns=selected_columns, index=X_test.index)
    )
    X_train_selected.columns = selected_columns
    X_test_selected.columns = selected_columns
    X_train_selected.index = X_train.index
    X_test_selected.index = X_test.index
    return X_train_selected, X_test_selected, selector
