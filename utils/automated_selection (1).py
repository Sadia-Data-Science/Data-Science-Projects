import numpy as np
import statsmodels.api as sm

def backward_elimination(X, y, significance_level=0.05, col_names=None):
    """
    Perform backward elimination on a set of features using statsmodels OLS.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (independent variables), without a constant column.
    y : array-like, shape (n_samples,)
        Target vector (dependent variable).
    significance_level : float, optional
        The alpha (p-value) threshold above which a feature will be removed.
    col_names : list of str, optional
        Column names for the features in X. If None, columns will be labeled
        as x0, x1, x2, etc.

    Returns
    -------
    X_modeled : ndarray, shape (n_samples, k)
        The feature matrix after removing insignificant features.
        A constant (intercept) column is included as the first column.
    selected_features : list of str
        Names of the features that remain after backward elimination
        (includes the "Intercept" at index 0).
    removed_features : list of str
        Names of the features that were removed during elimination.
    """

    # Convert X to float (statsmodels OLS expects float64 / numeric data)
    X_modeled = np.array(X, dtype=float)

    # Add a column of ones to the left (for the intercept)
    X_modeled = np.append(
        arr=np.ones((X_modeled.shape[0], 1)).astype(float),
        values=X_modeled,
        axis=1
    )

    # Generate default column names if none are provided
    if col_names is None:
        col_names = [f"x{i}" for i in range(X_modeled.shape[1] - 1)]
    # Insert "Intercept" at the front
    col_names = ["Intercept"] + list(col_names)

    removed_features = []

    # Iteratively remove the feature with the highest p-value
    while True:
        # Fit the OLS model
        model = sm.OLS(y, X_modeled).fit()
        p_values = model.pvalues

        # Find the max p-value
        max_p = p_values.max()
        max_idx = p_values.argmax()

        # Check if max p-value is above our significance level
        if max_p > significance_level:
            # Remove that feature from X_modeled
            feature_to_remove = col_names.pop(max_idx)
            removed_features.append(feature_to_remove)
            X_modeled = np.delete(X_modeled, max_idx, axis=1)
        else:
            # All remaining features are below significance_level; stop
            break

    return X_modeled, col_names, removed_features
