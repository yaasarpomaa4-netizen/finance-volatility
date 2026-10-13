"""Training-only scaling and unchanged estimator fitting."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def scaled_model(estimator):
    """Wrap an estimator so scaling is fitted inside each training fold."""
    return Pipeline([('scaler', StandardScaler()), ('model', estimator)])

def fit_model(model, frame, feature_names, target_name):
    """Fit the supplied estimator on caller-selected, already audited rows.

    The notebook owns the temporal split and frozen parameters. This function
    does not choose rows, tune hyperparameters, or inspect evaluation outcomes.
    It returns the same fitted estimator.
    """
    return model.fit(frame[feature_names], frame[target_name])
