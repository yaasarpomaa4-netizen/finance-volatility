"""Paired forecast scores, preserving negative predictions and bias sign."""
import numpy as np
import pandas as pd

def score_predictions(frame, methods, target='actual', references=('ewm_20',), reference_labels=None):
    """Return one row per method: MAE/RMSE/Bias and benchmark improvements.

    Every method uses the supplied shared rows. Bias is forecast minus actual.
    Predictions are scored unchanged. Returns columns, not a method index.
    The caller controls ordering/indexing to match its original notebook.
    """
    assert len(frame) > 0, 'No aligned forecasts to score.'
    assert np.isfinite(frame[[target, *methods]].to_numpy()).all()
    rows = []
    for method in methods:
        error = frame[method] - frame[target]
        rows.append({'method': method, 'n_forecasts': len(frame),
                     'MAE': error.abs().mean(),
                     'RMSE': np.sqrt((error ** 2).mean()),
                     'Bias': error.mean(),
                     'negative_forecasts': int((frame[method] < 0).sum())})
    scores = pd.DataFrame(rows)
    reference_labels = reference_labels or {}
    for reference in references:
        for metric in ['MAE', 'RMSE']:
            denominator = scores.set_index('method').loc[reference, metric]
            assert denominator > 0
            label = reference_labels.get(reference, reference)
            scores[f'{metric}_improvement_vs_{label}_pct'] = 100 * (
                1 - scores[metric] / denominator)
    return scores
