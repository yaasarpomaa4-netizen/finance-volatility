"""Feature and response construction; no model fitting or file writes."""
import numpy as np
import pandas as pd

def build_features(raw_frame, until="2024-12-31"):
    """Build original Day 13–15 features through `until` on observed S&P dates.

    Input is a DataFrame with datetime `date`, `sp500`, `vix`, `treasury_10yr`.
    Returns a new DataFrame. No missing VIX/yield values are filled. Returns
    use percentage points; rolling windows include the forecast origin.
    Future shifts construct response labels only, never predictor values.
    """
    data = raw_frame.copy()
    assert {'date', 'sp500', 'vix', 'treasury_10yr'}.issubset(data.columns)
    assert data['date'].notna().all(), 'Missing/invalid dates.'
    assert not data['date'].duplicated().any(), 'Duplicate dates.'
    data = data.loc[data['date'] <= pd.Timestamp(until)].sort_values('date').reset_index(drop=True)
    base_columns = ['sp500', 'vix', 'treasury_10yr']
    for col in base_columns:
        data[col] = pd.to_numeric(data[col], errors='raise')
    assert not np.isinf(data[base_columns].to_numpy()).any()
    data = data.dropna(subset=['sp500']).reset_index(drop=True)
    assert data['sp500'].gt(0).all()

    data['log_return'] = 100 * np.log(data['sp500']).diff()
    data['absolute_return'] = data['log_return'].abs()
    data['squared_return'] = data['log_return'] ** 2
    data['return_lag_1'] = data['log_return'].shift(1)
    data['abs_lag_1'] = data['absolute_return'].shift(1)
    data['abs_lag_5'] = data['absolute_return'].shift(5)
    for window in [5, 20]:
        data[f'mean_abs_{window}'] = data['absolute_return'].rolling(window, min_periods=window).mean()
        data[f'return_std_{window}'] = data['log_return'].rolling(window, min_periods=window).std(ddof=1)
    data['treasury_change'] = data['treasury_10yr'].diff()
    data['target_next_day'] = data['absolute_return'].shift(-1)
    data['target_date'] = data['date'].shift(-1)
    data['historical_mean'] = data['absolute_return'].expanding(min_periods=20).mean()
    data['ewm_20'] = data['absolute_return'].ewm(span=20, adjust=False, min_periods=20).mean()

    return data

def make_task(features, horizon, target_kind, ml_features, baseline_columns):
    """Align point/mean future targets and retain the original common sample.

    Feature eligibility stays fixed across ablation scenarios, as in Day 15.
    `horizon` counts observed S&P rows. Fit only labels matured by the origin.
    """
    assert isinstance(horizon, int) and horizon >= 1
    frame = features.copy()
    frame['outcome_date'] = frame['date'].shift(-horizon)
    if target_kind == 'point':
        frame['actual'] = frame['absolute_return'].shift(-horizon)
        frame['outcome_start_date'] = frame['outcome_date']
    elif target_kind == 'mean':
        future_returns = pd.concat(
            [frame['absolute_return'].shift(-j) for j in range(1, horizon + 1)], axis=1)
        frame['actual'] = future_returns.mean(axis=1, skipna=False)
        frame['outcome_start_date'] = frame['date'].shift(-1)
    else:
        raise ValueError(target_kind)
    frame = frame.dropna(subset=ml_features + baseline_columns +
                        ['actual', 'outcome_date', 'outcome_start_date']).copy()
    assert (frame['date'] < frame['outcome_start_date']).all()
    assert (frame['outcome_start_date'] <= frame['outcome_date']).all()
    assert np.isfinite(frame[ml_features + baseline_columns + ['actual']].to_numpy()).all()
    return frame
