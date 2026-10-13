# Forecasting Next-Day S&P 500 Return Magnitude

A research project comparing seven statistical, machine-learning, and benchmark methods for forecasting the size of the next trading day's S&P 500 movement.

**Author:** Yaa Sarpomaa Sarpong\
**Affiliation:** University of Memphis, Mathematical Sciences — Applied Statistics

## Overview

Financial markets move by different amounts each day. Forecasting the size of these movements may help with risk assessment and market monitoring, even when their direction is unknown.

This project forecasts the next trading day's absolute S&P 500 log return using past returns, VIX, and changes in the 10-year Treasury yield. It compares linear regression, ridge, lasso, and random forest with historical-mean and exponentially weighted-mean benchmarks.

Extended linear regression had the lowest MAE and RMSE in the 2025 test period. Its errors were 4.52% and 7.22% lower, respectively, than the exponentially weighted mean (EWM) benchmark. Random forest performed best during 2023–2024 validation.

The main finding is that the leading method changed across evaluation periods and specifications. Greater model complexity did not consistently improve accuracy.

## Research questions

- Which of the selected methods produces the most accurate next-day return-magnitude forecasts?
- Do additional predictors and more flexible models improve on simple benchmarks?
- How stable are the results across evaluation periods, training windows, predictor sets, and forecast targets?

## Data and prediction target

The analysis uses daily observations from the Federal Reserve Bank of St. Louis's FRED database.

| Input | FRED series | CSV column |
|------------------------|------------------------|------------------------|
| S&P 500 closing price-index level | [SP500](https://fred.stlouisfed.org/series/SP500) | `sp500` |
| CBOE VIX closing level | [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | `vix` |
| 10-year Treasury constant-maturity yield | [DGS10](https://fred.stlouisfed.org/series/DGS10) | `treasury_10yr` |

The saved input file is `data/financial_raw_data.csv`, with a `date` column and the three columns above. Exact reproduction requires the original saved observations.

For S&P 500 closing level $P_t$, the daily percentage log return is:

$$
r_t = 100\ln\left(\frac{P_t}{P_{t-1}}\right).
$$

The forecast target is:

$$
y_{t+1} = |r_{t+1}|.
$$

This target measures movement size regardless of direction. It is a noisy indicator of volatility, rather than a direct measurement of underlying conditional volatility.

Rows without an S&P 500 level are removed before calculating returns. Features and targets are then constructed along the retained S&P observation sequence. Rows missing required model inputs are excluded afterward, without imputation. All methods in each comparison are evaluated on common forecast dates.

## Models and predictors

| Method | Specification |
|------------------------------------|------------------------------------|
| Historical mean | Expanding average of observed absolute returns |
| EWM benchmark | Exponentially weighted average; span 20, `adjust=False` |
| Extended linear regression | Ordinary least squares with five predictors |
| Expanded linear regression | Ordinary least squares with twelve predictors |
| Ridge | Twelve predictors with L2 regularization |
| Lasso | Twelve predictors with L1 regularization |
| Random forest | Twelve predictors with nonlinear relationships and interactions |

Both benchmarks require at least 20 observations and update as new returns become available.

### Five-predictor specification

The extended regression uses:

| Predictor | Meaning |
|------------------------------------|------------------------------------|
| `absolute_return` | Today's absolute log return |
| `mean_abs_5` | Mean absolute return over the latest five trading observations |
| `mean_abs_20` | Mean absolute return over the latest twenty trading observations |
| `vix` | Today's observed VIX level |
| `treasury_change` | Change in the 10-year Treasury yield since the preceding S&P observation |

### Twelve-predictor specification

Expanded regression, ridge, lasso, and random forest add seven predictors:

| Predictor | Meaning |
|------------------------------------|------------------------------------|
| `log_return` | Today's signed log return |
| `return_lag_1` | The preceding observation's signed return |
| `abs_lag_1` | The preceding observation's absolute return |
| `abs_lag_5` | Absolute return five trading observations earlier |
| `squared_return` | Today's signed return squared |
| `return_std_5` | Sample standard deviation of returns over the latest five observations |
| `return_std_20` | Sample standard deviation of returns over the latest twenty observations |

Rolling windows include the forecast-origin observation. Standard deviations use `ddof=1` and are not annualized.

The comparison evaluates complete model specifications. Because extended regression uses five predictors and the other fitted methods use twelve, its advantage does not isolate the effect of the modeling method alone.

## Evaluation design

Observations are split chronologically by outcome date.

| Stage      | Period                      | Eligible observations |
|------------|-----------------------------|----------------------:|
| Training   | November 2016–December 2022 |                 1,528 |
| Validation | January 2023–December 2024  |                   496 |
| Final test | January–December 2025       |                   246 |

Hyperparameters are selected using five expanding chronological folds within the training period. Each evaluation fold contains 126 retained observations, with a one-row gap between training and evaluation. Predictor scaling is fitted using only the relevant training data.

Random forest was the primary candidate selected from validation, with lasso as a simpler competitor. The fitted models remain trained through 2022 during both validation and testing. Predictors and benchmarks update as new observations become available, but fitted parameters are not regularly re-estimated. The 2025 test uses the saved specifications without additional tuning.

Forecast accuracy is measured using:

- **MAE:** mean absolute error.
- **RMSE:** root mean squared error.
- **Bias:** mean forecast minus actual; positive values indicate average overprediction.

Errors are reported in percentage points. Lower MAE and RMSE indicate greater accuracy. Relative improvement over EWM is calculated as `100 × (1 − model error / EWM error)`.

## Main results

Values are rounded to four decimal places. Bold values mark the lowest error in each column.

| Method              | Validation MAE | Validation RMSE |   Test MAE |  Test RMSE |
|---------------------|---------------:|----------------:|-----------:|-----------:|
| Historical mean     |         0.4591 |          0.5471 |     0.5463 |     0.9198 |
| EWM benchmark       |         0.4229 |          0.5301 |     0.5153 |     0.8717 |
| Extended regression |         0.4153 |          0.5261 | **0.4920** | **0.8088** |
| Expanded regression |         0.4159 |          0.5281 |     0.5036 |     0.8565 |
| Ridge               |         0.4189 |          0.5218 |     0.5019 |     0.8580 |
| Lasso               |         0.4127 |          0.5182 |     0.4930 |     0.8371 |
| Random forest       |     **0.4065** |      **0.5180** |     0.5028 |     0.8526 |

Random forest reduced validation MAE by 3.89% and RMSE by 2.28% relative to EWM. Lasso's validation RMSE was nearly equal to random forest's.

Extended regression reduced test MAE by 4.52% and RMSE by 7.22% relative to EWM. Lasso remained close on test MAE. Expanded regression performed worse than extended regression in both periods.

These are observed rankings. The uncertainty in the error differences has not been assessed, so the results do not establish statistical superiority.

## Exploratory robustness checks

The analysis also examines shorter training windows, removal of selected predictors, and alternative forecast targets.

The table shows validation RMSE reductions relative to EWM. Positive values indicate improvement; negative values indicate higher error.

| Scenario | Extended regression | Lasso | Random forest |
|-----------------|-----------------:|-----------------:|-----------------:|
| Original setup | 0.75% | 2.24% | 2.28% |
| Train on 2018–2022 | 0.27% | 2.00% | 2.74% |
| Train on 2020–2022 | −3.59% | −0.35% | 2.69% |
| Remove VIX | −3.46% | 0.13% | −0.47% |
| Remove Treasury changes | 2.13% | 2.51% | 2.44% |
| Return history only | −1.39% | 0.52% | −0.46% |
| Fifth trading day's magnitude | 0.96% | 1.46% | 1.42% |
| Average magnitude over the next five trading days | 4.80% | 9.49% | 11.09% |

Removing VIX weakened the three fitted models examined. Removing Treasury changes slightly improved their validation performance. These findings concern predictive usefulness within the selected specifications and do not establish causal relationships.

Random forest remained competitive with shorter training windows. Larger improvements for the five-day average suggest that a smoother outcome may be easier to forecast. However, that outcome answers a different question from the primary next-day target; its raw errors should not be compared directly with next-day errors.

Quarterly comparisons also identified different leading methods within 2025. These checks are exploratory and use data already examined in the project.

## Project files

| Location | Contents |
|------------------------------------|------------------------------------|
| `data/financial_raw_data.csv` | Original input observations |
| `notebooks/` | Analysis notebooks and optional refactored copies |
| `src/finance_project/` | Shared functions for features, models, evaluation, and plots |
| `results/` | Predictions, error tables, tuning records, saved specifications, and audits |
| `figures/` | Model comparisons, actual-versus-forecast plots, and robustness figures |
| `requirements.txt` | Recorded package versions |
| `environment_info.json` | Recorded Python version and environment details |
| `README.md` | Project overview and reproduction instructions |

The written report and presentation can be included in a `reports/` folder when preparing the repository.

## Running the analysis

The analysis uses Python, NumPy, pandas, scikit-learn, Matplotlib, and joblib. Use the Python version recorded in `environment_info.json` and install the packages listed in `requirements.txt`.

1.  Download or clone the project and open its root folder.

2.  Activate a Python environment compatible with the recorded environment.

3.  Install the dependencies:

    ``` bash
    python -m pip install -r requirements.txt
    ```

4.  Select the same environment as the notebook kernel in Jupyter or VS Code.

5.  Place the original CSV at `data/financial_raw_data.csv`.

6.  Update `PROJECT_ROOT` in the notebooks to your local project folder.

7.  Run the notebooks below in order, executing their cells from top to bottom.

| Order | Notebook in `notebooks/` | Purpose |
|----------------------------:|----------------------|----------------------|
| 1 | `day13_machine_learning.ipynb` | Build predictors, tune and fit models, and evaluate 2023–2024 validation |
| 2 | `day14_final_test.ipynb` | Reproduce saved validation forecasts and evaluate the fixed specifications on 2025 |
| 3 | `day15_robustness_checks.ipynb` | Examine alternative specifications, training windows, targets, and subperiods |
| 4 | `day16_extract_functions.ipynb` | Organize shared functions and check refactored calculations |
| 5 | `day16_document_environment.ipynb` | Record the project environment |

Day 14 requires the Day 13 configuration, tuning summary, and validation predictions in `results/`. Day 15 also requires the Day 14 frozen specification, test predictions, and run record. Keep the input-fingerprint and forecast-reproduction checks enabled.

Saved hyperparameters and configuration files are the authoritative specifications for reproduction. A fresh data download may not reproduce the original analysis.

### Key outputs

- `results/day13_model_performance.csv`
- `results/day13_validation_predictions.csv`
- `results/day14_test_performance.csv`
- `results/day14_test_predictions.csv`
- `results/day15_robustness_results.csv`
- `results/day15_subperiod_diagnostics.csv`
- `figures/day14_test_forecasts.png`
- `figures/day15_robustness_comparison.png`
- `figures/day15_quarterly_stability.png`

## Limitations and future work

- Error differences remain uncertain. Future analysis should quantify uncertainty while accounting for dependence over time and overlapping five-day targets.
- Large movements are difficult to predict. The displayed forecasts missed many extreme observations, particularly around April 2025.
- Predictor availability is assumed. Historical release timing and availability at the forecasting time have not been verified.
- The models are fixed. The results describe fitting through 2022 and updating predictors afterward. Periodic model re-estimation may produce different results.
- The target measures magnitude. The analysis does not predict direction or demonstrate usefulness for trading, portfolio allocation, or loss prevention.
- Choosing extended regression after viewing the test requires fresh evaluation. Its observed 2025 lead should be assessed on independent data before drawing conclusions about persistence.

Further work should verify predictor availability, compare fixed and periodically re-estimated models, and study errors during extreme movements.

## Related research

The study is informed by research on persistence in absolute returns, heterogeneous autoregressive volatility models, and machine-learning forecasts. Its daily absolute-return target differs from the realized-volatility targets in several of these studies.

- Ding, Granger, and Engle (1993): [DOI](https://doi.org/10.1016/0927-5398(93)90006-D).
- Corsi (2009): [DOI](https://doi.org/10.1093/jjfinec/nbp001).
- Ding, Kambouroudis, and McMillan (2021): [DOI](https://doi.org/10.1016/j.intfin.2021.101386).
- Díaz, Hansen, and Cabrera (2024): [DOI](https://doi.org/10.1016/j.irfa.2024.103286).
- Kilic (2025, revised 2026): [Federal Reserve working paper](https://doi.org/10.17016/FEDS.2025.061r1).
