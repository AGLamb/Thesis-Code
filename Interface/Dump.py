import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from mcs import mcs

# Load data
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Estimate VAR model
model_var = VAR(data)
results_var = model_var.fit()

# Estimate AR models for each variable
results_ar = {}
for col in data.columns:
    model_ar = AR(data[col])
    results_ar[col] = model_ar.fit()

# Compute forecast errors for each model
errors_var = results_var.forecast_error()
errors_ar = {}
for col in data.columns:
    errors_ar[col] = results_ar[col].forecast_error()

# Compute test statistics for each model
test_stats_var = np.mean(errors_var**2, axis=0)
test_stats_ar = {}
for col in data.columns:
    test_stats_ar[col] = np.mean(errors_ar[col]**2)

# Compute p-values using bootstrap procedure
p_values_var = mcs.bootstrap_pvalues(errors_var, test_stats_var)
p_values_ar = {}
for col in data.columns:
    p_values_ar[col] = mcs.bootstrap_pvalues(errors_ar[col], test_stats_ar[col])

# Compute MCS
mcs_var = mcs.mcs(test_stats_var, p_values_var, 0.05)
mcs_ar = {}
for col in data.columns:
    mcs_ar[col] = mcs.mcs(test_stats_ar[col], p_values_ar[col], 0.05)