import numpy as np
import pandas as pd
from statsmodels.tsa.api import AutoReg
np.random.seed(123)


def random_walk(T: int, K: int, df: int, sigma=1) -> np.array:
    eps = np.random.standard_t(df, size=(T, K)) * sigma
    data = np.zeros((T, K))
    data[0, :] = np.random.normal(0, sigma, size=K)

    for t in range(1, T):
        data[t, :] = data[t - 1, :] + eps[t, :]

    return data


def AR_model(data: pd.DataFrame, lags: int) -> list:
    output_models = []
    for column in data:
        mod = AutoReg(data[column], lags=lags).fit()
        output_models.append(mod)
    return output_models

