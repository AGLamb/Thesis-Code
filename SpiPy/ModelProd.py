from __future__ import annotations
from typing import Any
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResults
from statsmodels.tsa.vector_ar.var_model import VARResults
from SpiPy.Backbone import Part1, Part2, Part3, Part4
from statsmodels.tsa.api import VAR, AutoReg
import numpy as np


np.random.seed(123)


def random_walk(sigma: float, geo_lev: str, pollution_data: pd.DataFrame) -> np.array:
    """
    :param sigma:
    :param geo_lev:
    :param pollution_data:
    :return:
    """

    if geo_lev == "street":
        df = 170
    else:
        df = 14

    T = len(pollution_data)
    K = len(pollution_data.columns)

    eps = np.random.standard_t(df, size=(T, K)) * sigma
    data = np.zeros((T, K))
    data[0, :] = np.random.normal(0, sigma, size=K)

    for t in range(1, T):
        data[t, :] = data[t - 1, :] + eps[t, :]

    return data


def ar_model(lags: int, pollution_data: pd.DataFrame) -> dict[Any, AutoRegResults]:
    """
    :param lags:
    :param pollution_data:
    :return:
    """

    output_models = {}
    for column in pollution_data:
        output_models[column] = AutoReg(pollution_data[column], lags=lags, trend='c').fit()

    return output_models


def create_set(geo_lev: str, time_lev: str) -> dict[str, VAR | list | Any]:
    """
    :param geo_lev:
    :param time_lev:
    :return:
    """
    path = r"/Users/main/Vault/Thesis/Data/Core/train_data.csv"

    clean_df = Part1(filepath=path, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    wind_spillover, space_spillover, W_matrix, WW_tensor = Part3(clean_df, pollution, w_speed, w_angle,
                                                                 geo_lev=geo_lev, time_lev=time_lev)

    return {"SWVAR(1) Model": Part4(wind_spillover),
            "SVAR(1) Model": Part4(space_spillover),
            "VAR(1) Model": VAR(pollution).fit(maxlags=1, trend='c'),
            "AR(1) Models": ar_model(lags=1, pollution_data=pollution),
            "Random Walk": random_walk(sigma=1, geo_lev=geo_lev, pollution_data=pollution)}
