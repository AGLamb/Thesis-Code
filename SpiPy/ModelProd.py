from __future__ import annotations
from typing import Any
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResults
from SpiPy.Backbone import part1, part2, part3, part4
from statsmodels.tsa.api import VAR, AutoReg
import numpy as np


np.random.seed(123)


def random_walk(sigma: float, df: int, pollution_data: pd.DataFrame, geo_level: int = "municipality") -> np.array:
    """
    :param geo_level:
    :param sigma:
    :param df:
    :param pollution_data:
    :return:
    """

    t = len(pollution_data)
    k = len(pollution_data.columns)

    if geo_level == "street":
        eps = np.random.standard_t(df, size=(t, k+1)) * sigma
        data = np.zeros((t, k+1))
        data[0, :] = eps[0, :]

        for i in range(1, t):
            data[i, :] = data[i - 1, :] + eps[i, :]

    else:
        eps = np.random.standard_t(df, size=(t, k)) * sigma
        data = np.zeros((t, k))
        data[0, :] = eps[0, :]

        for i in range(1, t):
            data[i, :] = data[i - 1, :] + eps[i, :]

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

    clean_df = part1(filepath=path, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = part2(geo_lev=geo_lev, time_lev=time_lev)
    wind_spillover, space_spillover, w_matrix, ww_tensor = part3(clean_df, pollution, w_speed, w_angle,
                                                                 geo_lev=geo_lev, time_lev=time_lev)

    return {"SWVAR(1) Model": part4(wind_spillover),
            "SVAR(1) Model": part4(space_spillover),
            "VAR(1) Model": VAR(pollution).fit(maxlags=1, trend='c'),
            "AR(1) Models": ar_model(lags=1, pollution_data=pollution),
            "Random Walk": random_walk(sigma=4, df=5, pollution_data=pollution, geo_level=geo_lev)}
