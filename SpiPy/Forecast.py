from __future__ import annotations
from typing import Any
from numpy import ndarray
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VARResults
from SpiPy.Backbone import Part1, Part2, Part3
import pandas as pd
import numpy as np


def get_performance(input_set: Any, geo_lev: str, time_lev: str) -> dict[str, ndarray | Any]:
    path = "/Users/main/Vault/Thesis/Data/Core/test_data.csv"

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    pollution, weight_matrix, W_tensor = get_test_data(path, geo_lev=geo_lev, time_lev=time_lev)

    rw_errors = random_walk_forecast(rw_data=input_set["Random Walk"], sigma=1,
                                     geo_lev=geo_lev, forecast=forecast_steps, pollution=pollution)

    ar_errors = AR_forecast(ar_set=input_set["AR(1) Models"], pollution=pollution, forecast=forecast_steps)

    VAR_errors = VAR_forecast(VAR_mod=input_set["VAR(1) Model"], pollution=pollution, forecast=forecast_steps)

    SVAR_errors = SVAR_forecast(VAR_mod=input_set["SVAR(1) Model"], pollution=pollution,
                                forecast=forecast_steps, W_matrix=weight_matrix)

    SWVAR_errors = SWVAR_forecast(VAR_mod=input_set["SWVAR(1) Model"], pollution=pollution,
                                  forecast=forecast_steps, WW_tensor=W_tensor)

    return {"Random Walk": rw_errors,
            "AR(1) Models": ar_errors,
            "VAR(1) Model": VAR_errors,
            "SVAR(1) Model": SVAR_errors,
            "SWVAR(1) Model": SWVAR_errors}


def random_walk_forecast(rw_data, sigma: float, geo_lev: str, forecast: int, pollution: pd.DataFrame) -> np.ndarray:

    if geo_lev == "street":
        df = 170
    else:
        df = 14

    T = forecast
    K = rw_data.shape[1]

    output_matrix = np.zeros((T, K))
    data = np.zeros((T, K))

    eps = np.random.standard_t(df, size=(T, K)) * sigma
    data[0, :] = rw_data[-1, :]

    for t in range(1, T):
        data[t, :] = data[t - 1, :] + eps[t, :]

    for i in range(K):
        output_matrix[:, i] =  np.abs(pollution.iloc[:T, i].to_numpy() - data[:, i])

    return output_matrix


def AR_forecast(ar_set: dict, pollution: pd.DataFrame, forecast: int) -> np.ndarray:
    output_matrix = np.zeros((forecast, len(pollution.columns)))

    for i in range(len(pollution.columns)):
        pred = ar_set[pollution.columns[i]].forecast(steps=forecast)
        output_matrix[:, i] =  np.abs(pollution.iloc[:forecast, i].to_numpy() - pred)

    return output_matrix


def VAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, forecast: int):
    pred = VAR_mod.forecast(y=VAR_mod.endog, steps=forecast)
    return np.abs(pollution.iloc[:forecast, :].to_numpy() - pred)


def SVAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, W_matrix: pd.DataFrame,
                  forecast: int) -> Any:

    pred = np.zeros((forecast, len(pollution.columns)))

    C = np.ones((1, 1))
    print(VAR_mod.endog[-1, :])
    X = np.concatenate([C, (W_matrix @  VAR_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (VAR_mod.params.T @ X).T

    for i in range(1, pred.shape[0]):
        X = np.concatenate([C, (W_matrix @  pred[i-1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (VAR_mod.params.T @ X).T

    return np.abs(pollution.iloc[:forecast, :].to_numpy() - pred)


def SWVAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, WW_tensor: pd.DataFrame,
                   forecast: int) -> Any:

    pred = np.zeros((forecast, len(pollution.columns)))

    C = np.ones((1, 1))
    X = np.concatenate([C, (WW_tensor[0, :, :] @ VAR_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (VAR_mod.params.T @ X).T

    for i in range(1, pred.shape[0]):
        X = np.concatenate([C, (WW_tensor[i, :, :] @ pred[i - 1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (VAR_mod.params.T @ X).T

    return np.abs(pollution.iloc[:forecast, :].to_numpy() - pred)


def get_test_data(filepath: str, geo_lev: str, time_lev: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_df = Part1(filepath, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    wind_spillover, space_spillover, W_array, WW_tensor = Part3(clean_df, pollution, w_speed, w_angle,
                                                                geo_lev=geo_lev, time_lev=time_lev)
    return pollution, W_array, WW_tensor




