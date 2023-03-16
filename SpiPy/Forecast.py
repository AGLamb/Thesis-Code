from __future__ import annotations
from typing import Any
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VARResults
from SpiPy.Backbone import part1, part2, part3
import pandas as pd
import numpy as np


def get_performance(input_set: Any, geo_lev: str, time_lev: str) -> DataFrame:
    # path = "/Users/main/Vault/Thesis/Data/Core/test_data.csv"
    path = r"C:\Users\VY72PC\PycharmProjects\Academia\Data\test_data.csv"

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    pollution, weight_matrix, w_tensor = get_test_data(path, geo_lev=geo_lev, time_lev=time_lev)

    rw_errors = random_walk_forecast(rw_data=input_set["Random Walk"], sigma=4, df=5,
                                     forecast=forecast_steps, pollution=pollution)

    ar_errors = ar_forecast(ar_set=input_set["AR(1) Models"], pollution=pollution, forecast=forecast_steps)

    var_errors = var_forecast(var_mod=input_set["VAR(1) Model"], pollution=pollution, forecast=forecast_steps)

    svar_errors = svar_forecast(var_mod=input_set["SVAR(1) Model"], pollution=pollution,
                                forecast=forecast_steps, w_matrix=weight_matrix)

    swvar_errors = swvar_forecast(var_mod=input_set["SWVAR(1) Model"], pollution=pollution,
                                  forecast=forecast_steps, ww_tensor=w_tensor)

    return pd.DataFrame({
            geo_lev + "-" + time_lev + "-Random Walk": rw_errors,
            geo_lev + "-" + time_lev + "-AR(1) Models": ar_errors,
            geo_lev + "-" + time_lev + "-VAR(1) Model": var_errors,
            geo_lev + "-" + time_lev + "-SVAR(1) Model": svar_errors,
            geo_lev + "-" + time_lev + "-SWVAR(1) Model": swvar_errors})


def random_walk_forecast(rw_data, sigma: float, df: int, forecast: int, pollution: pd.DataFrame) -> np.ndarray:
    t = forecast
    k = rw_data.shape[1]

    output_matrix = np.zeros(t)
    data = np.zeros((t, k))

    eps = np.random.standard_t(df, size=(t, k)) * sigma
    data[0, :] = rw_data[-1, :]

    for i in range(1, t):
        data[i, :] = data[i - 1, :] + eps[i, :]

    for i in range(t):
        output_matrix[i] = np.mean(np.abs(pollution.iloc[i, :].to_numpy() - data[i, :]))

    return output_matrix


def ar_forecast(ar_set: dict, pollution: pd.DataFrame, forecast: int) -> np.ndarray:
    output_matrix = np.zeros(forecast)
    pred = np.zeros((forecast, len(pollution.columns)))

    for i in range(len(pollution.columns)):
        pred[:, i] = ar_set[pollution.columns[i]].forecast(steps=forecast)

    for i in range(forecast):
        output_matrix[i] = np.mean(np.abs(pollution.iloc[i, :].to_numpy() - pred[i, :]))

    return output_matrix


def var_forecast(var_mod: VARResults, pollution: pd.DataFrame, forecast: int):
    pred = var_mod.forecast(y=var_mod.endog, steps=forecast)
    output_matrix = np.zeros(forecast)

    for i in range(forecast):
        output_matrix[i] = np.mean(np.abs(pollution.iloc[:forecast, :].to_numpy() - pred))

    return output_matrix


def svar_forecast(var_mod: VARResults, pollution: pd.DataFrame, w_matrix: np.ndarray,
                  forecast: int) -> Any:

    pred = np.zeros((forecast, len(pollution.columns)))
    output_matrix = np.zeros(forecast)

    c = np.ones((1, 1))
    argument = np.concatenate([c, (w_matrix @ var_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (var_mod.params.T @ argument).T
    output_matrix[0] = np.mean(np.abs(pollution.iloc[0, :].to_numpy() - pred[0, :]))

    for i in range(1, pred.shape[0]):
        argument = np.concatenate([c, (w_matrix @ pred[i - 1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (var_mod.params.T @ argument).T
        output_matrix[i] = np.mean(np.abs(pollution.iloc[i, :].to_numpy() - pred[i, :]))

    return output_matrix


def swvar_forecast(var_mod: VARResults, pollution: pd.DataFrame, ww_tensor: np.ndarray,
                   forecast: int) -> Any:

    pred = np.zeros((forecast, len(pollution.columns)))
    output_matrix = np.zeros(forecast)

    c = np.ones((1, 1))
    argument = np.concatenate([c, (ww_tensor[0, :, :] @ var_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (var_mod.params.T @ argument).T
    output_matrix[0] = np.mean(np.abs(pollution.iloc[0, :].to_numpy() - pred[0, :]))

    for i in range(1, pred.shape[0]):
        argument = np.concatenate([c, (ww_tensor[i, :, :] @ pred[i - 1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (var_mod.params.T @ argument).T
        output_matrix[i] = np.mean(np.abs(pollution.iloc[i, :].to_numpy() - pred[i, :]))

    return output_matrix


def get_test_data(filepath: str, geo_lev: str, time_lev: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_df = part1(filepath, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = part2(geo_lev=geo_lev, time_lev=time_lev)
    wind_spillover, space_spillover, w_array, ww_tensor = part3(clean_df, pollution, w_speed, w_angle,
                                                                geo_lev=geo_lev, time_lev=time_lev)
    return pollution, w_array, ww_tensor
