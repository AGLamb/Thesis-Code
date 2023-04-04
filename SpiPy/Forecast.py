from __future__ import annotations
from typing import Any, Callable
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VARResults
from SpiPy.Backbone import part1, part2, part3
import pandas as pd
import numpy as np


def get_performance(input_set: Any, geo_lev: str, time_lev: str, restricted: bool,
                    func: Callable[[np.ndarray, np.ndarray], float]) -> DataFrame:
    """

    :param input_set:
    :param geo_lev:
    :param restricted:
    :param time_lev:
    :type func: object
    """
    path = "/Users/main/Vault/Thesis/Data/Core/test_data.csv"

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    clean_pollution, weight_matrix, w_tensor = get_test_data(geo_lev=geo_lev, time_lev=time_lev)

    rw_errors = random_walk_forecast(rw_data=input_set["Random Walk"], sigma=1, df=5,
                                     forecast=forecast_steps, pollution=clean_pollution, func=func)
    ar_errors = ar_forecast(ar_set=input_set["AR(1) Models"], pollution=clean_pollution,
                            forecast=forecast_steps, func=func)
    var_errors = var_forecast(var_mod=input_set["VAR(1) Model"], pollution=clean_pollution,
                              forecast=forecast_steps, func=func)
    svar_errors = svar_forecast(var_mod=input_set["SVAR(1) Model"], pollution=clean_pollution,
                                forecast=forecast_steps, w_matrix=weight_matrix, func=func)
    swvar_errors = swvar_forecast(var_mod=input_set["SWVAR(1) Model"], pollution=clean_pollution,
                                  forecast=forecast_steps, ww_tensor=w_tensor, func=func)

    if restricted:
        res_swvar_errors = swvar_forecast(var_mod=input_set["Restricted SWVAR(1) Model"], pollution=clean_pollution,
                                          forecast=forecast_steps, ww_tensor=w_tensor, func=func)

        return pd.DataFrame({
                "Random Walk": rw_errors,
                "AR(1) Models": ar_errors,
                "VAR(1) Model": var_errors,
                "SVAR(1) Model": svar_errors,
                "SWVAR(1) Model": swvar_errors,
                "Restricted SWVAR(1) Model": res_swvar_errors})
    else:
        return pd.DataFrame({
            "Random Walk": rw_errors,
            "AR(1) Models": ar_errors,
            "VAR(1) Model": var_errors,
            "SVAR(1) Model": svar_errors,
            "SWVAR(1) Model": swvar_errors})


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def AIC(residuals: np.ndarray, endog: np.ndarray) -> float:
    k = endog.shape[1] * (endog.shape[1] + exog.shape[1] + 1)
    n = nobs
    sse = (residuals ** 2).sum()
    aic = n * np.log(sse / n) + 2 * k / n
    return aic


def random_walk_forecast(rw_data: np.ndarray, sigma: float, df: int, forecast: int,
                         pollution: pd.DataFrame, func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    t = forecast
    k = len(pollution.columns)

    output_matrix = np.zeros(t)
    data = np.zeros((t, k))
    eps = np.random.standard_t(df, size=(t, k)) * sigma
    data[0, :] = rw_data[-1, :k] + eps[0, :]

    for i in range(1, t):
        data[i, :] = data[i - 1, :] + eps[i, :]

    for i in range(t):
        output_matrix[i] = func(pollution.iloc[i, :].to_numpy(), data[i, :])

    return output_matrix


def ar_forecast(ar_set: dict, pollution: pd.DataFrame, forecast: int,
                func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    pred = np.zeros((forecast, len(pollution.columns)))
    output_matrix = np.zeros(forecast)

    i = 0
    for column in pollution:
        pred[:, i] = ar_set[column].forecast(steps=forecast)
        i += 1

    for i in range(forecast):
        output_matrix[i] = func(pollution.iloc[i, :].to_numpy(), pred[i, :])

    return output_matrix


def var_forecast(var_mod: VARResults, pollution: pd.DataFrame, forecast: int,
                 func: Callable[[np.ndarray, np.ndarray], float]):
    pred = var_mod.forecast(y=var_mod.endog, steps=forecast)
    output_matrix = np.zeros(forecast)

    for i in range(forecast):
        output_matrix[i] = func(pollution.iloc[i, :].to_numpy(), pred[i, :])
    return output_matrix


def svar_forecast(var_mod: VARResults, pollution: pd.DataFrame, w_matrix: np.ndarray,
                  forecast: int, func: Callable[[np.ndarray, np.ndarray], float]) -> Any:
    pred = np.zeros((forecast, len(pollution.columns)))
    output_matrix = np.zeros(forecast)

    c = np.ones((1, 1))
    argument = np.concatenate([c, (w_matrix @ var_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (var_mod.params.T @ argument).T
    output_matrix[0] = np.mean(np.abs(pollution.iloc[0, :].to_numpy() - pred[0, :]))

    for i in range(1, pred.shape[0]):
        argument = np.concatenate([c, (w_matrix @ pred[i - 1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (var_mod.params.T @ argument).T
        output_matrix[i] = func(pollution.iloc[i, :].to_numpy(), pred[i, :])

    return output_matrix


def swvar_forecast(var_mod: VARResults, pollution: pd.DataFrame, ww_tensor: np.ndarray,
                   forecast: int, func: Callable[[np.ndarray, np.ndarray], float]) -> Any:
    pred = np.zeros((forecast, len(pollution.columns)))
    output_matrix = np.zeros(forecast)

    c = np.ones((1, 1))
    argument = np.concatenate([c, (ww_tensor[0, :, :] @ var_mod.endog[-1, :].T).reshape(len(pollution.columns), 1)])
    pred[0, :] = (var_mod.params.T @ argument).T
    output_matrix[0] = np.mean(np.abs(pollution.iloc[0, :].to_numpy() - pred[0, :]))

    for i in range(1, pred.shape[0]):
        argument = np.concatenate([c, (ww_tensor[i, :, :] @ pred[i - 1, :].T).reshape(len(pollution.columns), 1)])
        pred[i, :] = (var_mod.params.T @ argument).T

    return func(pollution.iloc[:len(pred), :].to_numpy(), pred)


def get_test_data(geo_lev: str, time_lev: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_df = part1(geo_lev=geo_lev, time_lev=time_lev, type_key='test')
    pollution, w_speed, w_angle = part2(geo_lev=geo_lev, time_lev=time_lev)
    wind_spillover, space_spillover, w_array, ww_tensor = part3(clean_df, pollution, w_speed, w_angle,
                                                                geo_lev=geo_lev, time_lev=time_lev)
    return pollution, w_array, ww_tensor
