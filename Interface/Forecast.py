from statsmodels.tsa.vector_ar.var_model import VARResults
from Interface.Backbone import Part1, Part2, Part3
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np


def get_performance(input_set: list, geo_lev: str, time_lev: str) -> list:
    tensor_w = "wind"
    no_tensor = "space"
    path = "/Users/main/Vault/Thesis/Data/Core/test_data.csv"

    pollution, spillovers_wind = get_test_data(path, geo_lev=geo_lev, time_lev=time_lev, tensor_type=tensor_w)
    pollution, spillovers_space = get_test_data(path, geo_lev=geo_lev, time_lev=time_lev, tensor_type=no_tensor)

    rw_errors = random_walk_forecast(rw_data=input_set["Random Walk"], sigma=1,
                                     geo_lev=geo_lev, time_lev=time_lev)
    ar_errors = AR_forecast(ar_set=input_set["AR(1) Models"], pollution=pollution, time_lev=time_lev)
    VAR_errors = VAR_forecast(VAR_mod=input_set["VAR(1) Model"], pollution=pollution, time_lev=time_lev)
    SVAR_errors = SVAR_forecast(VAR_mod=input_set["SVAR(1) Model"], pollution=pollution,
                                time_lev=time_lev, spillover_matrix=spillovers_space)
    SWVAR_errors = SWVAR_forecast(VAR_mod=input_set["SWVAR(1) Model"], pollution=pollution,
                                  time_lev=time_lev, spillover_matrix=spillovers_wind)

    return {"Random Walk": rw_errors,
            "AR(1) Models": ar_errors,
            "VAR(1) Model": VAR_errors,
            "SVAR(1) Model": SVAR_errors,
            "SWVAR(1) Model": SWVAR_errors}


def random_walk_forecast(rw_data, sigma: float, geo_lev: str, time_lev: str) -> np.ndarray:

    if geo_lev == "street":
        df = 170
    else:
        df = 14

    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    output_matrix = np.zeros((len(pollution), len(pollution.columns)))

    T = len(pollution)
    K = len(pollution.columns)

    eps = np.random.standard_t(df, size=(T, K)) * sigma
    data = np.zeros((T, K))
    data[0, :] = rw_data[-1, :]

    for t in range(1, T):
        data[t, :] = data[t - 1, :] + eps[t, :]

    for i in range(len(pollution.columns)):
        output_matrix[:, i] = pollution.iloc[:, i] - data[:, i]

    return output_matrix


def AR_forecast(ar_set: dict, pollution: pd.DataFrame, time_lev: str) -> np.ndarray:
    output_matrix = np.zeros((len(pollution), len(pollution.columns)))

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365*24

    for i in range(len(pollution)):
        pred = ar_set[pollution.columns[i]].forecast(steps=forecast_steps)
        output_matrix[:, i] = pollution.iloc[:, i] - pred

    return output_matrix


def VAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, time_lev: str):

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    pred = VAR_mod.forecast(y=pollution.iloc[-1:], steps=forecast_steps)
    return pollution - pred


def SVAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, time_lev: str, spillover_matrix: pd.DataFrame):

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    pred = VAR_mod.forecast(y=pollution.iloc[-1:], steps=forecast_steps, exog_future=spillover_matrix.iloc[-1:])
    return pollution - pred


def SWVAR_forecast(VAR_mod: VARResults, pollution: pd.DataFrame, time_lev: str, spillover_matrix: pd.DataFrame):

    if time_lev == "day":
        forecast_steps = 365
    else:
        forecast_steps = 365 * 24

    pred = VAR_mod.forecast(y=pollution.iloc[-1:], steps=forecast_steps, exog_future=spillover_matrix.iloc[-1:])
    return pollution - pred


def get_test_data(filepath: str, geo_lev: str, time_lev: str, tensor_type: str) -> pd.DataFrame:
    clean_df = Part1(filepath, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    spillover_df = Part3(clean_df, pollution, w_speed, w_angle, geo_lev=geo_lev,
                         time_lev=time_lev, tensor_typ=tensor_type)
    return pollution, spillover_df




