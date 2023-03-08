from statsmodels.tsa.vector_ar.var_model import VARResults

from Interface.Backbone import Part1, Part2, Part3, Part4
from statsmodels.tsa.api import VAR, AutoReg
import numpy as np


np.random.seed(123)


def random_walk(df: int, sigma: float, geo_lev: str, time_lev: str) -> np.array:
    """
    :param df: degrees of freedom
    :param sigma: variance of the error term
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: matrix with simulated random walk
    """
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)

    T = len(pollution)
    K = len(pollution.columns)

    eps = np.random.standard_t(df, size=(T, K)) * sigma
    data = np.zeros((T, K))
    data[0, :] = np.random.normal(0, sigma, size=K)

    for t in range(1, T):
        data[t, :] = data[t - 1, :] + eps[t, :]

    return data


def AR_model(lags: int, geo_lev: str, time_lev: str) -> list:
    """
    :param lags: number of lags to include in the model
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: list of AR models for each geographical location
    """
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)

    output_models = []
    for column in pollution:
        mod = AutoReg(pollution[column], lags=lags).fit()
        output_models.append(mod)
    return output_models


def SWVAR(filepath: str, geo_lev: str, time_lev: str) -> VARResults:
    """
    :param filepath: filepath to the raw data
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: VAR model with wind
    """
    tensor = "wind"

    clean_df = Part1(filepath, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    spillover_df = Part3(clean_df, pollution, w_speed, w_angle, geo_lev=geo_lev, time_lev=time_lev, tensor_typ=tensor)
    SWVAR_Model: VARResults = Part4(pollution, spillover_df, geo_lev=geo_lev, time_lev=time_lev, tensor_typ=tensor)
    return SWVAR_Model


def SVAR(filepath: str, geo_lev: str, time_lev: str) -> VARResults:
    """
    :param filepath: filepath to the raw data
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: VAR model without wind
    """
    tensor = "space"

    clean_df = Part1(filepath, geo_lev=geo_lev, time_lev=time_lev)
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)
    spillover_df = Part3(clean_df, pollution, w_speed, w_angle, geo_lev=geo_lev, time_lev=time_lev, tensor_typ=tensor)
    SVAR_Model: VARResults = Part4(pollution, spillover_df, geo_lev=geo_lev, time_lev=time_lev, tensor_typ=tensor)
    return SVAR_Model


def standard_VAR(geo_lev: str, time_lev: str) -> VARResults:
    """
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: VAR model without spatial variables
    """
    pollution, w_speed, w_angle = Part2(geo_lev=geo_lev, time_lev=time_lev)

    model = VAR(pollution).fit(maxlags=1)
    return model


def create_set(geo_lev: str, time_lev: str) -> list:
    """
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: set of models
    """
    path = "/Users/main/Vault/Thesis/Data/Core/train_data.csv"
    return [SWVAR(path, geo_lev=geo_lev, time_lev=time_lev),
            SVAR(path, geo_lev=geo_lev, time_lev=time_lev),
            standard_VAR(geo_lev=geo_lev, time_lev=time_lev),
            AR_model(lags=1, geo_lev=geo_lev, time_lev=time_lev),
            random_walk(df=14, sigma=1, geo_lev=geo_lev, time_lev=time_lev)]
