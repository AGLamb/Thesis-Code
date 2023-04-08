from __future__ import annotations
from typing import Any
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResults
from SpiPy.Backbone import part1, part2, part3, part4
from SpiPy.Forecast import *
from statsmodels.tsa.api import VAR, AutoReg
import numpy as np

np.random.seed(123)


class ModelSet:
    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 geo_lev: str,
                 time_lev: str,
                 restricted: bool = False) -> None:
        self.random_walk = None
        self.ar_models = None
        self.var_model = None
        self.svar_model = None
        self.swvar_model = None
        self.res_swvar_model = None
        self.geo_lev = geo_lev
        self.time_lev = time_lev
        self.restricted = restricted
        self.train_data = train_data
        self.test_data = test_data

    def run(self) -> None:
        pollution, w_speed, w_angle = part2(geo_lev=self.geo_lev, time_lev=self.time_lev, type_key='train')
        wind_spillover, space_spillover, w_matrix, ww_tensor = part3(df_gen=self.train_data,
                                                                     df_pol=pollution,
                                                                     df_speed=w_speed,
                                                                     df_angle=w_angle,
                                                                     geo_lev=self.geo_lev,
                                                                     time_lev=self.time_lev)
        self.swvar_model = part4(wind_spillover, restricted=False)
        self.svar_model = part4(space_spillover, restricted=False)
        self.var_model = VAR(pollution).fit(maxlags=1, trend='n')
        self.ar_models = ar_model(pollution_data=pollution)
        self.random_walk = random_walk(sigma=4, df=5, pollution_data=pollution)
        if self.restricted:
            self.res_swvar_model = part4(wind_spillover, restricted=True)
        return None

    def get_performance(self) -> ForecastSet:
        performance = ForecastSet(trained_set=self)
        performance.run(test_data=self.test_data)
        return performance


class ForecastSet:
    def __init__(self,
                 trained_set: ModelSet,
                 data: pd.DataFrame = None) -> None:
        self.train_set = trained_set
        self.data = data
        self.forecast_steps = 365 if self.train_set.time_lev == "day" else (365 * 24)
        self.performance = {"MAPE": {}, "MSE": {}, "RMSE": {}, "MAE": {}, "SMAPE": {}}
        self.metric_func = [MAPE, MSE, RMSE, MAE, SMAPE]

    def run(self, test_data: pd.DataFrame) -> None:
        clean_pollution, weight_matrix, w_tensor = self.get_test_data(test_data=test_data)
        self.random_walk_forecast(sigma=1,
                                  df=5,
                                  pollution=clean_pollution)
        self.ar_forecast(pollution=clean_pollution)
        self.var_forecast(pollution=clean_pollution)
        self.svar_forecast(pollution=clean_pollution,
                           w_matrix=weight_matrix)
        self.swvar_forecast(pollution=clean_pollution,
                            ww_tensor=w_tensor)

        if self.train_set.restricted:
            self.restricted_forecast(pollution=clean_pollution)

        self.output_maker()
        return None

    def output_maker(self) -> None:
        for func in self.metric_func:
            self.performance[func.__name__] = pd.DataFrame(self.performance[func.__name__])
        return None

    def get_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        output = {}
        for func in self.metric_func:
            output[func.__name__] = func(y_true, y_pred)

    def get_test_data(self, test_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        pollution, w_speed, w_angle = part2(geo_lev=self.train_set.geo_lev,
                                            time_lev=self.train_set.time_lev,
                                            type_key='test')
        wind_spillover, space_spillover, w_array, ww_tensor = part3(df_gen=test_data,
                                                                    df_pol=pollution,
                                                                    df_speed=w_speed,
                                                                    df_angle=w_angle,
                                                                    geo_lev=self.train_set.geo_lev,
                                                                    time_lev=self.train_set.time_lev)
        return pollution, w_array, ww_tensor

    def random_walk_forecast(self, sigma: float, df: int,
                             pollution: pd.DataFrame) -> None:
        t = self.forecast_steps
        k = len(pollution.columns)

        pred = np.zeros((t, k))
        eps = np.random.standard_t(df, size=(t, k)) * sigma
        pred[0, :] = self.train_set.random_walk[-1, :k] + eps[0, :]
        for func in self.metric_func:
            self.performance[func.__name__]["RW"] = []

        for i in range(1, t):
            pred[i, :] = pred[i - 1, :] + eps[i, :]

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["RW"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def ar_forecast(self, pollution: pd.DataFrame) -> None:

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = np.zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["AR"] = []

        for i, col in enumerate(pollution):
            pred[:, i] = self.train_set.ar_models[col].forecast(steps=t)

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["AR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def var_forecast(self, pollution: pd.DataFrame) -> None:
        t = self.forecast_steps
        pred = self.train_set.var_model.forecast(y=self.train_set.var_model.endog, steps=t)
        for func in self.metric_func:
            self.performance[func.__name__]["VAR"] = []

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["VAR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def svar_forecast(self, pollution: pd.DataFrame, w_matrix: np.ndarray) -> None:
        t = self.forecast_steps
        k = len(pollution.columns)
        pred = np.zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["SVAR"] = []

        argument = w_matrix @ self.train_set.svar_model.endog[-1, :].T
        print(argument)
        pred[0, :] = (self.train_set.svar_model.params.T @ argument).T
        print(pred[0, :])
        for func in self.metric_func:
            self.performance[func.__name__]["SVAR"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = w_matrix @ pred[i - 1, :].T
            print(argument)
            pred[i, :] = (self.train_set.svar_model.params.T @ argument).T
            print(pred[i, :])
            for func in self.metric_func:
                self.performance[func.__name__]["SVAR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def swvar_forecast(self, pollution: pd.DataFrame, ww_tensor: np.ndarray) -> None:
        t = self.forecast_steps
        k = len(pollution.columns)
        pred = np.zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"] = []

        argument = ww_tensor[0, :, :] @ self.train_set.swvar_model.endog[-1, :].T
        print(argument)
        pred[0, :] = (self.train_set.swvar_model.params.T @ argument).T
        print(pred[0, :])
        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = ww_tensor[i, :, :] @ pred[i - 1, :].T
            print(argument)
            pred[i, :] = (self.train_set.swvar_model.params.T @ argument).T
            print(pred[i, :])
            for func in self.metric_func:
                self.performance[func.__name__]["SWVAR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def restricted_forecast(self, pollution: pd.DataFrame) -> None:
        t = self.forecast_steps
        k = len(pollution.columns)
        pred = np.zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"] = []
        return None


def ar_model(pollution_data: pd.DataFrame) -> dict[Any, AutoRegResults]:
    output_models = {}
    for column in pollution_data:
        output_models[column] = AutoReg(pollution_data[column], lags=1, trend='n').fit()
    return output_models


def random_walk(sigma: float, df: int, pollution_data: pd.DataFrame) -> np.array:
    t = len(pollution_data)
    k = len(pollution_data.columns)
    eps = np.random.standard_t(df, size=(t, k)) * sigma
    data = np.zeros((t, k))
    data[0, :] = eps[0, :]

    for i in range(1, t):
        data[i, :] = data[i - 1, :] + eps[i, :]
    return data
