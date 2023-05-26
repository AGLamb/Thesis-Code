from __future__ import annotations
from typing import Any

from statsmodels.tsa.api import VAR, AutoReg
from SpiPy.Models import SpatialRegression
from SpiPy.RunFlow.Backbone import *
from SpiPy.Utils.Forecast import *
from numpy import random, zeros


random.seed(123)


class ModelSet:
    def __init__(self,
                 database: RunFlow,
                 verbose: bool = False) -> None:
        self.random_walk = None
        self.ar_models = None
        self.var_model = None
        self.svar_model = None
        self.swvar_model = None
        self.constant_model = None
        self.diagonal_model = None
        self.ard_model = None
        self.database = database
        self.verbose = verbose

    def run(self) -> None:
        self.ard_model = self.regress(endog=self.database.train_data.pollution,
                                      exog=self.database.train_data.wSpillovers,
                                      model_type="ARD")
        self.diagonal_model = self.regress(endog=self.database.train_data.pollution,
                                           exog=self.database.train_data.wSpillovers,
                                           model_type="Diagonal")
        # self.constant_model = self.regress(endog=self.database.train_data.pollution,
        #                                    exog=self.database.train_data.wSpillovers,
        #                                    model_type="Constant")
        self.swvar_model = self.regress(endog=self.database.train_data.pollution,
                                        exog=self.database.train_data.wSpillovers,
                                        model_type="Unrestricted")
        self.svar_model = self.regress(endog=self.database.train_data.pollution,
                                       exog=self.database.train_data.sSpillovers,
                                       model_type="Unrestricted")
        self.var_model = VAR(endog=self.database.train_data.pollution).fit(maxlags=1, trend='n')
        self.ar_models = ar_model(pollution_data=self.database.train_data.pollution)
        self.random_walk = random_walk(sigma=10, df=5, pollution_data=self.database.train_data.pollution)
        return None

    def get_performance(self, time_lev: str) -> ForecastSet:
        performance = ForecastSet(time_lev=time_lev)
        performance.run(trained_set=self)
        return performance

    def regress(self, endog: DataFrame = None, exog: DataFrame = None, model_type: str = "Unrestricted") -> Any:
        model = SpatialRegression.SpatialVAR(endog=endog,
                                             exog=exog,
                                             model_type=model_type,
                                             verbose=self.verbose)
        model.fit()
        return model


class ForecastSet:
    def __init__(self, time_lev) -> None:
        self.forecast_steps = 100 if time_lev == "day" else (100 * 24)
        self.performance = {"MAPE": {}, "MSE": {}, "RMSE": {}, "MAE": {}, "SMAPE": {}}
        self.metric_func = [MAPE, MSE, RMSE, MAE, SMAPE]

    def run(self, trained_set: ModelSet) -> None:
        self.random_walk_forecast(sigma=1,
                                  df=5,
                                  trained_set=trained_set)
        self.ar_forecast(trained_set=trained_set)
        self.var_forecast(trained_set=trained_set)
        self.svar_forecast(trained_set=trained_set)
        self.swvar_forecast(trained_set=trained_set)
        # self.const_forecast(trained_set=trained_set)
        self.diag_forecast(trained_set=trained_set)
        self.ard_forecast(trained_set=trained_set)

        self.output_maker()
        return None

    def const_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        ww_tensor = trained_set.database.test_data.weight_tensor
        Beta = trained_set.constant_model.params

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))

        for func in self.metric_func:
            self.performance[func.__name__]["Constant"] = []

        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for func in self.metric_func:
            self.performance[func.__name__]["Constant"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

            for func in self.metric_func:
                self.performance[func.__name__]["Constant"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def diag_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        ww_tensor = trained_set.database.test_data.weight_tensor
        Beta = trained_set.diagonal_model.params

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["Diagonal"] = []

        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for func in self.metric_func:
            self.performance[func.__name__]["Diagonal"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

            for func in self.metric_func:
                self.performance[func.__name__]["Diagonal"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def ard_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        ww_tensor = trained_set.database.test_data.weight_tensor
        Beta = trained_set.ard_model.params
        Phi = trained_set.ard_model.phi

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["ARD"] = []

        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        pred[0, :] += pollution.iloc[-1, :].to_numpy() @ Phi
        for func in self.metric_func:
            self.performance[func.__name__]["ARD"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument + pollution.iloc[i - 1, :].to_numpy() @ Phi).T

            for func in self.metric_func:
                self.performance[func.__name__]["ARD"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def output_maker(self) -> None:
        for func in self.metric_func:
            self.performance[func.__name__] = DataFrame(self.performance[func.__name__])
        return None

    def get_performance(self, y_true: ndarray, y_pred: ndarray) -> None:
        output = {}
        for func in self.metric_func:
            output[func.__name__] = func(y_true, y_pred)

    def random_walk_forecast(self,
                             sigma: float,
                             df: int,
                             trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution

        t = self.forecast_steps
        k = len(pollution.columns)

        pred = zeros((t, k))
        eps = random.standard_t(df, size=(t, k)) * sigma
        pred[0, :] = trained_set.random_walk[-1, :] + eps[0, :]

        for func in self.metric_func:
            self.performance[func.__name__]["RW"] = []

        for i in range(1, t):
            pred[i, :] = pred[i - 1, :] + eps[i, :]

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["RW"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def ar_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["AR"] = []

        for i, col in enumerate(pollution):
            pred[:, i] = trained_set.ar_models[col].forecast(steps=t)

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["AR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def var_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        t = self.forecast_steps
        pred = trained_set.var_model.forecast(y=trained_set.var_model.endog, steps=t)
        for func in self.metric_func:
            self.performance[func.__name__]["VAR"] = []

        for i in range(t):
            for func in self.metric_func:
                self.performance[func.__name__]["VAR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def svar_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        w_matrix = trained_set.database.test_data.weight_matrix
        Beta = trained_set.svar_model.params

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["SVAR"] = []

        pred[0, :] = (Beta.T @ trained_set.svar_model.endog.iloc[-1, :].to_numpy().T).T
        for func in self.metric_func:
            self.performance[func.__name__]["SVAR"].append(func(pollution.iloc[0, :].to_numpy(), pred[0, :]))

        for i in range(1, t):
            argument = w_matrix @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

            for func in self.metric_func:
                self.performance[func.__name__]["SVAR"].append(func(pollution.iloc[i, :].to_numpy(), pred[i, :]))
        return None

    def swvar_forecast(self, trained_set: ModelSet) -> None:
        pollution = trained_set.database.test_data.pollution
        ww_tensor = trained_set.database.test_data.weight_tensor
        Beta = trained_set.swvar_model.params

        t = self.forecast_steps
        k = len(pollution.columns)
        pred = zeros((t, k))
        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"] = []

        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"].append(func(pollution.iloc[0, :], pred[0, :]))

        for i in range(1, t):
            argument = ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

            for func in self.metric_func:
                self.performance[func.__name__]["SWVAR"].append(func(pollution.iloc[i, :], pred[i, :]))
        return None


def ar_model(pollution_data: DataFrame) -> dict[Any, Any]:
    output_models = {}
    for column in pollution_data:
        output_models[column] = AutoReg(pollution_data[column], lags=1, trend='n').fit()
    return output_models


def random_walk(sigma: float, df: int, pollution_data: DataFrame) -> ndarray:
    t = len(pollution_data)
    k = len(pollution_data.columns)
    eps = random.standard_t(df, size=(t, k)) * sigma
    data = zeros((t, k))
    data[0, :] = eps[0, :]

    for i in range(1, t):
        data[i, :] = data[i - 1, :] + eps[i, :]
    return data
