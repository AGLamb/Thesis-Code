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

        self.t = 100 if time_lev == "day" else (100 * 24)
        self.metric_func = [MAPE, MSE, RMSE, MAE, SMAPE]
        self.ww_tensor = None
        self.pollution = None
        self.w_matrix = None
        self.k = None

        self.performance = {
            "MAPE": {
                'AR': zeros(self.t),
                'VAR': zeros(self.t),
                'SVAR': zeros(self.t),
                'SWVAR': zeros(self.t),
                # 'Constant': zeros(self.t),
                'ARD': zeros(self.t),
                'Diagonal': zeros(self.t)
            },
            "MSE": {
                'AR': zeros(self.t),
                'VAR': zeros(self.t),
                'SVAR': zeros(self.t),
                'SWVAR': zeros(self.t),
                # 'Constant': zeros(self.t),
                'ARD': zeros(self.t),
                'Diagonal': zeros(self.t)
            },
            "RMSE": {
                'AR': zeros(self.t),
                'VAR': zeros(self.t),
                'SVAR': zeros(self.t),
                'SWVAR': zeros(self.t),
                # 'Constant': zeros(self.t),
                'ARD': zeros(self.t),
                'Diagonal': zeros(self.t)
            },
            "MAE": {
                'AR': zeros(self.t),
                'VAR': zeros(self.t),
                'SVAR': zeros(self.t),
                'SWVAR': zeros(self.t),
                # 'Constant': zeros(self.t),
                'ARD': zeros(self.t),
                'Diagonal': zeros(self.t)
            },
            "SMAPE": {
                'AR': zeros(self.t),
                'VAR': zeros(self.t),
                'SVAR': zeros(self.t),
                'SWVAR': zeros(self.t),
                # 'Constant': zeros(self.t),
                'ARD': zeros(self.t),
                'Diagonal': zeros(self.t)
            },
        }

    def run(self, trained_set: ModelSet) -> None:
        self.pollution = trained_set.database.test_data.pollution
        self.k = len(self.pollution.columns)
        self.ww_tensor = trained_set.database.test_data.weight_tensor

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
        Beta = trained_set.constant_model.params

        pred = zeros((self.t, self.k))
        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for i in range(1, self.t):
            argument = self.ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

        for func in self.metric_func:
            self.performance[func.__name__]["Constant"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def diag_forecast(self, trained_set: ModelSet) -> None:
        Beta = trained_set.diagonal_model.params

        pred = zeros((self.t, self.k))
        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for i in range(1, self.t):
            argument = self.ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

        for func in self.metric_func:
            self.performance[func.__name__]["Diagonal"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def ard_forecast(self, trained_set: ModelSet) -> None:
        Beta = trained_set.ard_model.params
        Phi = trained_set.ard_model.phi

        pred = zeros((self.t, self.k))
        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        pred[0, :] += self.pollution.iloc[-1, :].to_numpy() @ Phi
        for i in range(1, self.t):
            argument = self.ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument + self.pollution.iloc[i - 1, :].to_numpy() @ Phi).T

        for func in self.metric_func:
            self.performance[func.__name__]["ARD"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def output_maker(self) -> None:
        for func in self.metric_func:
            self.performance[func.__name__] = DataFrame(self.performance[func.__name__])
        return None

    def get_performance(self, y_true: ndarray, y_pred: ndarray) -> None:
        output = {}
        for func in self.metric_func:
            output[func.__name__] = func(y_true, y_pred)

    def ar_forecast(self, trained_set: ModelSet) -> None:
        pred = zeros((self.t, self.k))

        for i, col in enumerate(self.pollution):
            pred[:, i] = trained_set.ar_models[col].forecast(steps=self.t)

        for func in self.metric_func:
            self.performance[func.__name__]["AR"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def var_forecast(self, trained_set: ModelSet) -> None:
        pred = trained_set.var_model.forecast(y=trained_set.var_model.endog, steps=self.t)
        for func in self.metric_func:
            self.performance[func.__name__]["VAR"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def svar_forecast(self, trained_set: ModelSet) -> None:
        w_matrix = trained_set.database.test_data.weight_matrix
        Beta = trained_set.svar_model.params

        pred = zeros((self.t, self.k))
        pred[0, :] = (Beta.T @ trained_set.svar_model.endog.iloc[-1, :].to_numpy().T).T
        for i in range(1, self.t):
            argument = w_matrix @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

        for func in self.metric_func:
            self.performance[func.__name__]["SVAR"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None

    def swvar_forecast(self, trained_set: ModelSet) -> None:
        Beta = trained_set.swvar_model.params

        pred = zeros((self.t, self.k))
        pred[0, :] = trained_set.swvar_model.endog.iloc[-1, :].to_numpy() @ Beta
        for i in range(1, self.t):
            argument = self.ww_tensor[i - 1, :, :] @ pred[i - 1, :].T
            pred[i, :] = (Beta.T @ argument).T

        for func in self.metric_func:
            self.performance[func.__name__]["SWVAR"][:] = func(self.pollution.iloc[:self.t].to_numpy(), pred[:, :])
        return None


def ar_model(pollution_data: DataFrame) -> dict[Any, Any]:
    output_models = {}
    for column in pollution_data:
        output_models[column] = AutoReg(pollution_data[column], lags=1, trend='n').fit()
    return output_models
