from __future__ import annotations

from statsmodels.tsa.ar_model import AutoReg
from numpy import zeros, ndarray
from numpy.linalg import eigvals
from statsmodels.api import OLS
from SpiPy.Models.SAR import *
from pandas import DataFrame


class SpatialVAR:
    def __init__(
            self, lags: int = 1,
            constant: bool = False,
            endog: DataFrame = None,
            exog: DataFrame = None,
            model_type: str = "Unrestricted",
            verbose: bool = True
    ) -> None:
        self.order = lags
        self.const = constant
        self.endog = endog.iloc[1:, :]
        self.exog = exog.shift(1).iloc[1:, :]
        self.verbose = verbose
        self.model = model_type
        self.params = None
        self.phi = None
        self.fitted_model = None

    def fit(self) -> None:
        if self.model == "Diagonal":
            self.fit_diagonal()
        elif self.model == "DSTSAR":
            self.fit_DSTSAR()
        elif self.model == "ARD":
            self.fit_ard()
        else:
            self.fit_unrestricted()
        return None

    def fit_ard(self) -> None:
        n = len(self.endog.columns)
        output_models = {}

        for column in self.endog.columns:
            output_models[column] = AutoReg(endog=self.endog[column], exog=self.exog[column], trend='n', lags=1).fit()
            if self.verbose:
                print(output_models[column].summary())

        self.params = zeros((n, n))
        self.phi = zeros((n, n))
        for i in range(n):
            self.params[i, i] = output_models[self.endog.columns[i]].params[1]
            self.phi[i, i] = output_models[self.endog.columns[i]].params[0]
        self.fitted_model = output_models
        return None

    def fit_unrestricted(self) -> None:
        spatial_var = OLS(endog=self.endog, exog=self.exog).fit()
        if self.verbose:
            print(spatial_var.summary())
        self.params = spatial_var.params
        self.fitted_model = spatial_var
        return None

    def fit_diagonal(self) -> None:
        n = len(self.endog.columns)
        output_models = {}

        for column in self.endog.columns:
            output_models[column] = OLS(endog=self.endog[column], exog=self.exog[column]).fit()
            if self.verbose:
                print(output_models[column].summary())

        self.params = zeros((n, n))

        for i in range(n):
            self.params[i, i] = output_models[self.endog.columns[i]].params[0]
        self.fitted_model = output_models
        return None

    def fit_DSTSAR(self) -> ndarray:
        # Initial guess
        N = len(self.endog.columns)
        initial_params = zeros(3*N+5)
        initial_params[:N] = [0.9] * N                                                          # Phi
        initial_params[N:2*N] = list(self.endog.var().values)           # Sigma
        initial_params[2*N:3*N] = list(self.endog.mean().values)        # Mu
        initial_params[-5] = 7.87                                                               # Alpha
        initial_params[-4] = -16.5                                                              # Rho
        initial_params[-3] = 0.05                                                               # Zeta
        initial_params[-2] = 5.09                                                               # Beta
        initial_params[-1] = 0.15                                                               # Gamma

        bounds = [(-20, 20)] * N             # Phi
        bounds += [(1, 1000)] * N            # Sigma
        bounds += [(0, 1000)] * N            # Mu
        bounds += [(-100, 100)]              # Alpha
        bounds += [(-100, 100)]              # Rho
        bounds += [(0, 1)]                   # Zeta
        bounds += [(-100, 100)] * 2          # Beta and Gamma

        reg_matrix = self.database.train_data.weight_matrix/max(eigvals(self.database.train_data.weight_matrix.values))
        optimizer = QMLEOptimizer(
            initial_params=initial_params,
            weight_matrix=reg_matrix,
            wind_tensor=self.database.train_data.wSpillovers,
            exog=self.database.train_data.pollution.values,
            bounds=bounds,
        )

        optimizer.fit()
        model = optimizer.get_best_params()
        return model

# def fit_smooth_transition() -> None:
#     return None
