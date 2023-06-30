from __future__ import annotations

from statsmodels.tsa.ar_model import AutoReg
from numpy import zeros, ndarray
from statsmodels.api import OLS
from SpiPy.Models.SAR import *
from pandas import DataFrame


class SpatialVAR:
    def __init__(
            self, lags: int = 1,
            constant: bool = False,
            endog: DataFrame = None,
            exog: DataFrame = None,
            mWeight: ndarray = None,
            tWind: ndarray = None,
            model_type: str = "Unrestricted",
            verbose: bool = True
    ) -> None:

        self.wSpillovers = tWind
        self.model = model_type
        self.mWeight = mWeight
        self.verbose = verbose
        self.const = constant
        self.order = lags
        self.endog = endog
        self.exog = exog

        self.fitted_model = None
        self.params = None
        self.phi = None

    def fit(self) -> None:
        if self.model == "Diagonal":
            self.fit_diagonal()
        elif self.model == "Constant":
            self.fit_DSTSAR()
        elif self.model == "ARD":
            self.fit_ard()
        elif self.model == "T":
            self.fit_T()
        else:
            self.fit_unrestricted()
        return None

    def fit_T(self) -> None:
        # Initial guess
        N = self.endog.shape[1]
        initial_params = zeros(3*N+6)
        initial_params[:N] = [1.5] * N                                           # Phi
        initial_params[N:2*N] = list(self.endog.var().values)                     # Sigma
        initial_params[2*N:3*N] = list(self.endog.mean().values)                  # Mu
        initial_params[-6] = 0.3                                                 # Alpha
        initial_params[-5] = 0.5                                                 # Rho
        initial_params[-4] = 0.7                                                 # Zeta
        initial_params[-3] = 0.5                                                 # Beta
        initial_params[-2] = 1.5                                                 # Gamma
        initial_params[-1] = 25                                                  # DF

        bounds = [(None, None)] * N                                              # Phi
        bounds += [(1, None)] * N                                                # Sigma
        bounds += [(0, None)] * N                                                # Mu
        bounds += [(None, None)]                                                 # Alpha
        bounds += [(None, None)]                                                 # Rho
        bounds += [(0, 1)]                                                       # Zeta
        bounds += [(None, None)]                                                 # Beta
        bounds += [(None, None)]                                                 # Gamma
        bounds += [(2.01, None)]                                                    # DF

        optimizer = TQMLEOptimizer(
            initial_params=initial_params,
            weight_matrix=self.mWeight,
            wind_tensor=self.wSpillovers,
            exog=self.endog.values,
            bounds=bounds
        )

        optimizer.fit()
        self.params = optimizer.get_best_params()
        self.fitted_model = optimizer
        return

    def fit_ard(self) -> None:
        n = len(self.endog.columns)
        output_models = {}

        for column in self.endog.columns:
            output_models[column] = AutoReg(
                endog=self.endog[column].iloc[1:], exog=self.exog[column].shift(1).iloc[1:],
                trend='n',
                lags=1
            ).fit()
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
        spatial_var = OLS(endog=self.endog.iloc[1:], exog=self.exog.shift(1).iloc[1:, :]).fit()
        if self.verbose:
            print(spatial_var.summary())
        self.params = spatial_var.params
        self.fitted_model = spatial_var
        return None

    def fit_diagonal(self) -> None:
        n = len(self.endog.columns)
        output_models = {}

        for column in self.endog.columns:
            output_models[column] = OLS(
                endog=self.endog[column].iloc[1:],
                exog=self.exog[column].shift(1).iloc[1:]
            ).fit()
            if self.verbose:
                print(output_models[column].summary())

        self.params = zeros((n, n))

        for i in range(n):
            self.params[i, i] = output_models[self.endog.columns[i]].params[0]
        self.fitted_model = output_models
        return None

    def fit_DSTSAR(self) -> None:
        N = self.endog.shape[1]
        initial_params = zeros(3*N+5)
        initial_params[:N] = [0.9] * N                                                          # Phi
        initial_params[N:2*N] = list(self.endog.var().values)                                   # Sigma
        initial_params[2*N:3*N] = list(self.endog.mean().values)                                # Mu
        initial_params[-5] = 7.87                                                               # Alpha
        initial_params[-4] = -16.5                                                              # Rho
        initial_params[-3] = 0.05                                                               # Zeta
        initial_params[-2] = 5.09                                                               # Beta
        initial_params[-1] = 0.15                                                               # Gamma

        bounds = [(-1, 1)] * N               # Phi
        bounds += [(1, 1000)] * N            # Sigma
        bounds += [(0, None)] * N            # Mu
        bounds += [(-1, 1)]                  # Alpha
        bounds += [(-1, 1)]                  # Rho
        bounds += [(0, 1)]                   # Zeta
        bounds += [(None, None)] * 2         # Beta and Gamma

        optimizer = QMLEOptimizer(
            initial_params=initial_params,
            weight_matrix=self.mWeight,
            wind_tensor=self.wSpillovers,
            exog=self.endog.values,
            bounds=bounds
        )

        optimizer.fit()
        self.params = optimizer.get_best_params()
        self.fitted_model = optimizer
        return None
