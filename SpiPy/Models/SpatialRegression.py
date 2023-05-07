from __future__ import annotations

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.api import OLS
from pandas import DataFrame
from numpy import zeros


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
        elif self.model == "Constant":
            self.fit_constant()
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

# def fit_constant(self) -> None:
#      variables = self.endog.columns
#      return None

# def fit_smooth_transition() -> None:
#     return None
