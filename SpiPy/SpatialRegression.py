from __future__ import annotations
from typing import Any
from pandas import DataFrame
from statsmodels.api import GLS
from statsmodels.tsa.api import VAR
import sklearn.metrics as skm
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.stats.correlation_tools import cov_nearest


def spatial_VAR(spillover_matrix: pd.DataFrame) -> VARResults:
    """
    :param spillover_matrix: dataset with the pollution spillovers from adjacent locations
    :return: VAR model
    """
    spatial_var = VAR(endog=spillover_matrix).fit(maxlags=1, trend='c')
    print(spatial_var.summary())
    # psd_cov = cov_nearest(spatial_var.cov, method='clipped', threshold=1e-15, n_fact=100, return_all=False)
    # spatial_var.irf(periods=10)
    return spatial_var


def restricted_spatial_VAR(spillover_matrix: pd.DataFrame) -> VARResults:
    """
    :param spillover_matrix: dataset with the pollution spillovers from adjacent locations
    :return: VAR model
    """

    n = len(spillover_matrix.columns)
    A = np.zeros((n, n))
    A[np.diag_indices(n)] = 1
    spatial_var = GLS(endog=spillover_matrix.iloc[1:, :], exog=spillover_matrix.shift(1).iloc[1:, :]).fit_constrained(A)
    print(spatial_var.summary())
    return spatial_var


def get_R2(model, location_dict) -> None:
    """
    :param model: input VAR model
    :param location_dict: dictionary with the name of all different locations in the model
    :return: None as it prints the R2 from each equation
    """
    for key in location_dict:
        R2 = skm.r2_score(model.fittedvalues[key] + model.resid[key], model.fittedvalues[key])
        print(f'The R-Squared of {key} is: {R2 * 100:.2f}%')
    return None


def spatial_data(path_data: str, path_spill: str) -> tuple[DataFrame | Any, DataFrame | Any]:
    """
    :param path_data: filepath to the pollution data
    :param path_spill: filepath to the spillover data
    :return: Pandas DataFrames of each dataset
    """
    return pd.read_csv(path_data, index_col=0), pd.read_csv(path_spill, index_col=0)
