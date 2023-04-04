from __future__ import annotations

from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tsa.vector_ar.var_model import VARResults
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
from statsmodels.api import GLS, add_constant
from rpy2.robjects.packages import importr
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from pandas import DataFrame
from rpy2.robjects import r
import rpy2.robjects as ro
from typing import Any
import pandas as pd
import numpy as np


class SpatialVAR:
    def __init__(self, lags: str):
        self.order = lags


def spatial_VAR(spillover_matrix: pd.DataFrame) -> VARResults:
    """
    :param spillover_matrix: dataset with the pollution spillovers from adjacent locations
    :return: VAR model
    """
    spatial_var = VAR(endog=spillover_matrix).fit(maxlags=1, trend='c')
    # print(spatial_var.summary())
    irf = spatial_var.irf(periods=5)
    irf.plot(orth=False)
    plt.show()
    return spatial_var


def restricted_spatial_VAR(spillover_matrix: pd.DataFrame) -> VARResults:
    """
    :param spillover_matrix: dataset with the pollution spillovers from adjacent locations
    :return: VAR model
    """
    print(spillover_matrix)
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(spillover_matrix)
    n = len(spillover_matrix.columns)
    # Define R script
    r_script = """
    # Load required library
    library("vars")
    # Run a normal VAR model
    t = VAR(data, p = 1, type = "const")
    restrict_m <- matrix(1, nrow = n, ncol = n + 1)
    model <- restrict(t, method = "man", resmat = restrict_m)
    estimates = as.data.frame(model$varresult$Phi)
    """
    r_env = ro.globalenv
    r_env['data'] = r_df
    r_env['n'] = n
    ro.r(r_script)
    model = r_env['estimates']
    model_df = pandas2ri.DataFrame(model)
    # with localconverter(ro.default_converter + pandas2ri.converter):
    #     model_dict = ro.conversion.rpy2py(model)
    # model_df = pd.DataFrame(model_dict)
    print(model_df)
    return model


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
