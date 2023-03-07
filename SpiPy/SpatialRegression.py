from statsmodels.tsa.api import AutoReg, VAR
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd
import numpy as np


def spatial_VAR(pollution, spillover_matrix) -> VAR:
    lagged_spillover = spillover_matrix.shift(1)
    lagged_spillover.at[spillover_matrix.index[0], :] = 0
    spatialVAR = VAR(pollution, exog=lagged_spillover).fit(maxlags=1, trend='c')
    return spatialVAR


def get_R2(model, location_dict) -> None:
    for key in location_dict:
        R2 = skm.r2_score(model.fittedvalues[key] + model.resid[key], model.fittedvalues[key])
        print(f'The R-Squared of {key} is: {R2 * 100:.2f}%')
    return None


def spatial_data(path_data: str, path_spill: str) -> pd.DataFrame:
    return pd.read_csv(path_data, index_col=0), pd.read_csv(path_spill, index_col=0)
