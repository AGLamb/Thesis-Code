from statsmodels.tsa.api import AutoReg, VAR
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd
import numpy as np


def spatial_VAR(pollution, spillover_matrix):
    lagged_spillover = spillover_matrix.shift(1)
    lagged_spillover.at[spillover_matrix.index[0], :] = 0
    spatialVAR = VAR(pollution, exog=lagged_spillover).fit(maxlags=1, trend='c')
    print(spatialVAR.summary())
    return spatialVAR


def get_R2(model, location_dict):
    for key in location_dict:
        R2 = skm.r2_score(model.fittedvalues[key] + model.resid[key], model.fittedvalues[key])
        print(f'The R-Squared of {key} is: {R2 * 100:.2f}%')
    return


def spatial_data(path_data: str, path_spill: str):
    return pd.read_csv(path_data, index_col=0), pd.read_csv(path_spill, index_col=0)


def main() -> None:
    filepath_pol = "/Users/main/Vault/Thesis/Code/Data/pollution.csv"
    filepath_spill = "/Users/main/Vault/Thesis/Code/Data/spillover_effects.csv"
    df_pol, WWY = spatial_data(filepath_pol, filepath_spill)
    spatial_model = spatial_VAR(df_pol, WWY)

    print(spatial_model.test_normality(signif=0.05).summary())
    return None


if __name__ == "__main__":
    main()
