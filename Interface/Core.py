from SpiPy import DataPrep, Separator, SpatialTools, SpatialRegression, ModelConfidenceSet
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np
import Template


def SWVAR(filepath: str, geo_lev: str, time_lev: str) -> VAR:
    tensor = "wind"

    clean_df = Part1(filepath, geo_lev, time_lev)
    pollution, w_speed, w_angle = Part2(geographical_level, time_level)
    spillover_df = Part3(clean_df, pollution, w_speed, w_angle, geographical_level, time_level, tensor)
    SVAR_Model = Part4(pollution, spillover_df, geographical_level, time_level)
    return SVAR_Model


def SVAR(filepath: str, geo_lev: str, time_lev: str) -> VAR:
    tensor = "space"

    clean_df = Part1(filepath, geo_lev, time_lev)
    pollution, w_speed, w_angle = Part2(geographical_level, time_level)
    spillover_df = Part3(clean_df, pollution, w_speed, w_angle, geographical_level, time_level, tensor)
    SVAR_Model = Part4(pollution, spillover_df, geographical_level, time_level)
    return SVAR_Model


def main():
    path = '/Users/main/Vault/Thesis/Data/pm25_weer.csv'
    geographical_level = "municipality"
    time_level = "hours"

    model_SWVAR = SWVAR(path, geographical_level, time_level)
    model_SVAR = SVAR(path, geographical_level, time_level)
    print(model_SWVAR.summary(), "\n", model_SVAR.summary())
    return


if __name__ == "__main__":
    main()
