from SpiPy import DataPrep, Separator, SpatialTools, SpatialRegression, ModelConfidenceSet
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VAR
import pandas as pd
import numpy as np


def Part1(filepath: str, geo_lev: str, time_lev: str) -> pd.DataFrame:
    """
    :param filepath: file path to the raw training set
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: dataframe with the clean data
    """
    data = DataPrep.group_data(DataPrep.format_data(DataPrep.get_data(filepath)), geo_lev, time_lev)
    data.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_data.csv")
    return data


def Part2(geo_lev: str, time_lev: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: individual dataframes for each variable
    """
    filepath = "/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_data.csv"
    data = Separator.get_clean_data(filepath)

    if geo_lev == "municipality":
        no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]
    else:
        no_sensors = []
    pollution, w_speed, w_angle = Separator.matrix_creator(data, geo_lev, no_sensors)
    pollution.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "pollution.csv")
    w_speed.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "wind_speed.csv")
    w_angle.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "wind_angle.csv")
    return pollution, w_speed, w_angle


def Part3(df_gen: pd.DataFrame, df_pol: pd.DataFrame, df_speed: pd.DataFrame,
          df_angle: pd.DataFrame, geo_lev: str, time_lev: str) -> (pd.DataFrame, np.ndarray):
    """
    :param df_gen: cleaned dataset
    :param df_pol: dataset of pollution levels
    :param df_speed: dataset of wind speeds
    :param df_angle: dataset of wind directions
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :param tensor_typ: conditional to include wind variables in the calculations
    :return: dataframe with spatial spillover effects
    """

    coordinates = SpatialTools.coordinate_dict(df_gen, geo_lev, df_pol)
    weight_matrix, angle_matrix = SpatialTools.weight_angle_matrix(coordinates)

    wind_spillover_matrix, tensor_W = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed,
                                                                  weight_matrix, angle_matrix,
                                                                  tensor_type="wind")
    space_spillover_matrix = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed, weight_matrix,
                                                         angle_matrix, tensor_type="space")

    np.save("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "tensor_W.npy", tensor_W)
    wind_spillover_matrix.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/"
                                 + "spillover_effects_wind.csv")
    space_spillover_matrix.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/"
                            + "spillover_effects_space.csv")
    return wind_spillover_matrix, space_spillover_matrix, weight_matrix, tensor_W


def Part4(spillovers: pd.DataFrame) -> VARResults:
    """
    :param spillovers: dataset with spatial spillover effects
    :return: VAR model
    """

    spatial_model = SpatialRegression.spatial_VAR(spillovers)
    return spatial_model
