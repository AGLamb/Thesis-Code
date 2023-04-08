from SpiPy import DataPrep, SpatialTools, SpatialRegression
from statsmodels.tsa.vector_ar.var_model import VARResults
import pandas as pd
import numpy as np


def part1(geo_lev: str, time_lev: str) -> tuple:
    """
    :param type_key:
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: dataframe with the clean data
    """
    path_train = r"/Users/main/Vault/Thesis/Data/Core/train_data.csv"
    path_test = "/Users/main/Vault/Thesis/Data/Core/test_data.csv"

    if geo_lev == "street":
        geo_group = "name"
    else:
        geo_group = "tag"

    no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]

    train_df = DataPrep.group_data(DataPrep.format_data(DataPrep.get_data(path_train),
                                                        faulty=no_sensors), geo_lev, time_lev)

    test_df = DataPrep.group_data(DataPrep.format_data(DataPrep.get_data(path_train),
                                                       faulty=no_sensors), geo_lev, time_lev)
    misplaced = set(train_df[geo_group].unique()) - set(test_df[geo_group].unique())
    train_data = delete_places(df_input=train_df, pop_places=misplaced, key=geo_group)
    test_data = delete_places(df_input=test_df, pop_places=misplaced, key=geo_group)
    train_data.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_train_data.csv")
    test_data.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_test_data.csv")
    return train_data, test_data


def delete_places(df_input: pd.DataFrame, pop_places: set, key: str) -> pd.DataFrame:
    """
    :param key:
    :param pop_places:
    :param df_input: dataset to analyse
    :return: dataset without the removed sensors
    """
    if len(pop_places) > 0:
        df = df_input.copy()
        df = df[~df[key].isin(pop_places)]
        return df
    else:
        return df_input


def part2(geo_lev: str, time_lev: str, type_key: str, save_data: bool = False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param save_data:
    :param type_key:
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: individual dataframes for each variable
    """
    filepath = "/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_" + type_key + "_data.csv"
    data = DataPrep.get_clean_data(filepath)

    pollution, w_speed, w_angle = DataPrep.matrix_creator(data, geo_lev)

    if save_data:
        pollution.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "pollution.csv")
        w_speed.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "wind_speed.csv")
        w_angle.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "wind_angle.csv")
    return pollution, w_speed, w_angle


def part3(df_gen: pd.DataFrame,
          df_pol: pd.DataFrame,
          df_speed: pd.DataFrame,
          df_angle: pd.DataFrame,
          geo_lev: str,
          time_lev: str,
          save_data: bool = False) -> (pd.DataFrame, np.ndarray):
    """
    :param save_data:
    :param df_gen: cleaned dataset
    :param df_pol: dataset of pollution levels
    :param df_speed: dataset of wind speeds
    :param df_angle: dataset of wind directions
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: dataframe with spatial spillover effects
    """

    coordinates = SpatialTools.coordinate_dict(df_gen, geo_lev, df_pol)
    weight_matrix, angle_matrix = SpatialTools.weight_angle_matrix(coordinates)

    wind_spillover_matrix, tensor_w = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed,
                                                                  weight_matrix, angle_matrix,
                                                                  tensor_type="wind")
    space_spillover_matrix = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed, weight_matrix,
                                                         angle_matrix, tensor_type="space")

    if save_data:
        np.save("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/" + "tensor_W.npy", tensor_w)
        wind_spillover_matrix.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/"
                                     + "spillover_effects_wind.csv")
        space_spillover_matrix.to_csv("/Users/main/Vault/Thesis/Data/" + time_lev + "/" + geo_lev + "/"
                                      + "spillover_effects_space.csv")
    return wind_spillover_matrix, space_spillover_matrix, weight_matrix, tensor_w


def part4(spillovers: pd.DataFrame, restricted: bool = False) -> VARResults:
    """
    :param restricted:
    :param spillovers: dataset with spatial spillover effects
    :return: VAR model
    """
    if restricted:
        spatial_model = SpatialRegression.restricted_spatial_VAR(spillovers)
    else:
        spatial_model = SpatialRegression.spatial_VAR(spillovers)
    return spatial_model
