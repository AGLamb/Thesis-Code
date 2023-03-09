from statsmodels.tsa.vector_ar.var_model import VARResults

from SpiPy import DataPrep, Separator, SpatialTools, SpatialRegression, ModelConfidenceSet
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np


def Part1(filepath: str, geo_lev: str, time_lev: str) -> pd.DataFrame:
    """
    :param filepath: local filepath to the raw dataset
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: grouped data at the desired geographical level and time interval
    """
    data = DataPrep.group_data(DataPrep.format_data(DataPrep.get_data(filepath)), geo_lev, time_lev)
    # data.to_csv("../Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_data.csv")
    data.to_csv("Cleaned_data.csv")

    return data


def Part2(geo_lev: str, time_lev: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time interval
    :return: separated dataframes for each variable
    """
    # filepath = "../Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_data.csv"
    filepath = r'Cleaned_data.csv'
    data = Separator.get_clean_data(filepath)

    no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]
    pollution, w_speed, w_angle = Separator.matrix_creator(data, geo_lev, no_sensors)
    # pollution.to_csv("../Data/" + time_lev + "/" + geo_lev + "/" + "pollution.csv")
    # w_speed.to_csv("../Data/" + time_lev + "/" + geo_lev + "/" + "wind_speed.csv")
    # w_angle.to_csv("../Data/" + time_lev + "/" + geo_lev + "/" + "wind_angle.csv")
    return pollution, w_speed, w_angle


def Part3(df_gen: pd.DataFrame, df_pol: pd.DataFrame, df_speed: pd.DataFrame,
          df_angle: pd.DataFrame, geo_lev: str, time_lev: str, tensor_typ: str) -> pd.DataFrame:
    """
    :param df_gen: clean dataset
    :param df_pol: dataset with pollution data
    :param df_speed: dataset with wind speed
    :param df_angle:dataset with wind direction
    :param geo_lev: granularity of geographical divison
    :param time_lev: granularity of time interval
    :param tensor_typ: conditional that determines if to include wind varibles in the calculations
    :return: spatial spillover dataframe
    """
    # filepath_cleaned = "./Data/" + time_lev + "/" + geo_lev + "/" + "Cleaned_data.csv"
    # filepath_pol = "./Data/" + time_lev + "/" + geo_lev + "/" + "pollution.csv"
    # filepath_speed = "./Data/" + time_lev + "/" + geo_lev + "/" + "wind_speed.csv"
    # filepath_angle = "./Data/" + time_lev + "/" + geo_lev + "/" + "wind_angle.csv"
    # df_gen, df_pol, df_speed, df_angle = Module3.get_data(filepath_cleaned,
    #                                                       filepath_pol,
    #                                                       filepath_speed,
    #                                                       filepath_angle)

    coordinates = SpatialTools.coordinate_dict(df_gen, geo_lev, df_pol)
    weight_matrix, angle_matrix = SpatialTools.weight_angle_matrix(coordinates)

    if tensor_typ == "wind":
        spillover_matrix, tensor_W = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed,
                                                                 weight_matrix, angle_matrix,
                                                                 tensor_type=tensor_typ)
        # np.save("../Data/" + time_lev + "/" + geo_lev + "/" + "tensor_W.npy", tensor_W)
    else:
        spillover_matrix = SpatialTools.spatial_tensor(df_pol, df_angle, df_speed, weight_matrix,
                                                       angle_matrix, tensor_type=tensor_typ)

    # spillover_matrix.to_csv("../Data/" + time_lev + "/" + geo_lev + "/" + "spillover_effects" + tensor_typ + ".csv")
    return spillover_matrix


def Part4(pollution: pd.DataFrame, WWY: pd.DataFrame, geo_lev: str, time_lev: str, tensor_typ: str) -> VARResults:
    """
    :param pollution: dataset with the pollution levels
    :param WWY: dataset with the spatial spillover effects
    :param geo_lev: granularity of the geographical division
    :param time_lev: granularity of the time intervals
    :param tensor_typ: conditional that determines if to include wind varibles in the calculations
    :return: VAR model
    """
    # filepath_pol = "./Data/" + time_lev + "/" + geo_lev + "/" + "pollution.csv"
    # filepath_spill = "./Data/" + time_lev + "/" + geo_lev + "/" + "spillover_effects" + tensor_typ + ".csv"
    # df_pol, WWY = SpatialRegression.spatial_data(filepath_pol, filepath_spill)

    spatial_model = SpatialRegression.spatial_VAR(pollution, WWY)
    print(spatial_model.test_normality(signif=0.05).summary())
    return spatial_model
