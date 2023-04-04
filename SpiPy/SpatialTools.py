from __future__ import annotations
from typing import Any
from geopy.distance import great_circle
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
import numpy as np
import math


def get_bearing(coor1: float, coor2: float) -> float:
    """
    :param coor1: Pairs of longitude and latitude of the first location
    :param coor2: Latitute
    :return:
    """
    d_lon = (coor2[1] - coor1[1])
    y = math.sin(d_lon) * math.cos(coor2[0])
    x = math.cos(coor1[0]) * math.sin(coor2[0]) - math.sin(coor1[0]) * math.cos(coor2[0]) * math.cos(d_lon)
    brng = math.atan2(y, x)
    brng = np.rad2deg(brng)
    return brng


def get_data(path_df:pd.DataFrame, path_pol:pd.DataFrame, path_angle: str, path_wind: str) -> tuple[
             DataFrame | Any, DataFrame | Any, DataFrame | Any, DataFrame | Any]:
    """
    :param path_df: filepath to the clean data
    :param path_pol: filepath to the pollution data
    :param path_angle: filepath to the wind direction data
    :param path_wind: filepath to the wind speed data
    :return: Returns Pandas DataFrames for each variable
    """
    df = pd.read_csv(path_df, index_col=0)
    pol = pd.read_csv(path_pol, index_col=0)
    angle = pd.read_csv(path_angle, index_col=0)
    wind = pd.read_csv(path_wind, index_col=0)
    return df, pol, angle, wind


def coordinate_dict(df: pd.DataFrame, geo_level: str, pol: pd.DataFrame):
    """
    :param df: input dataset
    :param geo_level: variable that defines the granularity of the geographical divisions
    :param pol: dataset with the pollution level data
    :return: A dictionary with all the locations and their averaged coordinates
    """
    if geo_level == "street":
        geo_att = "name"
    else:
        geo_att = "tag"

    locations = list(pol.columns)
    c_dict = dict()

    for item in locations:
        c_dict[item] = [df.loc[df[geo_att] == item, 'latitude'].mean(),
                        df.loc[df[geo_att] == item, 'longitude'].mean()]

    return c_dict


def weight_angle_matrix(loc_dict: dict) -> tuple[ndarray, ndarray]:
    """
    :param loc_dict: Dictionary with location names and corresponding coordinates
    :return: Two matrices, one that contains the inverse of the distance between two points,
             and the other contains the bearing between both points
    """
    w_matrix = np.zeros((len(loc_dict), len(loc_dict)))
    angle_matrix = np.zeros((len(loc_dict), len(loc_dict)))
    locations = list(loc_dict.keys())

    for i in range(len(loc_dict)):

        for j in range(len(loc_dict)):

            if i != j:
                theta = get_bearing(loc_dict[locations[i]], loc_dict[locations[j]])
                w_matrix[i, j] = 1 / great_circle(loc_dict[locations[i]], loc_dict[locations[j]]).km
                angle_matrix[i, j] = theta
            else:
                w_matrix[i, i] = 0
                angle_matrix[i, i] = 0

    return w_matrix, angle_matrix


def spatial_tensor(pol: pd.DataFrame, angle: pd.DataFrame, wind: pd.DataFrame,
                   w_matrix: np.ndarray, angle_matrix: np.ndarray, tensor_type: str) -> (pd.DataFrame, np.ndarray):
    """
    :param pol: dataset of pollution levels
    :param angle: dataset of wind direction
    :param wind: dataset of wind speed
    :param w_matrix: spatial weight matrix of the inverse distance between locations
    :param angle_matrix: matrix with the bearing of two locations
    :param tensor_type: conditional to see if wind should be included in the calculations or just distance
    :return: dataframe with the spatial spillovers and possibly a tensor with the spatial interaction tensor
             between time variant wind speed, wind direction and the inverse distance of the locations
    """

    if tensor_type == "wind":
        ww_tensor = np.zeros((len(angle), len(angle.columns), len(angle.columns)))
        wwy = np.zeros((len(angle), len(pol.columns)))

        for i in range(len(angle)):
            time_angle = angle.iloc[i, :].to_numpy()
            time_speed = wind.iloc[i, :].to_numpy()
            ww_tensor[i, :, :] = np.cos(angle_matrix - time_angle[np.newaxis, :]) * time_speed[np.newaxis, :] * w_matrix

            for j in range(len(angle.columns)):
                ww_tensor[i, j, j] = 1

            wwy[i, :] = ww_tensor[i, :, :] @ pol.iloc[i, :].T

        wwy = pd.DataFrame(wwy)
        for i in range(len(pol.columns)):
            wwy.rename(columns={i: pol.columns[i]}, inplace=True)

        wwy.set_index(pol.index, inplace=True)
        return wwy, ww_tensor

    else:
        wwy = np.zeros((len(angle), len(pol.columns)))
        for i in range(w_matrix.shape[0]):
            w_matrix[i, i] = 0

        for i in range(len(angle)):
            wwy[i, :] = w_matrix @ pol.iloc[i, :].to_numpy()
        wwy = pd.DataFrame(wwy)

        for i in range(len(pol.columns)):
            wwy.rename(columns={i: pol.columns[i]}, inplace=True)

        wwy.set_index(pol.index, inplace=True)
        return wwy
