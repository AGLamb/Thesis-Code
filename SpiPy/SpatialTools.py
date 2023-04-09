from __future__ import annotations
from geopy.distance import geodesic as GD
from numpy import ndarray
import pandas as pd
import numpy as np
import math


def get_bearing(coor1: float, coor2: float) -> float:
    d_lon = (coor2[1] - coor1[1])
    y = math.sin(d_lon) * math.cos(coor2[0])
    x = math.cos(coor1[0]) * math.sin(coor2[0]) - math.sin(coor1[0]) * math.cos(coor2[0]) * math.cos(d_lon)
    brng = math.atan2(y, x)
    brng = np.rad2deg(brng)
    return brng


def coordinate_dict(df: pd.DataFrame, geo_level: str, pol: pd.DataFrame):
    locations = list(pol.columns)
    c_dict = dict()
    for item in locations:
        c_dict[item] = [df.loc[df[geo_level] == item, 'latitude'].mean(),
                        df.loc[df[geo_level] == item, 'longitude'].mean()]
    return c_dict


def weight_angle_matrix(loc_dict: dict) -> tuple[ndarray, ndarray]:
    w_matrix = np.zeros((len(loc_dict), len(loc_dict)))
    angle_matrix = np.zeros((len(loc_dict), len(loc_dict)))
    locations = list(loc_dict.keys())

    for i in range(len(loc_dict)):
        for j in range(len(loc_dict)):
            if i != j:
                theta = get_bearing(loc_dict[locations[i]], loc_dict[locations[j]])
                w_matrix[i, j] = 1 / GD(loc_dict[locations[i]], loc_dict[locations[j]]).km
                angle_matrix[i, j] = theta
            else:
                w_matrix[i, i] = 0
                angle_matrix[i, i] = 0
    return w_matrix, angle_matrix


def spatial_tensor(pol: pd.DataFrame,
                   angle: pd.DataFrame,
                   wind: pd.DataFrame,
                   w_matrix: np.ndarray,
                   angle_matrix: np.ndarray) -> (pd.DataFrame, np.ndarray):

    ww_tensor = np.zeros((len(angle), len(angle.columns), len(angle.columns)))
    wwy_wind = np.zeros((len(angle), len(pol.columns)))
    wwy_space = np.zeros((len(angle), len(pol.columns)))

    row_sums = w_matrix.sum(axis=1)
    w_matrix = w_matrix / row_sums[:, np.newaxis]

    for i in range(len(angle)):
        time_angle = angle.iloc[i, :].to_numpy()
        time_speed = wind.iloc[i, :].to_numpy()
        ww_tensor[i, :, :] = np.cos(angle_matrix - time_angle[np.newaxis, :])
        ww_tensor[i, :, :] = ww_tensor[i, :, :] * time_speed[np.newaxis, :]
        ww_tensor[i, :, :] = ww_tensor[i, :, :] * w_matrix

        row_sums = ww_tensor[i, :, :].sum(axis=1)
        ww_tensor[i, :, :] = ww_tensor[i, :, :] / row_sums[:, np.newaxis]

        wwy_wind[i, :] = ww_tensor[i, :, :] @ pol.iloc[i, :].T
        wwy_space[i, :] = w_matrix @ pol.iloc[i, :].to_numpy()

    wwy_wind = pd.DataFrame(wwy_wind)
    wwy_space = pd.DataFrame(wwy_space)
    for i in range(len(pol.columns)):
        wwy_wind.rename(columns={i: pol.columns[i]}, inplace=True)
        wwy_space.rename(columns={i: pol.columns[i]}, inplace=True)

    wwy_wind.set_index(pol.index, inplace=True)
    wwy_space.set_index(pol.index, inplace=True)
    return wwy_wind, wwy_space, ww_tensor


def has_nan(matrix):
    return np.isnan(matrix).any()
