from __future__ import annotations

from numpy import rad2deg, ndarray, zeros, newaxis, isnan, minimum, maximum
from geopy.distance import geodesic
from numpy.linalg import eigvals
from pandas import DataFrame
from math import sin, atan2
from numpy import cos, clip


def get_bearing(coor1: float, coor2: float) -> float:
    d_lon = (coor2[1] - coor1[1])
    y = sin(d_lon) * cos(coor2[0])
    x = cos(coor1[0]) * sin(coor2[0]) - sin(coor1[0]) * cos(coor2[0]) * cos(d_lon)
    brng = atan2(y, x)
    brng = rad2deg(brng)
    return brng


def coordinate_dict(df: DataFrame, geo_level: str, pol: DataFrame):
    locations = list(pol.columns)
    c_dict = dict()
    for item in locations:
        c_dict[item] = [df.loc[df[geo_level] == item, 'latitude'].mean(),
                        df.loc[df[geo_level] == item, 'longitude'].mean()]
    return c_dict


def weight_angle_matrix(loc_dict: dict) -> tuple[ndarray, ndarray]:
    n = len(loc_dict)
    w_matrix = zeros((n, n))
    angle_matrix = zeros((n, n))

    for i, value1 in enumerate(loc_dict):
        for j, value2 in enumerate(loc_dict):
            if i != j:
                theta = get_bearing(loc_dict[value1], loc_dict[value2])
                w_matrix[i, j] = 1 / geodesic(loc_dict[value1], loc_dict[value2]).km
                angle_matrix[i, j] = theta
            else:
                w_matrix[i, i] = 0
                angle_matrix[i, i] = 0
    return w_matrix, angle_matrix


def spatial_tensor(pol: DataFrame,
                   angle: DataFrame,
                   wind: DataFrame,
                   w_matrix: ndarray,
                   angle_matrix: ndarray
                   ) -> tuple[DataFrame, DataFrame, ndarray, ndarray]:

    t = len(angle)
    n = len(angle.columns)

    ww_tensor = zeros((t, n, n))
    Z = zeros((t, n, n))
    wwy_wind = zeros((t, n))
    wwy_space = zeros((t, n))

    for i in range(t):
        time_angle = angle.iloc[i, :].to_numpy()
        time_speed = wind.iloc[i, :].to_numpy()

        # This function is from the module math, so it might now work; substitute to numpy then
        ww_tensor[i, :, :] = cos(angle_matrix - time_angle[newaxis, :])
        ww_tensor[i, :, :] = ww_tensor[i, :, :] * time_speed[newaxis, :]
        ww_tensor[i, :, :] = ww_tensor[i, :, :] * w_matrix

        # Change method to spectral radius normalization
        ww_tensor[i, :, :] = ww_tensor[i, :, :] / max(eigvals(ww_tensor[i, :, :]))
        ww_tensor[i, :, :] = clip(ww_tensor[i, :, :], a_min=0, a_max=None)

        wwy_wind[i, :] = ww_tensor[i, :, :] @ pol.iloc[i, :].T
        wwy_space[i, :] = (w_matrix / max(eigvals(w_matrix))) @ pol.iloc[i, :].to_numpy()

        Z[i, :, :] = minimum(ww_tensor[i, :, :], w_matrix) / maximum(ww_tensor[i, :, :], w_matrix)

    wwy_wind = DataFrame(wwy_wind)
    wwy_space = DataFrame(wwy_space)
    for i, _ in enumerate(pol.columns):
        wwy_wind.rename(columns={i: pol.columns[i]}, inplace=True)
        wwy_space.rename(columns={i: pol.columns[i]}, inplace=True)

    wwy_wind.set_index(pol.index, inplace=True)
    wwy_space.set_index(pol.index, inplace=True)
    return wwy_wind, wwy_space, ww_tensor, Z


def has_nan(matrix):
    return isnan(matrix).any()
