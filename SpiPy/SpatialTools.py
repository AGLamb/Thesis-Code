from geopy.distance import great_circle
from tqdm import tqdm
import pandas as pd
import numpy as np
import math


def get_bearing(coor1, coor2) -> float:
    dLon = (coor2[1] - coor1[1])
    y = math.sin(dLon) * math.cos(coor2[0])
    x = math.cos(coor1[0]) * math.sin(coor2[0]) - math.sin(coor1[0]) * math.cos(coor2[0]) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = np.rad2deg(brng)
    return brng


def get_data(path_df, path_pol, path_angle, path_wind) -> pd.DataFrame:
    df = pd.read_csv(path_df, index_col=0)
    pol = pd.read_csv(path_pol, index_col=0)
    angle = pd.read_csv(path_angle, index_col=0)
    wind = pd.read_csv(path_wind, index_col=0)
    return df, pol, angle, wind


def coordinate_dict(df, geo_level, pol):
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


def weight_angle_matrix(loc_dict) -> np.ndarray:
    W = np.zeros((len(loc_dict), len(loc_dict)))
    AngleMatrix = np.zeros((len(loc_dict), len(loc_dict)))
    locations = list(loc_dict.keys())

    for i in range(len(loc_dict)):

        for j in range(len(loc_dict)):

            if i != j:
                theta = get_bearing(loc_dict[locations[i]], loc_dict[locations[j]])
                W[i, j] = 1 / great_circle(loc_dict[locations[i]], loc_dict[locations[j]]).km
                AngleMatrix[i, j] = theta
            else:
                W[i, j] = 0

    return W, AngleMatrix


def spatial_tensor(pol: pd.DataFrame, angle: pd.DataFrame, wind: pd.DataFrame,
                   W_matrix: np.ndarray, AngleMatrix: np.ndarray, tensor_type: str) -> (pd.DataFrame, np.ndarray):

    if tensor_type == "wind":
        WW = np.zeros((len(angle), len(angle.columns), len(angle.columns)))
        WWY = np.zeros((len(angle), len(pol.columns)))

        for i in tqdm(range(len(angle))):
            time_angle = angle.iloc[i, :].to_numpy().reshape(len(angle.columns), 1)
            time_speed = wind.iloc[i, :].to_numpy().reshape(len(angle.columns), 1)
            WW[i, :, :] = np.cos(AngleMatrix - time_angle[np.newaxis, :]) * time_speed[np.newaxis, :] * W_matrix
            WWY[i, :] = np.matmul(WW[i, :, :], pol.iloc[i, :].to_numpy())
        WWY = pd.DataFrame(WWY)

        for i in range(len(pol.columns)):
            WWY.rename(columns={i: 'Spatial ' + pol.columns[i]}, inplace=True)

        WWY.set_index(pol.index, inplace=True)

        return WWY, WW

    else:
        WWY = np.zeros((len(angle), len(pol.columns)))

        for i in tqdm(range(len(angle))):
            WWY[i, :] = W_matrix @ pol.iloc[i, :].to_numpy()
        WWY = pd.DataFrame(WWY)

        for i in range(len(pol.columns)):
            WWY.rename(columns={i: 'Spatial ' + pol.columns[i]}, inplace=True)

        WWY.set_index(pol.index, inplace=True)

        return WWY



