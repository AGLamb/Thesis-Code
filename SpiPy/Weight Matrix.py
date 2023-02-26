from geopy.distance import great_circle
from hampel import hampel
from tqdm import tqdm
import pandas as pd
import numpy as np
import math


def get_bearing(coor1, coor2):
    dLon = (coor2[1] - coor1[1])
    y = math.sin(dLon) * math.cos(coor2[0])
    x = math.cos(coor1[0]) * math.sin(coor2[0]) - math.sin(coor1[0]) * math.cos(coor2[0]) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = np.rad2deg(brng)
    return brng


def get_data(path_df, path_pol, path_angle, path_wind):
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
        c_dict[item] = [df.loc[df[geo_att] == item, "latitude"].mean(),
                        df.loc[df[geo_att] == item, "longitude"].mean()]

    return c_dict


def weight_angle_matrix(loc_dict):
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


def wind_tensor(pol, angle, wind, W_matrix, AngleMatrix):
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


def main() -> None:
    filepath_cleaned = "/Users/main/Vault/Thesis/Code/Data/Cleaned_data.csv"
    filepath_pol = "/Users/main/Vault/Thesis/Code/Data/pollution.csv"
    filepath_speed = "/Users/main/Vault/Thesis/Code/Data/wind_speed.csv"
    filepath_angle = "/Users/main/Vault/Thesis/Code/Data/wind_angle.csv"
    geographical = "municipality"

    df_gen, df_pol, df_speed, df_angle = get_data(filepath_cleaned, filepath_pol, filepath_speed, filepath_angle)
    coordinates = coordinate_dict(df_gen, geographical, df_pol)
    weight_matrix, angle_matrix = weight_angle_matrix(coordinates)
    spillover_matrix, tensor_W = wind_tensor(df_pol, df_angle, df_speed, weight_matrix, angle_matrix)

    np.save("/Users/main/Vault/Thesis/Code/Data/tensor_W.npy", tensor_W)
    spillover_matrix.to_csv('/Users/main/Vault/Thesis/Code/Data/spillover_effects.csv')
    return None


if __name__ == "__main__":
    main()
