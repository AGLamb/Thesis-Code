from hampel import hampel
import pandas as pd
import numpy as np


def clean_data(filepath: str):
    return pd.read_csv(filepath)

# grouped_df.drop(columns=["latitude", "longitude"], inplace=True)


# UniqueNames = grouped_df.tag.unique()
#
# PolDict = {elem: pd.DataFrame() for elem in UniqueNames}
# AngleDict = {elem: pd.DataFrame() for elem in UniqueNames}
# WindDict = {elem: pd.DataFrame() for elem in UniqueNames}
#
# for key in PolDict.keys():
#     PolDict[key] = grouped_df[:][grouped_df.tag == key]
#     PolDict[key].rename(columns={"pm25": key}, inplace=True)
#     PolDict[key].drop(["Angle", "Wind"], axis=1, inplace=True)
#     del PolDict[key]["tag"]
#
#     AngleDict[key] = grouped_df[:][grouped_df.tag == key]
#     AngleDict[key].rename(columns={"Angle": key}, inplace=True)
#     AngleDict[key].drop(["pm25", "Wind"], axis=1, inplace=True)
#     del AngleDict[key]["tag"]
#
#     WindDict[key] = grouped_df[:][grouped_df.tag == key]
#     WindDict[key].rename(columns={"Wind": key}, inplace=True)
#     WindDict[key].drop(["pm25", "Angle"], axis=1, inplace=True)
#     del WindDict[key]["tag"]
#
# df_pol = pd.DataFrame()
# df_angle = pd.DataFrame()
# df_wind = pd.DataFrame()
#
# for key in PolDict:
#     df_pol = df_pol.combine_first(PolDict[key])
#     df_angle = df_angle.combine_first(AngleDict[key])
#     df_wind = df_wind.combine_first(WindDict[key])
#
# for column in df_pol:
#     median_values = (df_pol[column].median(), df_angle[column].median(), df_wind[column].median())
#     df_pol[column].fillna(value=median_values[0], inplace=True)
#     df_angle[column].fillna(value=median_values[1], inplace=True)
#     df_wind[column].fillna(value=median_values[2], inplace=True)

#
# WW = np.zeros((len(df_pol), len(df_angle.columns), len(df_angle.columns)))
# WWY = np.zeros((len(df_pol), len(df_pol.columns)))
#
# filtered_pol = df_pol.copy()
# for column in filtered_pol:
#     filtered_pol[column] = hampel(filtered_pol[column], window_size=12, n=3, imputation=True)
#
# for i in range(len(df_angle)):
#
#     for j in range(len(df_angle.columns)):
#
#         for k in range(len(df_angle.columns)):
#
#             if W[j, k] != 0:
#                 WW[i, j, k] = np.cos(AngleMatrix[j, k] - df_angle.iloc[i, j]) * df_wind.iloc[i, j] / W[j, k]
#             else:
#                 WW[i, j, k] = 0
#
#     WWY[i, :] = np.matmul(WW[i, :, :], filtered_pol.iloc[i, :].to_numpy().T)
#
