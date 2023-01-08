from statsmodels.tsa.stattools import adfuller, grangercausalitytests, add_constant
from statsmodels.tsa.api import AutoReg, VAR
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from tqdm import tqdm
import pandas as pd
import numpy as np
import math

data = pd.read_csv(r"C:\Users\VY72PC\OneDrive - ING\Documents\STARIMA Project\Data\pm25_weer.csv")
data.drop(data.iloc[:, 0:7], axis=1, inplace=True)
data.drop(["jaar", "maand", "weeknummer", "#STN", "timestamp", "components", "dag", "tijd", "uur", "datum", "weekdag", "H", "T", "U", "FH", "sensortype"], axis=1, inplace=True)

grouped_df = data.groupby(["YYYYMMDD", "tag"])["pm25", "longitude", "latitude", "DD"].mean().copy().reset_index()
grouped_df.rename(columns={"U":"Wind", "DD":"Angle"}, inplace=True)
grouped_df.head(5)

Locations = grouped_df["tag"].unique()
LocDict = dict()

for i in range(len(Locations)):
    LocDict[Locations[i]] = (grouped_df[grouped_df.tag == Locations[i]]["latitude"].mean(), grouped_df[grouped_df.tag == Locations[i]]["longitude"].mean())

grouped_df["Date"] = grouped_df["YYYYMMDD"].astype(str)
grouped_df.set_index("Date", inplace=True)
grouped_df.drop(columns=["YYYYMMDD", "latitude", "longitude"], inplace=True)

UniqueNames = grouped_df.tag.unique()

PolDict = {elem : pd.DataFrame() for elem in UniqueNames}
# WindDict = {elem : pd.DataFrame() for elem in UniqueNames}
AngleDict = {elem : pd.DataFrame() for elem in UniqueNames}

for key in PolDict.keys():
    PolDict[key] = grouped_df[:][grouped_df.tag == key]
    PolDict[key].rename(columns={"pm25":key}, inplace=True)
    PolDict[key].drop(["Angle"], axis=1, inplace=True)
    del PolDict[key]["tag"]

    # WindDict[key] = grouped_df[:][grouped_df.tag == key]
    # WindDict[key].rename(columns={"Wind":key}, inplace=True)
    # WindDict[key].drop(["pm25", "Angle"], axis=1, inplace=True)
    # del WindDict[key]["tag"]

    AngleDict[key] = grouped_df[:][grouped_df.tag == key]
    AngleDict[key].rename(columns={"Angle":key}, inplace=True)
    AngleDict[key].drop(["pm25"], axis=1 , inplace=True)
    del AngleDict[key]["tag"]

df_pol = pd.DataFrame(PolDict["Amsterdam"].copy())
# df_wind = pd.DataFrame(WindDict["Amsterdam"].copy())
df_angle = pd.DataFrame(AngleDict["Amsterdam"].copy())

for key in PolDict:
    df_pol = df_pol.combine_first(PolDict[key])
    # df_wind = df_wind.combine_first(WindDict[key])
    df_angle = df_angle.combine_first(AngleDict[key])

for column in df_pol:
    median_value = (df_pol[column].median(), df_angle[column].median())  #, df_wind[column].median())
    df_pol[column].fillna(value=median_value[0], inplace = True)
    df_angle[column].fillna(value=median_value[1], inplace = True)
    # df_wind[column].fillna(value=median_value[2], inplace = True)
    
VARModel = VAR(df_pol, ).fit(trend="n")
VARModel.summary()

for key in PolDict:
    R2 = skm.r2_score(VARModel.fittedvalues[key] + VARModel.resid[key], VARModel.fittedvalues[key])
    print(F'The R-Squared of {key} is: {R2*100:.2f}%')

WY = pd.DataFrame(np.matmul(df_pol.to_numpy(), W))

i = 0
for key in PolDict:
    WY.rename(columns={i:f'{key}'}, inplace=True)
    i += 1

SVAR = VAR(WY).fit(trend="n")
SVAR.summary()

for key in PolDict:
    R2 = skm.r2_score(SVAR.fittedvalues[key] + SVAR.resid[key], SVAR.fittedvalues[key])
    print(F'The R-Squared of {key} is: {R2*100:.2f}%')

    WW = list()

wind = np.random.lognormal(mean=2.5, sigma=0.5, size=len(df_angle))

for i in range(len(df_angle)):
    wind_direction = np.zeros((len(df_angle.columns), len(df_angle.columns)))
    
    for j in range(len(df_angle.columns)):
        wind_direction[j, :] = AngleMatrix[j, :] - df_angle.iloc[i, j]
        wind_direction = (np.cos(wind_direction) * W)  * wind[i]
        wind_direction = np.nan_to_num(wind_direction, nan=0, posinf=0, neginf=0)

    WW.append(wind_direction)

WWY = np.zeros((len(df_pol), len(df_pol.columns)))

for i in range(len(df_pol)):
    WWY[i, :] = np.matmul(df_pol.iloc[i, :].to_numpy(), WW[i])

WWY = pd.DataFrame(WWY)

i = 0
for key in PolDict:
    WWY.rename(columns={i:f'{key}'}, inplace=True)
    i += 1

EXOG = pd.DataFrame(np.concatenate((df_pol, WWY), axis=1))

i = 0
for key in PolDict:
    EXOG.rename(columns={i:f'{key}'}, inplace=True)
    EXOG.rename(columns={i + 11:f'Spatial - {key}'}, inplace=True)
    i += 1

SWVAR = VAR(EXOG).fit(trend="n")
SWVAR.summary()

for key in PolDict:
    R2 = skm.r2_score(SWVAR.fittedvalues[key] + SWVAR.resid[key], SWVAR.fittedvalues[key])
    print(F'The R-Squared of {key} is: {R2*100:.2f}%')