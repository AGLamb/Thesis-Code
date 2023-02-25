from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.api import AutoReg, VAR, VARMAX
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from hampel import hampel
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


Locations = grouped_df["tag"].unique()
LocDict = dict()

for i in range(len(Locations)):
    LocDict[Locations[i]] = (grouped_df[grouped_df.tag == Locations[i]]["latitude"].mean(),
                             grouped_df[grouped_df.tag == Locations[i]]["longitude"].mean())


LocDict.pop('Velsen-Zuid')
LocDict.pop('Uithoorn')
LocDict.pop('Koog aan de Zaan')
LocDict.pop('Wijk aan Zee')
Locations = np.delete(Locations, [10, 9, 7, 3])


W = np.zeros((len(LocDict), len(LocDict)))
AngleMatrix = np.zeros((len(LocDict), len(LocDict)))

for i in range(len(LocDict)):
    for j in range(len(LocDict)):
        if i != j:
            theta = get_bearing(LocDict[Locations[i]], LocDict[Locations[j]])
            W[i, j] = great_circle(LocDict[Locations[i]], LocDict[Locations[j]]).km
            AngleMatrix[i, j] = theta
#