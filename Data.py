import pandas as pd
import requests
from scipy.io import netcdf


API_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjI4ZWZlOTZkNDk2ZjQ3ZmE5YjMzNWY5NDU3NWQyMzViIiwiaCI6Im11cm11cjEyOCJ9"
df = requests.get('https://api.dataplatform.knmi.nl/open-data/v1/datasets/Actuele10mindataKNMIstations/versions/2/files', headers=API_key)


# file2read = netcdf.NetCDFFile(path+'state.nc','r')
# temp = file2read.variables[var] # var can be 'Theta', 'S', 'V', 'U' etc..
# data = temp[:]*1
# file2read.close()

print(df)


