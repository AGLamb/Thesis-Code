import pandas as pd
import numpy as np
import math


def get_data(filepath: str):
    """
    This is a temporal function while using a snapshot of the data being gathered by
    Hollandse Luchten. The upcoming update substitutes this functions with a connection
    to the API. This will allow for constant sourcing of data.

    :param filepath: Local filepath to the raw dataset
    :return: Pandas DataFrame with the raw data
    """
    return pd.read_csv(filepath)


def format_data(df):
    """

    :param df: Raw data from Hollandse Luchten formatting
    :return: Source data set with variables of interest
    """

    df["FH"] = df["FH"] * 0.36  # To be refactored in a self-contained function
    df.drop(['id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min', 'pm25_cal',
            'datum', 'tijd', 'pm25_fac', 'pm25_max', 'pm25_min', 'components', 'sensortype',
            'weekdag', 'uur', '#STN', 'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U'],
            axis=1, inplace=True)
    for row in df.itertuples():
        while row.DD > 360:
            row.DD -= 360
    df.rename(columns={"DD": " Wind Angle", "FH": "Wind Speed"}, inplace=True)
    return df


def group_data(df, geo_level: str, time_interval: str):

    """
    :param df:
    :param geo_level:
    :param time_interval:
    :return:
    """

    if geo_level == "street":
        geo_group = "name"
    else:
        geo_group = "tag"

    if time_interval == "days":
        time_group = "YYYYMMDD"
    else:
        time_group = "timestamp"

    grouped_df = df.groupby(by=[geo_group, time_group]).median().copy().reset_index()
    grouped_df["Date"] = pd.to_datetime(grouped_df[time_group].astype(str))
    grouped_df.drop(columns=["YYYYMMDD", "timestamp"], inplace=True)
    grouped_df.set_index("Date", inplace=True)
    return grouped_df


def main() -> None:
    path = '/Users/main/Vault/Thesis/Data/pm25_weer.csv'
    geographical_level = "municipality"
    time_level = "hours"
    data = group_data(format_data(get_data(path)), geographical_level, time_level)
    data.to_csv('/Users/main/Vault/Thesis/Code/Data/Cleaned_data.csv')
    return


if __name__ == "__main__":
    main()

