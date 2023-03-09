import pandas as pd
import numpy as np


def get_clean_data(path: str) -> pd.DataFrame:
    """
    :param path: filepath to the raw data
    :return: Pandas DataFrame with the raw data
    """
    return pd.read_csv(path, index_col=0)


def matrix_creator(input_df: pd.DataFrame, geo_level: str,
                   faulty: list) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param input_df: dataset with the raw data
    :param geo_level: granularity for the geographical division
    :param faulty: sensors that have to be eliminated
    :return: three dataframes, each containing one the individual data for a variable
    """

    if geo_level == "street":
        geo_att = "name"
    else:
        geo_att = "tag"

    df = input_df.drop(columns=["latitude", "longitude"]).copy(deep=True)
    UniqueNames = df[geo_att].unique()

    df_pol = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "pm25"])
    df_pol.rename(columns={"pm25": UniqueNames[0]}, inplace=True)
    df_wind = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "Wind Speed"])
    df_wind.rename(columns={"Wind Speed": UniqueNames[0]}, inplace=True)
    df_angle = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "Wind Angle"])
    df_angle.rename(columns={"Wind Angle": UniqueNames[0]}, inplace=True)

    for i in range(1, len(UniqueNames)):
        df_pol = df_pol.combine_first(pd.DataFrame(df.loc[df[geo_att] == UniqueNames[i], "pm25"]))
        df_pol.rename(columns={"pm25": UniqueNames[i]}, inplace=True)

        df_wind = df_wind.combine_first(pd.DataFrame(df.loc[df[geo_att] == UniqueNames[i], "Wind Speed"]))
        df_wind.rename(columns={"Wind Speed": UniqueNames[i]}, inplace=True)

        df_angle = df_angle.combine_first(pd.DataFrame(df.loc[df[geo_att] == UniqueNames[i], "Wind Angle"]))
        df_angle.rename(columns={"Wind Angle": UniqueNames[i]}, inplace=True)

    for column in df_pol:
        median_values = (df_pol[column].median(), df_angle[column].median(), df_wind[column].median())
        df_pol[column].fillna(value=median_values[0], inplace=True)
        df_angle[column].fillna(value=median_values[1], inplace=True)
        df_wind[column].fillna(value=median_values[2], inplace=True)

    return delete_sensors(df_pol, faulty), delete_sensors(df_wind, faulty), delete_sensors(df_angle, faulty)


def delete_sensors(df_input: pd.DataFrame, pop_sensors: list) -> pd.DataFrame:
    """
    :param df_input: dataset to analyse
    :param pop_sensors: list of sensors that has to be deleted
    :return: dataset without the removed sensors
    """
    if len(pop_sensors) > 0:
        df = df_input.copy()
        df.drop(columns=pop_sensors, inplace=True)
        return df
    else:
        return df_input
