from hampel import hampel
import pandas as pd
import numpy as np


def get_clean_data(path: str):
    return pd.read_csv(path, index_col=0)


def matrix_creator(df, geo_level: str, faulty: list):
    if geo_level == "street":
        geo_att = "name"
    else:
        geo_att = "tag"

    df.drop(columns=["latitude", "longitude"], inplace=True)
    UniqueNames = df[geo_att].unique()

    df_pol = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "pm25"])
    df_pol.rename(columns={"pm25": UniqueNames[0]}, inplace=True)
    df_wind = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "Wind Speed"])
    df_wind.rename(columns={"Wind Speed": UniqueNames[0]}, inplace=True)
    df_angle = pd.DataFrame(df.loc[df[geo_att] == UniqueNames[0], "Wind Angle"])
    df_angle.rename(columns={"Wind Angle": UniqueNames[0]}, inplace=True)

    for i in range(1, len(UniqueNames)):
        df_pol = df_pol.combine_first(pd.DataFrame(df.loc[df["tag"] == UniqueNames[i], "pm25"]))
        df_pol.rename(columns={"pm25": UniqueNames[i]}, inplace=True)

        df_wind = df_wind.combine_first(pd.DataFrame(df.loc[df["tag"] == UniqueNames[i], "Wind Speed"]))
        df_wind.rename(columns={"Wind Speed": UniqueNames[i]}, inplace=True)

        df_angle = df_angle.combine_first(pd.DataFrame(df.loc[df["tag"] == UniqueNames[i], "Wind Angle"]))
        df_angle.rename(columns={"Wind Angle": UniqueNames[i]}, inplace=True)

    for column in df_pol:
        median_values = (df_pol[column].median(), df_angle[column].median(), df_wind[column].median())
        df_pol[column].fillna(value=median_values[0], inplace=True)
        df_angle[column].fillna(value=median_values[1], inplace=True)
        df_wind[column].fillna(value=median_values[2], inplace=True)

    return delete_sensors(df_pol, faulty), delete_sensors(df_wind, faulty), delete_sensors(df_angle, faulty)


def delete_sensors(df, pop_sensors: list):
    df.drop(columns=pop_sensors, inplace=True)
    return df


def main():
    filepath = '/Users/main/Vault/Thesis/Code/Data/Cleaned_data.csv'
    geographical = "municipality"

    data = get_clean_data(filepath)
    no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]
    pollution, w_speed, w_angle = matrix_creator(data, geographical, no_sensors)

    pollution.to_csv('/Users/main/Vault/Thesis/Code/Data/pollution.csv')
    w_speed.to_csv('/Users/main/Vault/Thesis/Code/Data/wind_speed.csv')
    w_angle.to_csv('/Users/main/Vault/Thesis/Code/Data/wind_angle.csv')
    return


if __name__ == "__main__":
    main()
