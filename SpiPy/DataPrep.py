import pandas as pd


def angle_correction(angle: int) -> int:
    """
    :param angle: Faulty angle to be corrected
    :return: angle in the desired range
    """
    if angle > 360:
        angle -= 360
        angle = angle_correction(angle)
    return angle


def get_data(filepath: str) -> pd.DataFrame:
    """
    This is a temporal function while using a snapshot of the data being gathered by
    Hollandse Luchten. The upcoming update substitutes this functions with a connection
    to the API. This will allow for constant sourcing of data.

    :param filepath: Local filepath to the raw dataset
    :return: Pandas DataFrame with the raw data
    """
    return pd.read_csv(filepath)


def format_data(df_input: pd.DataFrame, faulty: list) -> pd.DataFrame:
    """
    :param faulty:
    :param df_input: Raw data from Hollandse Luchten formatting
    :return: Source data set with variables of interest
    """
    df = df_input.copy()
    df["FH"] = df["FH"] * 0.36
    df.drop(['id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min', 'pm25_cal',
             'datum', 'tijd', 'pm25_fac', 'pm25_max', 'pm25_min', 'components', 'sensortype',
             'weekdag', 'uur', '#STN', 'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U'],
            axis=1, inplace=True)

    for i in range(len(df)):
        df.at[i, 'DD'] = angle_correction(df.at[i, 'DD'])

    df.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
    return delete_sensors(df, faulty)


def group_data(df: pd.DataFrame, geo_level: str, time_interval: str) -> pd.DataFrame:
    """
    :param df: cleaned dataset
    :param geo_level: granularity of the geographical division
    :param time_interval: granularity of the time interval
    :return: a grouped dataframe at the desired geographical level and time interval
    """
    if geo_level == "street":
        geo_group = "name"
    else:
        geo_group = "tag"

    if time_interval == "day":
        time_group = "YYYYMMDD"
    else:
        time_group = "timestamp"

    grouped_df = df.groupby(by=[geo_group, time_group]).median().copy().reset_index()
    grouped_df["Date"] = pd.to_datetime(grouped_df[time_group].astype(str))

    if time_interval == "day":
        grouped_df.drop(columns=["YYYYMMDD"], inplace=True)
    else:
        grouped_df.drop(columns=["YYYYMMDD", "timestamp"], inplace=True)

    grouped_df.set_index("Date", inplace=True)
    return grouped_df


def delete_sensors(df_input: pd.DataFrame, pop_sensors: list) -> pd.DataFrame:
    """
    :param df_input: dataset to analyse
    :param pop_sensors: list of sensors that has to be deleted
    :return: dataset without the removed sensors
    """
    if len(pop_sensors) > 0:
        df = df_input.copy()
        df = df[~df['tag'].isin(pop_sensors)]
        return df
    else:
        return df_input


def matrix_creator(input_df: pd.DataFrame, geo_level: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param input_df: dataset with the raw data
    :param geo_level: granularity for the geographical division
    :return: three dataframes, each containing one the individual data for a variable
    """

    if geo_level == "street":
        geo_att = "name"
    else:
        geo_att = "tag"

    df = input_df.drop(columns=["latitude", "longitude"]).copy(deep=True)
    unique_names = df[geo_att].unique()

    df_pol = pd.DataFrame(df.loc[df[geo_att] == unique_names[0], "pm25"])
    df_pol.rename(columns={"pm25": unique_names[0]}, inplace=True)
    df_wind = pd.DataFrame(df.loc[df[geo_att] == unique_names[0], "Wind Speed"])
    df_wind.rename(columns={"Wind Speed": unique_names[0]}, inplace=True)
    df_angle = pd.DataFrame(df.loc[df[geo_att] == unique_names[0], "Wind Angle"])
    df_angle.rename(columns={"Wind Angle": unique_names[0]}, inplace=True)

    for i in range(1, len(unique_names)):
        df_pol = df_pol.combine_first(pd.DataFrame(df.loc[df[geo_att] == unique_names[i], "pm25"]))
        df_pol.rename(columns={"pm25": unique_names[i]}, inplace=True)

        df_wind = df_wind.combine_first(pd.DataFrame(df.loc[df[geo_att] == unique_names[i], "Wind Speed"]))
        df_wind.rename(columns={"Wind Speed": unique_names[i]}, inplace=True)

        df_angle = df_angle.combine_first(pd.DataFrame(df.loc[df[geo_att] == unique_names[i], "Wind Angle"]))
        df_angle.rename(columns={"Wind Angle": unique_names[i]}, inplace=True)

    for column in df_pol:
        median_values = (df_pol[column].median(), df_angle[column].median(), df_wind[column].median())
        df_pol[column].fillna(value=median_values[0], inplace=True)
        df_angle[column].fillna(value=median_values[1], inplace=True)
        df_wind[column].fillna(value=median_values[2], inplace=True)

    return df_pol, df_wind, df_angle


def get_clean_data(path: str) -> pd.DataFrame:
    """
    :param path: filepath to the raw data
    :return: Pandas DataFrame with the raw data
    """
    return pd.read_csv(path, index_col=0)
