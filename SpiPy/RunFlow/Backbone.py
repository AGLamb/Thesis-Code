from __future__ import annotations

from pandas import DataFrame, read_csv, to_datetime
from SpiPy.Utils import SpatialTools
from DTO.Database import HLDatabase
from itertools import product
from numpy import save
import os


class DataBase:
    def __init__(self,
                 filepath: str,
                 faulty: list,
                 geo_level: str,
                 time_interval: str) -> None:

        self.data = None
        self.path = filepath
        self.faulty_sensors = faulty
        self.geo_lev = "name" if geo_level == "street" else "tag"
        self.time_lev = "YYYYMMDD" if time_interval == "day" else "timestamp"
        self.pollution = None
        self.wind_speed = None
        self.wind_direction = None
        self.weight_matrix = None
        self.angle_matrix = None
        self.weight_tensor = None
        self.sSpillovers = None
        self.wSpillovers = None
        self.coordinate_dict = None
        self.Z = None

    def run(self) -> None:
        self.matrix_creator()
        return None

    def get_data(self) -> None:
        self.data = read_csv(self.path)
        self.format_data()
        self.group_data()
        return None

    def format_data(self) -> None:

        self.data["FH"] = self.data["FH"] * 0.36
        self.data.drop(['id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min', 'pm25',
                        'datum', 'tijd', 'pm25_fac', 'pm25_max', 'pm25_min', 'components', 'sensortype',
                        'weekdag', 'uur', '#STN', 'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U'],
                       axis=1, inplace=True)

        for i in range(len(self.data)):
            self.data.at[i, 'DD'] = angle_correction(self.data.at[i, 'DD'])

        self.data.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
        self.delete_entries(pop_values=self.faulty_sensors, key=self.geo_lev)
        return None

    def delete_entries(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.data = self.data[~self.data[key].isin(pop_values)]
        else:
            pass

    def group_data(self) -> None:
        grouped_df = self.data.groupby(by=[self.geo_lev, self.time_lev]).median().copy().reset_index()
        grouped_df["Date"] = to_datetime(grouped_df[self.time_lev])

        if self.time_lev == "YYYYMMDD":
            grouped_df.drop(columns=["YYYYMMDD"], inplace=True)
        else:
            grouped_df.drop(columns=["YYYYMMDD", "timestamp"], inplace=True)

        grouped_df.set_index("Date", inplace=True)
        self.data = grouped_df
        return None

    def matrix_creator(self) -> None:
        df = self.data.drop(columns=["latitude", "longitude"]).copy(deep=True)
        unique_names = df[self.geo_lev].unique()

        df_pol = DataFrame(df.loc[df[self.geo_lev] == unique_names[0], "pm25_cal"])
        df_pol.rename(columns={"pm25_cal": unique_names[0]}, inplace=True)
        df_wind = DataFrame(df.loc[df[self.geo_lev] == unique_names[0], "Wind Speed"])
        df_wind.rename(columns={"Wind Speed": unique_names[0]}, inplace=True)
        df_angle = DataFrame(df.loc[df[self.geo_lev] == unique_names[0], "Wind Angle"])
        df_angle.rename(columns={"Wind Angle": unique_names[0]}, inplace=True)

        for i in range(1, len(unique_names)):
            df_pol = df_pol.combine_first(DataFrame(df.loc[df[self.geo_lev] == unique_names[i], "pm25_cal"]))
            df_pol.rename(columns={"pm25_cal": unique_names[i]}, inplace=True)

            df_wind = df_wind.combine_first(DataFrame(df.loc[df[self.geo_lev] == unique_names[i], "Wind Speed"]))
            df_wind.rename(columns={"Wind Speed": unique_names[i]}, inplace=True)

            df_angle = df_angle.combine_first(DataFrame(df.loc[df[self.geo_lev] == unique_names[i], "Wind Angle"]))
            df_angle.rename(columns={"Wind Angle": unique_names[i]}, inplace=True)

        for column in df_pol:
            median_values = (df_pol[column].median(), df_angle[column].median(), df_wind[column].median())
            df_pol[column].fillna(value=median_values[0], inplace=True)
            df_angle[column].fillna(value=median_values[1], inplace=True)
            df_wind[column].fillna(value=median_values[2], inplace=True)

        self.pollution = invalid_values(df_pol)
        self.wind_speed = invalid_values(df_wind)
        self.wind_direction = invalid_values(df_angle)
        return None

    def SpatialComponents(self) -> None:
        self.coordinate_dict = SpatialTools.coordinate_dict(df=self.data, geo_level=self.geo_lev, pol=self.pollution)
        self.weight_matrix, self.angle_matrix = SpatialTools.weight_angle_matrix(self.coordinate_dict)
        self.wSpillovers, self.sSpillovers, \
            self.weight_tensor, self.Z = SpatialTools.spatial_tensor(self.pollution,
                                                                     self.wind_direction,
                                                                     self.wind_speed,
                                                                     self.weight_matrix,
                                                                     self.angle_matrix)
        return None


class RunFlow:
    def __init__(self, save_data: bool = False, bWorkLaptop: bool = False) -> None:
        self.save_data = save_data
        self.train_data = None
        self.test_data = None
        self.bWorkLaptop = bWorkLaptop

    def data_saver(self) -> None:

        save(r"../DTO/train_tWind",
             self.test_data.weight_tensor)
        save(r"../DTO/test_tWind",
             self.test_data.weight_tensor)
        save(r"../DTO/train_tZ",
             self.test_data.Z)
        save(r"../DTO/test_tZ",
             self.test_data.Z)

        mTrainMatrix = DataFrame(self.train_data.weight_matrix)
        mTestMatrix = DataFrame(self.test_data.weight_matrix)
        for i, _ in enumerate(self.train_data.pollution.columns):
            mTrainMatrix.rename(columns={i: self.train_data.pollution.columns[i]}, inplace=True)
            mTestMatrix.rename(columns={i: self.test_data.pollution.columns[i]}, inplace=True)

        db = HLDatabase()
        data = {
            'Train-': {
                'All': self.train_data.data,
                'Pollution': self.train_data.pollution,
                'Wind Direction': self.train_data.wind_direction,
                'Wind Speed': self.train_data.wind_speed,
                'Anisotropic': self.train_data.wSpillovers,
                'Isotropic': self.train_data.sSpillovers,
                'Weight Matrix': mTrainMatrix
            },
            'Test-': {
                'All': self.test_data.data,
                'Pollution': self.test_data.pollution,
                'Wind Direction': self.test_data.wind_direction,
                'Wind Speed': self.test_data.wind_speed,
                'Anisotropic': self.test_data.wSpillovers,
                'Isotropic': self.test_data.sSpillovers,
                'Weight Matrix': mTestMatrix,
            }
        }

        for combination in product(data.keys(), data['Test-'].keys()):
            # We save the train data to the Database
            db.add_dataframe(
                table_name=combination[0] + combination[1],
                geo_level=self.train_data.geo_lev,
                time_level=self.train_data.time_lev,
                df=data[combination[0]][combination[1]]
            )

        return None

    def run(self, geo_lev: str, time_lev: str) -> None:
        if self.bWorkLaptop:
            path_train = r"C:\Users\VY72PC\PycharmProjects\Academia\Data\train_data.csv"
            path_test = r"C:\Users\VY72PC\PycharmProjects\Academia\Data\test_data.csv"
        else:
            path_train = r"/Users/main/Vault/Thesis/Data/Core/train_data.csv"
            path_test = r"/Users/main/Vault/Thesis/Data/Core/test_data.csv"

        no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]

        self.train_data = DataBase(filepath=path_train,
                                   faulty=no_sensors,
                                   geo_level=geo_lev,
                                   time_interval=time_lev)
        self.test_data = DataBase(filepath=path_test,
                                  faulty=no_sensors,
                                  geo_level=geo_lev,
                                  time_interval=time_lev)

        self.train_data.get_data()
        self.test_data.get_data()

        train_names = set(self.train_data.data[self.train_data.geo_lev].unique())
        test_names = set(self.test_data.data[self.test_data.geo_lev].unique())

        misplaced = (train_names - test_names) | (test_names - train_names)

        self.train_data.delete_entries(pop_values=misplaced, key=self.train_data.geo_lev)
        self.test_data.delete_entries(pop_values=misplaced, key=self.test_data.geo_lev)

        self.train_data.run()
        self.test_data.run()
        self.train_data.SpatialComponents()
        self.test_data.SpatialComponents()

        if self.save_data:
            self.data_saver()
        return None


def invalid_values(df: DataFrame) -> DataFrame:
    output_df = df.copy()
    output_df.reset_index(inplace=True, drop=True)
    for index, row in output_df.iterrows():
        if (row > 0).all():
            continue
        for col in output_df.columns:
            if row[col] <= 0:
                prev_val = output_df.iloc[max(index - 1, 0)][col]
                next_val = output_df.iloc[min(index + 1, len(output_df) - 1)][col]
                new_val = (prev_val + next_val) / 2
                output_df.at[index, col] = new_val

    output_df.set_index(df.index, inplace=True)
    return output_df


def angle_correction(angle: int) -> int:
    if angle > 360:
        angle -= 360
        angle = angle_correction(angle)
    return angle
