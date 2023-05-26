from __future__ import annotations

from pandas import DataFrame, read_csv, to_datetime
from fancyimpute import IterativeImputer
from SpiPy.Utils import SpatialTools
from DTO.Database import HLDatabase
from itertools import product
from hampel import hampel
from numpy import save


class DataBase:
    def __init__(self,
                 df: DataFrame,
                 faulty: list,
                 geo_level: str,
                 time_interval: str) -> None:

        self.data = df
        self.faulty_sensors = faulty
        self.geo_lev = geo_level
        self.time_lev = time_interval
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
        self.X = None
        self.Y = None

    def run(self) -> None:
        self.matrix_creator()
        return None

    def matrix_creator(self) -> None:
        unique_names = self.data[self.geo_lev].unique()

        self.pollution = DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[0], "pm25_cal"])
        self.pollution.rename(columns={"pm25_cal": unique_names[0]}, inplace=True)
        self.wind_speed = DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[0], "Wind Speed"])
        self.wind_speed.rename(columns={"Wind Speed": unique_names[0]}, inplace=True)
        self.wind_direction = DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[0], "Wind Angle"])
        self.wind_direction.rename(columns={"Wind Angle": unique_names[0]}, inplace=True)

        for i in range(1, len(unique_names)):
            self.pollution = self.pollution.combine_first(
                DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[i], "pm25_cal"])
            )
            self.pollution.rename(columns={"pm25_cal": unique_names[i]}, inplace=True)

            self.wind_speed = self.wind_speed.combine_first(
                DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[i], "Wind Speed"])
            )
            self.wind_speed.rename(columns={"Wind Speed": unique_names[i]}, inplace=True)

            self.wind_direction = self.wind_direction.combine_first(
                DataFrame(self.data.loc[self.data[self.geo_lev] == unique_names[i], "Wind Angle"])
            )
            self.wind_direction.rename(columns={"Wind Angle": unique_names[i]}, inplace=True)

        Locations = self.pollution.columns
        imp_pol = IterativeImputer(max_iter=10, random_state=0)
        imp_pol.fit(self.pollution.iloc[2000:])
        self.pollution = DataFrame(imp_pol.transform(self.pollution.iloc[2000:]))
        self.pollution.columns = Locations

        imp_wind_speed = IterativeImputer(max_iter=10, random_state=0)
        imp_wind_speed.fit(self.pollution.iloc[2000:])
        self.wind_speed = DataFrame(imp_wind_speed.transform(self.pollution.iloc[2000:]))
        self.wind_speed.columns = Locations

        imp_wind_dir = IterativeImputer(max_iter=10, random_state=0)
        imp_wind_dir.fit(self.pollution.iloc[2000:])
        self.wind_direction = DataFrame(imp_wind_dir.transform(self.pollution.iloc[2000:]))
        self.wind_direction.columns = Locations

        for column in self.pollution:
            self.pollution[column] = hampel(self.pollution[column], window_size=12, n=3, imputation=True)

        self.pollution = invalid_values(self.pollution)
        self.wind_speed = invalid_values(self.wind_speed)
        self.wind_direction = invalid_values(self.wind_direction)
        return None

    def SpatialComponents(self) -> None:
        self.coordinate_dict = SpatialTools.coordinate_dict(df=self.data, geo_level=self.geo_lev, pol=self.pollution)
        self.weight_matrix, self.angle_matrix = SpatialTools.weight_angle_matrix(self.coordinate_dict)
        self.wSpillovers, self.sSpillovers, \
            self.weight_tensor, self.Z, \
            self.X, self.Y = SpatialTools.spatial_tensor(self.pollution,
                                                         self.wind_direction,
                                                         self.wind_speed,
                                                         self.weight_matrix,
                                                         self.angle_matrix)
        return None

    def delete_entries(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.data = self.data[~self.data[key].isin(pop_values)]
        else:
            pass


class RunFlow:
    def __init__(self,
                 geo_level: str,
                 time_interval: str,
                 save_data: bool = False,
                 bWorkLaptop: bool = False,
                 ) -> None:
        self.save_data = save_data
        self.geo_lev = "name" if geo_level == "street" else "tag"
        self.time_lev = "YYYYMMDD" if time_interval == "day" else "timestamp"
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        self.bWorkLaptop = bWorkLaptop

    def get_data(self, path: str, faulty: set | list) -> None:
        self.raw_data = read_csv(path)
        self.format_data(faulty=faulty)
        self.group_data()
        self.processed_data = DataBase(df=self.raw_data,
                                       faulty=faulty,
                                       geo_level=self.geo_lev,
                                       time_interval=self.time_lev)
        self.processed_data.run()
        return None

    def format_data(self, faulty: set | list) -> None:
        self.raw_data["FH"] *= 0.36
        self.raw_data.drop(['id', 'no2', 'pm10', 'pm10_cal', 'pm10_fac', 'pm10_max', 'pm10_min', 'pm25',
                            'datum', 'tijd', 'pm25_fac', 'pm25_max', 'pm25_min', 'components', 'sensortype',
                            'weekdag', 'uur', '#STN', 'jaar', 'maand', 'weeknummer', 'dag', 'H', 'T', 'U'],
                           axis=1, inplace=True)

        for i in range(len(self.raw_data)):
            self.raw_data.at[i, 'DD'] = angle_correction(self.raw_data.at[i, 'DD'])

        self.raw_data.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
        self.delete_entries(pop_values=faulty, key=self.geo_lev)
        return None

    def delete_entries(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.raw_data = self.raw_data[~self.raw_data[key].isin(pop_values)]
        else:
            pass

    def group_data(self) -> None:
        grouped_df = self.raw_data.groupby(by=[self.geo_lev, self.time_lev]).median().copy().reset_index()
        grouped_df["Date"] = to_datetime(grouped_df[self.time_lev])

        if self.time_lev == "YYYYMMDD":
            grouped_df.drop(columns=["YYYYMMDD"], inplace=True)
        else:
            grouped_df.drop(columns=["YYYYMMDD", "timestamp"], inplace=True)

        grouped_df.set_index("Date", inplace=True)
        self.raw_data = grouped_df
        return None

    def data_saver(self) -> None:

        save(r"../DTO/train_tWind",
             self.train_data.weight_tensor)
        save(r"../DTO/test_tWind",
             self.test_data.weight_tensor)
        save(r"../DTO/train_tZ",
             self.train_data.Z)
        save(r"../DTO/test_tZ",
             self.test_data.Z)
        save(r"../DTO/train_tX",
             self.train_data.X)
        save(r"../DTO/test_tX",
             self.test_data.Y)
        save(r"../DTO/train_tY",
             self.train_data.Y)
        save(r"../DTO/test_tY",
             self.test_data.Y)

        mTrainMatrix = DataFrame(self.train_data.weight_matrix)
        mTestMatrix = DataFrame(self.test_data.weight_matrix)
        for i, _ in enumerate(self.train_data.pollution.columns):
            mTrainMatrix.rename(columns={i: self.train_data.pollution.columns[i]}, inplace=True)
            mTestMatrix.rename(columns={i: self.test_data.pollution.columns[i]}, inplace=True)

        db = HLDatabase(bWorkLaptop=self.bWorkLaptop)
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
            db.add_dataframe(
                table_name=combination[0] + combination[1],
                geo_level=self.train_data.geo_lev,
                time_level=self.train_data.time_lev,
                df=data[combination[0]][combination[1]]
            )

        return None

    def run(self) -> None:
        if self.bWorkLaptop:
            path = r"C:\Users\VY72PC\PycharmProjects\Academia\Data\pm25_weer.csv"
        else:
            path = r"/Users/main/Vault/Thesis/Data/pm25_weer.csv"

        no_sensors = ["Uithoorn", "Velsen-Zuid", "Koog aan de Zaan", "Wijk aan Zee"]

        self.get_data(path=path, faulty=no_sensors)
        self.split(faulty=no_sensors)

        train_names = set(self.train_data.data[self.train_data.geo_lev].unique())
        test_names = set(self.test_data.data[self.test_data.geo_lev].unique())

        misplaced = (train_names - test_names) | (test_names - train_names)

        self.train_data.delete_entries(pop_values=misplaced, key=self.train_data.geo_lev)
        self.test_data.delete_entries(pop_values=misplaced, key=self.test_data.geo_lev)

        self.train_data.SpatialComponents()
        self.test_data.SpatialComponents()

        if self.save_data:
            self.data_saver()
        return None

    def split(self, faulty: set | list, separation: float = 0.75) -> None:
        cutout = int(len(self.processed_data.data) * separation)

        self.train_data = DataBase(df=self.processed_data.data.iloc[:cutout, :].copy(),
                                   faulty=faulty,
                                   geo_level=self.geo_lev,
                                   time_interval=self.time_lev)
        self.train_data.pollution = self.processed_data.pollution.iloc[:cutout, :].copy()
        self.train_data.wind_speed = self.processed_data.wind_speed.iloc[:cutout, :].copy()
        self.train_data.wind_direction = self.processed_data.wind_direction.iloc[:cutout, :].copy()

        self.test_data = DataBase(df=self.processed_data.data.iloc[cutout:, :].copy(),
                                  faulty=faulty,
                                  geo_level=self.geo_lev,
                                  time_interval=self.time_lev)
        self.test_data.pollution = self.processed_data.pollution.iloc[cutout:, :].copy()
        self.test_data.wind_speed = self.processed_data.wind_speed.iloc[cutout:, :].copy()
        self.test_data.wind_direction = self.processed_data.wind_direction.iloc[cutout:, :].copy()

        return


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
