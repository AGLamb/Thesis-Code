from __future__ import annotations

from sklearn.experimental import enable_iterative_imputer
from pandas import DataFrame, read_csv, to_datetime
from sklearn.impute import IterativeImputer
from SpiPy.Utils import SpatialTools
from DTO.Database import HLDatabase
from itertools import product
from numpy import save, abs
from hampel import hampel


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
        self.coordinate_dict = None
        self.weight_tensor = None
        self.sSpillovers = None
        self.wSpillovers = None
        self.X = None
        self.Y = None

    def run(self) -> None:
        self.coordinate_dict = SpatialTools.coordinate_dict(
            df=self.data,
            geo_level=self.geo_lev
        )
        self.weight_matrix, self.angle_matrix = SpatialTools.weight_angle_matrix(self.coordinate_dict)
        self.matrix_creator()
        return None

    def create_and_impute_matrix(self, column_name):
        matrix = self.data.pivot(columns=self.geo_lev, values=column_name)
        names = matrix.columns

        n = 0 if self.time_lev == 'YYYYMMDD' else 2000
        imp = IterativeImputer(max_iter=10, random_state=0)
        matrix = DataFrame(imp.fit_transform(matrix.iloc[n:]), index=matrix.index[n:])
        matrix.columns = names

        return matrix

    def matrix_creator(self) -> None:
        self.pollution = self.create_and_impute_matrix('pm25_cal')
        self.wind_speed = self.create_and_impute_matrix('Wind Speed')
        self.wind_direction = abs(self.create_and_impute_matrix('Wind Angle'))

        unique_names = list(self.pollution.columns)
        self.wind_speed.drop(columns=unique_names[1:], inplace=True)
        self.wind_direction.drop(columns=unique_names[1:], inplace=True)

        self.wind_direction[unique_names[0]] = self.wind_direction[unique_names[0]].apply(angle_correction)

        def replace_values(val, mean):
            if val <= 0:
                return mean
            elif val > 120:
                return 2 * mean
            else:
                return val

        mean_val = self.wind_speed.mean()
        self.wind_speed = self.wind_speed.apply(replace_values, args=mean_val)

        for column in self.pollution:
            self.pollution[column] = hampel(self.pollution[column], window_size=84, n=3, imputation=True)
        self.pollution = invalid_values(self.pollution)
        self.pollution.columns = unique_names

        return None

    def SpatialComponents(self) -> None:
        names = list(self.pollution.columns)

        self.pollution = self.create_and_impute_matrix('pm25_cal')
        self.wind_speed = self.create_and_impute_matrix('Wind Speed')
        self.wind_direction = self.create_and_impute_matrix('Wind Angle')
        self.wSpillovers, self.sSpillovers, \
            self.weight_tensor, \
            self.X, self.Y = SpatialTools.spatial_tensor(pol=self.pollution,
                                                         angle=self.wind_direction,
                                                         wind=self.wind_speed,
                                                         w_matrix=self.weight_matrix,
                                                         angle_matrix=self.angle_matrix)

        wImp = IterativeImputer(max_iter=10, random_state=0)
        self.wSpillovers = DataFrame(wImp.fit_transform(self.wSpillovers), index=self.wSpillovers.index)
        self.wSpillovers.columns = names

        sImp = IterativeImputer(max_iter=10, random_state=0)
        self.sSpillovers = DataFrame(sImp.fit_transform(self.sSpillovers), index=self.sSpillovers.index)
        self.sSpillovers.columns = names
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

        self.raw_data.rename(columns={"DD": "Wind Angle", "FH": "Wind Speed"}, inplace=True)
        self.delete_entries_raw(pop_values=faulty, key=self.geo_lev)
        return None

    def delete_entries_raw(self, pop_values: set | list, key: str) -> None:
        if len(pop_values) > 0:
            self.raw_data = self.raw_data[~self.raw_data[key].isin(pop_values)]
        else:
            pass

    def group_data(self) -> None:
        grouped_df = self.raw_data.groupby(by=[self.geo_lev, self.time_lev]).median().copy().reset_index()
        if self.time_lev == 'YYYYMMDD':
            grouped_df[self.time_lev] = to_datetime(grouped_df[self.time_lev].astype(str), format='%Y%m%d')
        else:
            grouped_df[self.time_lev] = to_datetime(grouped_df[self.time_lev])
        grouped_df.set_index(self.time_lev, inplace=True, drop=True)
        self.raw_data = grouped_df
        return None

    def data_saver(self) -> None:

        save(r"../DTO/train_tWind",
             self.train_data.weight_tensor)
        save(r"../DTO/test_tWind",
             self.test_data.weight_tensor)

        mTrainMatrix = DataFrame(self.train_data.weight_matrix)
        mTestMatrix = DataFrame(self.test_data.weight_matrix)
        for i, _ in enumerate(self.train_data.pollution.columns):
            mTrainMatrix.rename(columns={i: self.train_data.pollution.columns[i]}, inplace=True)
            mTestMatrix.rename(columns={i: self.test_data.pollution.columns[i]}, inplace=True)

        db = HLDatabase(bWorkLaptop=self.bWorkLaptop)
        data = {
            'Train-': {
                f'All - {self.geo_lev} - {self.time_lev}': self.train_data.data,
                f'Pollution - {self.geo_lev} - {self.time_lev}': self.train_data.pollution,
                f'Wind Direction - {self.geo_lev} - {self.time_lev}': self.train_data.wind_direction,
                f'Wind Speed - {self.geo_lev} - {self.time_lev}': self.train_data.wind_speed,
                f'Anisotropic - {self.geo_lev} - {self.time_lev}': self.train_data.wSpillovers,
                f'Isotropic - {self.geo_lev} - {self.time_lev}': self.train_data.sSpillovers,
                f'Weight Matrix - {self.geo_lev} - {self.time_lev}': mTrainMatrix
            },
            'Test-': {
                f'All - {self.geo_lev} - {self.time_lev}': self.test_data.data,
                f'Pollution - {self.geo_lev} - {self.time_lev}': self.test_data.pollution,
                f'Wind Direction - {self.geo_lev} - {self.time_lev}': self.test_data.wind_direction,
                f'Wind Speed - {self.geo_lev} - {self.time_lev}': self.test_data.wind_speed,
                f'Anisotropic - {self.geo_lev} - {self.time_lev}': self.test_data.wSpillovers,
                f'Isotropic - {self.geo_lev} - {self.time_lev}': self.test_data.sSpillovers,
                f'Weight Matrix - {self.geo_lev} - {self.time_lev}': mTestMatrix,
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
        self.processed_data.SpatialComponents()
        self.split(faulty=no_sensors)

        if self.save_data:
            self.data_saver()
        return None

    def split(self, faulty: set | list, separation: float = 0.75) -> None:
        cutout = int(len(self.processed_data.pollution) * separation)

        self.train_data = DataBase(df=self.processed_data.data,
                                   faulty=faulty,
                                   geo_level=self.geo_lev,
                                   time_interval=self.time_lev)
        self.train_data.wind_direction = self.processed_data.wind_direction.iloc[:cutout].copy()
        self.train_data.wind_speed = self.processed_data.wind_speed.iloc[:cutout].copy()
        self.train_data.pollution = self.processed_data.pollution.iloc[:cutout, :].copy()
        self.train_data.coordinate_dict = self.processed_data.coordinate_dict
        self.train_data.weight_matrix = self.processed_data.weight_matrix
        self.train_data.angle_matrix = self.processed_data.angle_matrix
        self.train_data.weight_tensor = self.processed_data.weight_tensor[:cutout, :, :]
        self.train_data.sSpillovers = self.processed_data.sSpillovers.iloc[:cutout, :]
        self.train_data.wSpillovers = self.processed_data.wSpillovers.iloc[:cutout, :]
        self.train_data.X = self.processed_data.X[:cutout, :, :]
        self.train_data.Y = self.processed_data.Y[:cutout, :, :]

        self.test_data = DataBase(df=self.processed_data.data,
                                  faulty=faulty,
                                  geo_level=self.geo_lev,
                                  time_interval=self.time_lev)
        self.test_data.wind_direction = self.processed_data.wind_direction.iloc[cutout:].copy()
        self.test_data.wind_speed = self.processed_data.wind_speed.iloc[cutout:].copy()
        self.test_data.pollution = self.processed_data.pollution.iloc[cutout:, :].copy()
        self.test_data.coordinate_dict = self.processed_data.coordinate_dict
        self.test_data.weight_matrix = self.processed_data.weight_matrix
        self.test_data.angle_matrix = self.processed_data.angle_matrix
        self.test_data.weight_tensor = self.processed_data.weight_tensor[cutout:, :, :]
        self.test_data.sSpillovers = self.processed_data.sSpillovers.iloc[cutout:, :]
        self.test_data.wSpillovers = self.processed_data.wSpillovers.iloc[cutout:, :]
        self.test_data.X = self.processed_data.X[cutout:, :, :]
        self.test_data.Y = self.processed_data.Y[cutout:, :, :]

        self.delete_entries()
        return None

    def delete_entries(self) -> None:
        train_names = set(self.train_data.data[self.train_data.geo_lev].unique())
        test_names = set(self.test_data.data[self.test_data.geo_lev].unique())

        misplaced_train = train_names - (test_names & train_names)
        misplaced_test = test_names - (test_names & train_names)

        self.train_data.pollution.drop(columns=misplaced_train, inplace=True)
        self.test_data.pollution.drop(columns=misplaced_test, inplace=True)
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
    while angle > 360:
        angle -= 360
    return angle
