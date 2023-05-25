from numpy import load, linspace
from SpiPy.Models.SAR import *
from DTO.Database import *
from time import time
import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=True)
    pollution = db_manager.get_table(table_name='Train-Pollution')
    pollution.set_index('Date', inplace=True, drop=True)
    mSpatial = db_manager.get_table(table_name='Train-Weight Matrix')
    mSpatial = mSpatial.drop(labels='index', axis=1)

    tWind = load(r"../DTO/train_tWind.npy")
    tZ = load(r"../DTO/train_tZ.npy")

    N = len(pollution.columns)
    initial_params = [0.5]*N + [0.2] + [0.5] + list(pollution.var().values)
    bounds = [(-1, 1)]*N + [(-1, 1)] + [(0, 1)] + [(1, 1000)]*N
    zeta_values = linspace(0, 1, 11)

    tZ_1 = tZ[8000:9000, :, :]
    pollution_1 = pollution.values[8000:9000, :]
    tWind_1 = tWind[8000:9000, :, :]

    optimizer = QMLEOptimizerWithZetaGridSearch(
        initial_params=initial_params,
        bounds=bounds,
        zeta_values=zeta_values,
        weight_matrix=mSpatial,
        wind_tensor=tWind_1,
        exog=pollution_1,
        Z=tZ_1
    )

    optimizer.fit()
    best_params = optimizer.get_best_params()

    print("Best Parameters:")
    print("alpha:", best_params[N])
    print("rho:", best_params[N+1])
    print("zeta:", best_params[-1])
    print("phi:", best_params[:N])
    print("Sigma:", best_params[N+2:-1])
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        start_time = time()
        warnings.simplefilter("ignore")
        main()
        end_time = time()
        print("Time taken: ", end_time - start_time, "seconds")
