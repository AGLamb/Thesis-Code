from SpiPy.Models.SAR import *
from DTO.Database import *
from numpy import load
from time import time


# import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=True)
    pollution = db_manager.get_table(table_name='Train-Pollution')
    pollution.set_index('Date', inplace=True, drop=True)
    mSpatial = db_manager.get_table(table_name='Train-Weight Matrix')
    mSpatial = mSpatial.drop(labels='index', axis=1)

    tWind = load(r"../DTO/train_tWind.npy")
    tZ = load(r"../DTO/train_tX.npy")

    N = len(pollution.columns)
    initial_params = [0.19, 0.23, 0.34, 0.21, 0.36, 0.41, 0.32]  # Phi
    initial_params += [0.2]  # Alpha
    initial_params += [0.85]  # Rho
    initial_params += list(pollution.var().values)  # Sigma
    initial_params += [20]  # Zeta
    initial_params += [10.0]  # Beta
    initial_params += [200]  # Gamma

    bounds = [(-0.9, 0.9)] * N + [(-1, 1)] + [(0, 1)] + [(1, 1000)] * N + [(0, 1)] + [(-1000, 1000)] * 2

    tZ_1 = tZ[8000:10000, :, :]
    pollution_1 = pollution.values[8000:10000, :]
    tWind_1 = tWind[8000:10000, :, :]

    optimizer = QMLEOptimizer(
        initial_params=initial_params,
        bounds=bounds,
        weight_matrix=mSpatial,
        wind_tensor=tWind_1,
        exog=pollution_1,
        Z=tZ_1,
        method='Normal'
    )

    optimizer.fit()
    best_params = optimizer.get_best_params()

    print("Best Parameters:")
    print("Phi:", best_params[:N])
    print("Alpha:", best_params[N])
    print("Rho:", best_params[N + 1])
    print("Zeta:", best_params[-3])
    print("Beta:", best_params[-2])
    print("Gamma:", best_params[-1])
    print("Sigma:", best_params[N + 2:-3])
    return None


if __name__ == "__main__":
    # with warnings.catch_warnings():
    start_time = time()
    # warnings.simplefilter("ignore")
    main()
    end_time = time()
    print("Time taken: ", end_time - start_time, "seconds")
