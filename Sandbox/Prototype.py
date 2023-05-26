from SpiPy.Models.SAR import *
from DTO.Database import *
from numpy import load
from time import time


import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=False)
    pollution = db_manager.get_table(table_name='Train-Pollution')
    pollution.drop(labels='index', axis=1, inplace=True)
    mSpatial = db_manager.get_table(table_name='Train-Weight Matrix')
    mSpatial.drop(labels='index', axis=1, inplace=True)

    tWind = load(r"../DTO/train_tWind.npy")
    tZ = load(r"../DTO/train_tX.npy")

    N = len(pollution.columns)
    initial_params = [0.19, -0.13, 0.18, -0.11, 0.16, -0.15, 0.14]        # Phi
    initial_params += list(pollution.var().values)                     # Sigma
    # initial_params += [0.21]                                            # Alpha
    initial_params += [0.85]                                           # Rho
    initial_params += [0.33]                                            # Zeta
    initial_params += [2]                                              # Beta
    initial_params += [10]                                            # Gamma

    bounds = [(-1, 1)] * N + [(1, 1000)] * N + [(0, 1)] + [(0, 1)] + [(-1000, 1000)] * 2
    # [(-1, 1)]  # Alpha bound
    tZ_1 = tZ[9500:10000, :, :]
    pollution_1 = pollution.values[9500:10000, :]
    tWind_1 = tWind[9500:10000, :, :]

    optimizer = QMLEOptimizer(
        initial_params=initial_params,
        weight_matrix=mSpatial,
        wind_tensor=tWind_1,
        exog=pollution_1,
        method='Normal',
        bounds=bounds,
        Z=tZ_1
    )

    optimizer.fit()
    best_params = optimizer.get_best_params()

    print("Best Parameters:")
    # print("Alpha:", best_params[-5])
    print("Rho:", best_params[-4])
    print("Zeta:", best_params[-3])
    print("Beta:", best_params[-2])
    print("Gamma:", best_params[-1])
    print("Phi:", best_params[:N])
    print("Sigma:", best_params[N:-4])
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        start_time = time()
        warnings.simplefilter("ignore")
        main()
        end_time = time()
        print("Time taken: ", end_time - start_time, "seconds")
