from numpy import load, log, zeros
from numpy.linalg import eigvals
from SpiPy.Models.SAR import *
from DTO.Database import *
from time import time


import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=False)
    pollution = db_manager.get_table(table_name='Train-Pollution')
    pollution.drop(labels='Date', axis=1, inplace=True)
    mSpatial = db_manager.get_table(table_name='Train-Weight Matrix')
    mSpatial.drop(labels='index', axis=1, inplace=True)

    pollution = log(pollution)

    tWind = load(r"../DTO/train_tWind.npy")

    # Initial guess
    N = len(pollution.columns)
    initial_params = zeros(3*N+5)
    initial_params[:N] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]                 # Phi
    initial_params[N:2*N] = list(pollution.var().values)                     # Sigma
    initial_params[2*N:3*N] = list(pollution.mean().values)                  # Mu
    initial_params[-5] = 7.87                                                # Alpha
    initial_params[-4] = -16.5                                               # Rho
    initial_params[-3] = 0.05                                                # Zeta
    initial_params[-2] = 5.09                                                # Beta
    initial_params[-1] = 0.15                                                # Gamma

    bounds = [(-20, 20)] * N             # Phi
    bounds += [(1, 1000)] * N            # Sigma
    bounds += [(0, 1000)] * N            # Mu
    bounds += [(-100, 100)]              # Alpha
    bounds += [(-100, 100)]              # Rho
    bounds += [(0, 1)]                   # Zeta
    bounds += [(-100, 100)] * 2          # Beta and Gamma

    optimizer = QMLEOptimizer(
        initial_params=initial_params,
        weight_matrix=(mSpatial.values/max(eigvals(mSpatial.values))),
        wind_tensor=tWind,
        exog=pollution.values,
        bounds=bounds,
    )

    optimizer.fit()
    best_params = optimizer.get_best_params()
    print("Best Parameters:")
    print("Alpha:",      best_params[-5])
    print("Rho:",        best_params[-4])
    print("Zeta:",       best_params[-3])
    print("Beta:",       best_params[-2])
    print("Gamma:",      best_params[-1])
    print("Phi:",        best_params[:N])
    print("Sigma:",   best_params[N:2*N])
    print("Mu:",    best_params[2*N:3*N])
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        start_time = time()
        warnings.simplefilter("ignore")
        main()
        end_time = time()
        print("Time taken: ", end_time - start_time, "seconds")
