from numpy.linalg import eigvals
from SpiPy.Models.SAR import *
from numpy import load, zeros
from DTO.Database import *
from time import time


import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=True)
    pollution = db_manager.get_table(table_name='Train-Pollution - tag - timestamp')
    pollution.drop(labels='timestamp', axis=1, inplace=True)

    mWeight = db_manager.get_table(table_name='Train-Weight Matrix - tag - timestamp')
    mWeight.drop(columns='index', axis=1, inplace=True)
    mWeight /= max(eigvals(mWeight.values))

    tWind = load(r"../DTO/train_tWind.npy")

    # Initial guess
    N = pollution.shape[1]
    initial_params = zeros(3*N+5)
    initial_params[:N] = [0.5] * N                                           # Phi
    initial_params[N:2*N] = list(pollution.var().values)                     # Sigma
    initial_params[2*N:3*N] = list(pollution.mean().values)                  # Mu
    initial_params[-5] = 0.2                                                 # Alpha
    initial_params[-4] = 0.5                                                 # Rho
    initial_params[-3] = 0.7                                                 # Zeta
    initial_params[-2] = 1.0                                                 # Beta
    initial_params[-1] = 0.5                                                 # Gamma

    bounds = [(None, None)] * N                                              # Phi
    bounds += [(1, None)] * N                                                # Sigma
    bounds += [(0, None)] * N                                                # Mu
    bounds += [(None, None)]                                                 # Alpha
    bounds += [(None, None)]                                                 # Rho
    bounds += [(0, 1)]                                                       # Zeta
    bounds += [(None, None)]                                                 # Beta
    bounds += [(None, None)]                                                 # Gamma

    optimizer = QMLEOptimizer(
        initial_params=initial_params,
        weight_matrix=mWeight.values,
        wind_tensor=tWind,
        exog=pollution.values,
        bounds=bounds
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

    print(f'AIC: {optimizer.calculate_aic()}')
    print(f'BIC: {optimizer.calculate_bic()}')
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        start_time = time()
        warnings.simplefilter("ignore")
        main()
        end_time = time()
        print("Time taken: ", end_time - start_time, "seconds")
