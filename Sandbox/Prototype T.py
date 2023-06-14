from SpiPy.Models.SAR import *
from numpy import load, zeros
from DTO.Database import *
from time import time


import warnings


def main() -> None:
    db_manager = HLDatabase(bWorkLaptop=False)
    pollution = db_manager.get_table(table_name='Train-Pollution - tag - timestamp')
    pollution.drop(labels='timestamp', axis=1, inplace=True)

    tWind = load(r"../DTO/train_tWind.npy")
    tX = load(r"../DTO/train_tX.npy", allow_pickle=True)

    # Initial guess
    N = pollution.shape[1]
    initial_params = zeros(3*N+6)
    initial_params[:N] = [1.5] * N                                           # Phi
    initial_params[N:2*N] = list(pollution.var().values)                     # Sigma
    initial_params[2*N:3*N] = list(pollution.mean().values)                  # Mu
    initial_params[-6] = 0.5                                                 # Alpha
    initial_params[-5] = 0.5                                                 # Rho
    initial_params[-4] = 0.7                                                 # Zeta
    initial_params[-3] = 10.0                                                 # Beta
    initial_params[-2] = 5.0                                                 # Gamma
    initial_params[-1] = 15                                                  # DF

    bounds = [(-5, 5)] * N                                                # Phi
    bounds += [(1, 1000)] * N                                             # Sigma
    bounds += [(0, 500)] * N                                              # Mu
    bounds += [(-2, 2)]                                                   # Alpha
    bounds += [(-2, 2)]                                                   # Rho
    bounds += [(0, 1)]                                                    # Zeta
    bounds += [(-50, 50)]                                                 # Beta
    bounds += [(-50, 50)]                                                 # Gamma
    bounds += [(4, 30)]                                                   # DF

    optimizer = TQMLEOptimizer(
        initial_params=initial_params,
        wind_tensor=tWind,
        exog=pollution.values,
        bounds=bounds,
        ratio=tX
    )

    optimizer.fit()
    best_params = optimizer.get_best_params()
    print("Best Parameters:")
    print("Alpha:",      best_params[-6])
    print("Rho:",        best_params[-5])
    print("Zeta:",       best_params[-4])
    print("Beta:",       best_params[-3])
    print("Gamma:",      best_params[-2])
    print("DF:",         best_params[-1])
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
