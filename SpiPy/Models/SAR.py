from scipy.optimize import minimize
import numpy as np


class MLEOptimizerWithZetaGridSearch:
    def __init__(self, initial_params, bounds, zeta_values):
        self.initial_params = initial_params
        self.bounds = bounds
        self.zeta_values = zeta_values

        self.min_nll = np.inf
        self.best_params = None

    def likelihood_function(self, params):
        # Calculate likelihood of observing the data given these parameters
        likelihood = ...
        return likelihood

    def neg_log_likelihood(self, params):
        return -np.log(self.likelihood_function(params))

    def fit(self):
        for zeta in self.zeta_values:
            # Set zeta as the last parameter
            initial_params_with_zeta = self.initial_params + [zeta]
            bounds_with_zeta = self.bounds + [(0, 1)]

            result = minimize(self.neg_log_likelihood, initial_params_with_zeta, bounds=bounds_with_zeta)

            if result.fun < self.min_nll:
                self.min_nll = result.fun
                self.best_params = result.x

    def get_best_params(self):
        return self.best_params


def simpleML(
        rho: float,
        phi: np.ndarray,
        sigma: np.ndarray
) -> float:
    output = ...
    return output


def smoothThresholdML(
        rho: float,
        alpha: float,
        c: float,
        zeta: float,
        phi: np.ndarray,
        sigma: np.ndarray
) -> float:

    output = ...
    return output


# initial_params = [0.5, 0.5, 0.5]  # Initial guesses for rho, phi, sigma
# bounds = [(0, 1), None, None]  # Bounds for rho, phi, sigma
# zeta_values = np.linspace(0, 1, 10)  # Some possible zeta values
# mle = MLEOptimizerWithZetaGridSearch(initial_params, bounds, zeta_values)
# mle.fit()
# best_params = mle.get_best_params()
# print(f"Best parameters: {best_params}")
