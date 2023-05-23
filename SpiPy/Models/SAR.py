from scipy.optimize import minimize
import numpy as np


class MLEOptimizerWithZetaGridSearch:
    def __init__(self, initial_params, bounds, zeta_values, weight_matrix, wind_tensor):
        self.mY_t = None
        self.mZ_t = None
        self.mW_1_t = None
        self.mW_0 = None
        self.initial_params = initial_params
        self.bounds = bounds
        self.zeta_values = zeta_values

        self.min_nll = np.inf
        self.best_params = None

    @staticmethod
    def likelihood_function(params, mW_0, mW_1_t, mZ_t, mY_t):
        phi_1, alpha, rho, zeta, Sigma = params

        # Calculate the log-likelihood at each time step
        log_likelihood = np.zeros_like(mW_1_t)
        for t in range(mW_1_t.shape[0]):
            # Calculate A_t_minus_1 at time t
            W_0_t = mW_0[t]
            W_1_t = mW_1_t[t]
            Z_t = mZ_t[t]

            #  Make phi a diagonal matrix
            A_t_minus_1 = phi_1 + alpha * (
                    rho * W_0_t + (1 - rho) * (1 / (1 + np.exp(-zeta * (Z_t - 1)))) * W_1_t)

            # Calculate the residual at time t
            y_t_minus_1 = mY_t[t - 1]
            residual_t = mY_t[t] - np.dot(A_t_minus_1, y_t_minus_1)

            # Calculate the components of the log-likelihood function
            component1 = -0.5 * np.dot(residual_t.T, np.dot(np.linalg.inv(Sigma), residual_t))

            # Calculate the log-likelihood at time t
            log_likelihood[t] = component1

        return np.sum(log_likelihood)

    def neg_log_likelihood(self, params):
        return -self.likelihood_function(params, self.mW_0, self.mW_1_t, self.mZ_t, self.mY_t)

    def fit(self, mW_0, mW_1_t, mZ_t, mY_t):
        self.mW_0 = mW_0
        self.mW_1_t = mW_1_t
        self.mZ_t = mZ_t
        self.mY_t = mY_t

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

# initial_params = [0.5, 0.5, 0.5]  # Initial guesses for rho, phi, sigma
# bounds = [(0, 1), None, None]  # Bounds for rho, phi, sigma
# zeta_values = np.linspace(0, 1, 10)  # Some possible zeta values
# mle = MLEOptimizerWithZetaGridSearch(initial_params, bounds, zeta_values)
# mle.fit()
# best_params = mle.get_best_params()
# print(f"Best parameters: {best_params}")
