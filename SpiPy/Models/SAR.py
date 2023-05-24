from numpy import inf, zeros_like, exp, dot, linalg, sum
from scipy.optimize import minimize
from pandas import DataFrame
from numpy import ndarray


class MLEOptimizerWithZetaGridSearch:
    def __init__(
            self,
            initial_params,
            bounds,
            zeta_values,
            weight_matrix,
            wind_tensor,
            exog: DataFrame,
            Z: ndarray
    ):

        self.initial_params = initial_params
        self.zeta_values = zeta_values
        self.mW_0 = weight_matrix
        self.mW_1 = wind_tensor
        self.bounds = bounds
        self.min_nll = inf
        self.mY_t = exog

        self.best_params = None
        self.mZ_t = Z

    def likelihood_function(self, params: list):

        phi, alpha, rho, zeta, Sigma = params

        # Calculate the log-likelihood at each time step
        log_likelihood = zeros_like(self.mW_1)
        for t in range(self.mW_1.shape[0]):
            # Calculate A_t_minus_1 at time t
            W_1_t = self.mW_1[t, :, :]
            Z_t = self.mZ_t[t, :, :]

            #  Make phi a diagonal matrix
            A_t_minus_1 = phi + alpha * (
                    rho * self.mW_0 + (1 - rho) * (1 / (1 + exp(-zeta * (Z_t - 1)))) * W_1_t)

            # Calculate the residual at time t
            y_t_minus_1 = self.mY_t[t - 1]
            residual_t = self.mY_t[t] - dot(A_t_minus_1, y_t_minus_1)

            # Calculate the components of the log-likelihood function
            component1 = -0.5 * dot(residual_t.T, dot(linalg.inv(Sigma), residual_t))

            # Calculate the log-likelihood at time t
            log_likelihood[t] = component1

        return sum(log_likelihood)

    def neg_log_likelihood(self, params):
        return -self.likelihood_function(params=params)

    def fit(self):

        for zeta in self.zeta_values:
            # Set zeta as the last parameter
            initial_params_with_zeta = self.initial_params + [zeta]
            bounds_with_zeta = self.bounds + [(0, 1)]

            result = minimize(
                self.neg_log_likelihood,
                initial_params_with_zeta,
                bounds=bounds_with_zeta
            )

            if result.fun < self.min_nll:
                self.min_nll = result.fun
                self.best_params = result.x

    def get_best_params(self):
        return self.best_params
