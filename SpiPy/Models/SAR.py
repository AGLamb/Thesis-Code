from numpy import inf, exp, dot, zeros, fill_diagonal, log
from numpy.linalg import inv, det
from scipy.optimize import minimize
from pandas import DataFrame
from numpy import ndarray
from tqdm import tqdm


class QMLEOptimizerWithZetaGridSearch:
    def __init__(
            self,
            initial_params,
            bounds: list,  # Expected size is 3N + 1, without zeta
            zeta_values: ndarray,
            weight_matrix: DataFrame,
            wind_tensor: ndarray,
            exog: ndarray,
            Z: ndarray
    ) -> None:

        N = wind_tensor.shape[1]  # Assuming N is the number of columns in weight_matrix
        assert len(initial_params) == 2 * N + 2, "initial_params must be of size 2N + 1"
        assert len(bounds) == 2 * N + 2, "bounds must be of size 2N + 1"

        self.initial_params = initial_params
        self.mW_0 = weight_matrix.values
        self.zeta_values = zeta_values
        self.mW_1 = wind_tensor
        self.mY_t = exog
        self.bounds = bounds
        self.min_nll = inf
        self.mZ_t = Z * 100

        self.best_params = None

    def likelihood_function(self, params: list):
        M = self.mW_1.shape[0]
        N = self.mW_0.shape[1]

        phi = zeros((N, N))
        fill_diagonal(phi, params[:N])

        alpha = params[N]
        rho = params[N + 1]
        zeta = params[-1]

        Sigma = zeros((N, N))
        fill_diagonal(Sigma, params[N + 2:-1])
        det_Sigma = -0.5 * log(det(Sigma)) * M

        log_likelihood = 0.0
        for t in range(1, M):
            A_t_minus_1 = phi+alpha*rho*self.mW_0
            A_t_minus_1 += \
                alpha*(1-rho)*(1/(1+exp(-zeta*(self.mZ_t[t-1, :, :]-1))))*self.mW_1[t, :, :]

            residual_t = self.mY_t[t, :] - dot(A_t_minus_1, self.mY_t[t - 1, :])

            log_likelihood += \
                log(det(A_t_minus_1))-0.5*dot(residual_t.T, dot(inv(Sigma), residual_t))

        return log_likelihood + det_Sigma

    def neg_log_likelihood(self, params: list):
        return -self.likelihood_function(params=params)

    def fit_with_zeta(self, zeta):
        params_with_zeta = self.initial_params + [zeta]
        bounds_with_zeta = self.bounds + [(0, 1)]

        result = minimize(
            fun=self.neg_log_likelihood,
            x0=params_with_zeta,
            bounds=bounds_with_zeta,
        )

        return result.fun, result.x

    def fit(self):
        for zeta in tqdm(self.zeta_values, desc='Optimizing: '):
            initial_params_with_zeta = self.initial_params + [zeta]
            bounds_with_zeta = self.bounds + [(0, 1)]

            result = minimize(
                fun=self.neg_log_likelihood,
                x0=initial_params_with_zeta,
                bounds=bounds_with_zeta,
            )

            if result.fun < self.min_nll:
                self.min_nll = result.fun
                self.best_params = result.x

    def get_best_params(self):
        return self.best_params
