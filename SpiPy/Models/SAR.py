from scipy.optimize import minimize
from numba import jit
import numpy as np


class QMLEOptimizer:
    def __init__(
            self,
            initial_params: np.ndarray,
            bounds: list,
            weight_matrix: np.ndarray,
            wind_tensor: np.ndarray,
            exog: np.ndarray
    ) -> None:

        self.N: int = exog.shape[1]

        assert len(initial_params) == 3 * self.N + 5, "Initial_params must be of size 2N + 5"

        self.initial_params = initial_params
        self.mW_0 = weight_matrix
        self.mW_1 = wind_tensor
        self.bound = bounds
        self.mY_t = exog

        self.min_nll = None
        self.best_params = None

    @staticmethod
    @jit(nopython=True)
    def likelihood_function(params: np.ndarray, mW_1: np.ndarray, mY_t: np.ndarray) -> float:
        M: int = mY_t.shape[0]
        N: int = mY_t.shape[1]

        phi: np.ndarray = np.zeros((N, N))
        np.fill_diagonal(phi, params[:N])

        Sigma: np.ndarray = np.zeros((N, N))
        np.fill_diagonal(Sigma, params[N:2*N])

        mu: np.ndarray = np.zeros(N)
        mu[:] = params[2*N:3*N]

        alpha: float = params[-5]
        rho: float = params[-4]
        zeta: float = params[-3]
        beta: float = params[-2]
        gamma: float = params[-1]

        log_likelihood: float = 0.0
        det_Sigma_term: float = -0.5 * np.log(np.linalg.det(Sigma))

        for t in range(2, M):
            A_t_minus_1: np.ndarray = phi + (alpha + rho * (1 / (1 + np.exp(-zeta * (
                    mW_1[t-1, :, :]@mY_t[t-1, :]-beta-gamma*mW_1[t-2, :, :]@mY_t[t-2, :]
            ))))) * mW_1[t-1, :, :]

            _, s_A, _ = np.linalg.svd(A_t_minus_1)
            det_A: float = np.log(np.prod(s_A))

            residual_t: np.ndarray = mY_t[t, :] - mu - A_t_minus_1 @ mY_t[t-1, :]

            log_likelihood += det_A
            log_likelihood -= 0.5 * np.dot(residual_t.T, np.dot(np.linalg.pinv(Sigma), residual_t))

        return -(log_likelihood + det_Sigma_term * (M - 2))

    def fit(self) -> None:

        result = minimize(
            fun=self.likelihood_function,
            x0=self.initial_params,
            bounds=self.bound,
            args=(self.mW_1, self.mY_t),
            callback=self.callback
        )

        if result.success:
            self.min_nll: float = result.fun
            self.best_params: np.ndarray = result.x
        return None

    def get_best_params(self) -> np.ndarray:
        return self.best_params

    @staticmethod
    def callback(xk):
        print('.', end='')
