from scipy.optimize import minimize
from numba import njit, prange
import numpy as np


class QMLEOptimizer:
    def __init__(
            self,
            initial_params: np.ndarray,
            bounds: list,
            wind_tensor: np.ndarray,
            exog: np.ndarray,
            ratio: np.ndarray
    ) -> None:

        self.N: int = exog.shape[1]

        assert len(initial_params) == 3 * self.N + 5, "Initial_params must be of size 2N + 5"

        self.initial_params = initial_params
        self.mW_1 = wind_tensor
        self.bound = bounds
        self.iteration = 0
        self.mY_t = exog
        self.X = ratio

        self.min_nll = None
        self.params = None

    @staticmethod
    @njit
    def likelihood_function(
            params: np.ndarray,
            mW_1: np.ndarray,
            mY_t: np.ndarray,
            X: np.ndarray
    ) -> float:

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

        det_Sigma_term: float = -0.5 * np.log(np.linalg.det(Sigma))
        pinv_Sigma = np.linalg.pinv(Sigma)
        log_likelihood: np.ndarray = np.zeros(M-1)

        for t in prange(1, M):
            A_t_minus_1: np.ndarray = phi + (alpha + rho / (1 + np.exp(
                -zeta * (X[t, :, :] - beta - gamma * X[t, :, :])))) @ mW_1[t-1, :, :]

            _, s_A, _ = np.linalg.svd(A_t_minus_1)
            det_A: float = np.log(np.prod(s_A))

            residual_t: np.ndarray = mY_t[t, :] - mu - A_t_minus_1 @ mY_t[t-1, :]
            log_likelihood[t-1] = det_A - 0.5 * np.dot(residual_t.T, np.dot(pinv_Sigma, residual_t))

        return -(np.sum(log_likelihood) + det_Sigma_term * (M - 1))

    def fit(self) -> None:
        print("Optimisation started...")
        result = minimize(
            fun=self.likelihood_function,
            x0=self.initial_params,
            bounds=self.bound,
            args=(self.mW_1, self.mY_t, self.X),
            callback=self.callback,
            method='L-BFGS-B'
        )

        print(result.success)
        if result.success:
            self.min_nll: float = result.fun
            self.params: np.ndarray = result.x
        return None

    def get_best_params(self) -> np.ndarray:
        return self.params

    def callback(self, xk):
        self.iteration += 1
        print(f"Iteration: {self.iteration}")
