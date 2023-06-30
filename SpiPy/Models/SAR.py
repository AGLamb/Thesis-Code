from scipy.optimize import minimize
from numba import njit, prange
import numpy as np
import math


class QMLEOptimizer:
    def __init__(
            self,
            initial_params: np.ndarray,
            bounds: list,
            wind_tensor: np.ndarray,
            weight_matrix: np.ndarray,
            exog: np.ndarray
    ) -> None:

        self.N: int = exog.shape[1]

        assert len(initial_params) == 3 * self.N + 5, "Initial_params must be of size 2N + 5"

        self.initial_params = initial_params
        self.mW_0 = weight_matrix
        self.mW_1 = wind_tensor
        self.bound = bounds
        self.iteration = 0
        self.mY_t = exog

        self.min_nll = None
        self.params = None

    @staticmethod
    @njit
    def likelihood_function(
            params: np.ndarray,
            mW_0: np.ndarray,
            mW_1: np.ndarray,
            mY_t: np.ndarray
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

        det_Sigma_term: float = 0.5 * np.log(np.linalg.det(Sigma))
        pinv_Sigma = np.linalg.pinv(Sigma)
        log_likelihood: np.ndarray = np.zeros(M-1)

        for t in prange(1, M):
            A_t_minus_1: np.ndarray = phi + alpha * mW_0 + 1 / (1 + np.exp(
                -zeta * (mW_1[t-1, :, :] - beta - gamma * mW_1[t-1, :, :] ** 2))) * mW_1[t-1, :, :]

            residual_t: np.ndarray = mY_t[t, :] - mu - A_t_minus_1 @ mY_t[t-1, :]
            log_likelihood[t-1] = -0.5 * np.dot(residual_t.T, np.dot(pinv_Sigma, residual_t))

        return -(np.sum(log_likelihood) - det_Sigma_term * (M - 1) - 0.5 * N * (M - 1) * np.log(2 * np.pi))

    def fit(self) -> None:
        print("Optimisation started...")
        result = minimize(
            fun=self.likelihood_function,
            x0=self.initial_params,
            bounds=self.bound,
            args=(self.mW_0, self.mW_1, self.mY_t),
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

    def calculate_aic(self):
        K = len(self.initial_params)
        return 2 * K + 2 * self.min_nll

    def calculate_bic(self):
        K = len(self.initial_params)
        n = len(self.mY_t)
        return K * np.log(n) + 2 * self.min_nll


class TQMLEOptimizer:
    def __init__(
            self,
            initial_params: np.ndarray,
            bounds: list,
            weight_matrix: np.ndarray,
            wind_tensor: np.ndarray,
            exog: np.ndarray
    ) -> None:

        self.N: int = exog.shape[1]
        self.initial_params = initial_params
        self.mW_0 = weight_matrix
        self.mW_1 = wind_tensor
        self.bound = bounds
        self.iteration = 0
        self.mY_t = exog

        self.min_nll = None
        self.params = np.zeros(self.initial_params.shape[0])

    @staticmethod
    @njit
    def likelihood_function(
            params: np.ndarray,
            mW_0: np.ndarray,
            mW_1: np.ndarray,
            mY_t: np.ndarray
    ) -> float:

        M: int = mY_t.shape[0]
        N: int = mY_t.shape[1]

        phi: np.ndarray = np.zeros((N, N))
        np.fill_diagonal(phi, params[:N])

        Sigma: np.ndarray = np.zeros((N, N))
        np.fill_diagonal(Sigma, params[N:2*N])

        mu: np.ndarray = np.zeros(N)
        mu[:] = params[2*N:3*N]

        alpha: float = params[-6]
        rho: float = params[-5]
        zeta: float = params[-4]
        beta: float = params[-3]
        gamma: float = params[-2]
        v: float = params[-1]

        det_Sigma_term: float = 0.5 * np.log(np.linalg.det(Sigma))
        pinv_Sigma = np.linalg.pinv(Sigma)
        log_likelihood: np.ndarray = np.zeros(M - 1)

        constant_term: float = np.log(math.gamma((v + N) / 2)) - np.log(math.gamma(v / 2)) - (N / 2) * np.log(v - 2)

        for t in prange(1, M):
            A_t_minus_1: np.ndarray = phi + alpha * mW_0 + 1 / (1 + np.exp(
                -zeta * (mW_1[t-1, :, :] - beta - gamma * mW_1[t-1, :, :] ** 2))) * mW_1[t - 1, :, :]

            residual_t: np.ndarray = mY_t[t, :] - mu - A_t_minus_1 @ mY_t[t - 1, :]
            log_likelihood[t - 1] = constant_term - ((v + N) / 2) * np.log(
                1 + np.dot(residual_t.T, np.dot(pinv_Sigma, residual_t)) / (v - 2))

        return -(np.sum(log_likelihood) - det_Sigma_term * (M - 1))

    def fit(self) -> None:
        print("Optimisation started...")
        result = minimize(
            fun=self.likelihood_function,
            x0=self.initial_params,
            bounds=self.bound,
            args=(self.mW_0, self.mW_1, self.mY_t),
            callback=self.callback,
            method='L-BFGS-B'
        )

        if result.success:
            self.min_nll: float = result.fun
            self.params: np.ndarray = result.x
        return None

    def get_best_params(self) -> np.ndarray:
        return self.params

    def callback(self, xk):
        self.iteration += 1
        print(f"Iteration: {self.iteration}")

    def calculate_aic(self):
        K = len(self.initial_params)
        return 2 * K + 2 * self.min_nll

    def calculate_bic(self):
        K = len(self.initial_params)
        n = len(self.mY_t)
        return K * np.log(n) + 2 * self.min_nll
