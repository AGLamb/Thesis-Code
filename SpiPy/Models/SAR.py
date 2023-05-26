from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from numpy import inf, exp, dot, zeros, fill_diagonal, log, prod
from pyswarms.single import GlobalBestPSO
from numpy.linalg import pinv, det, svd
from pandas import DataFrame
from numpy import ndarray


class QMLEOptimizer:
    def __init__(
            self,
            initial_params,
            bounds: list,
            weight_matrix: DataFrame,
            wind_tensor: ndarray,
            exog: ndarray,
            Z: ndarray,
            method: str = 'DE'
    ) -> None:

        self.M = wind_tensor.shape[0]
        self.N = wind_tensor.shape[1]

        assert len(initial_params) == 2 * self.N + 4, "initial_params must be of size 2N + 5"
        assert len(bounds) == 2 * self.N + 4, "bounds must be of size 2N + 5"

        self.initial_params = initial_params
        self.mW_0 = weight_matrix.values
        self.mW_1 = wind_tensor
        self.bounds = bounds
        self.mZ_t = Z
        self.type = method
        self.min_nll = 1000000
        self.mY_t = exog

        self.best_params = None

    def likelihood_function(self, params: list):

        phi = zeros((self.N, self.N))
        fill_diagonal(phi, params[:self.N])

        # alpha = params[-5]
        rho = params[-4]
        zeta = params[-3]
        beta = params[-2]
        gamma = params[-1]

        Sigma = zeros((self.N, self.N))
        fill_diagonal(Sigma, params[self.N:-4])
        det_Sigma = -0.5 * log(det(Sigma)) * self.M

        log_likelihood = 0.0
        for t in range(1, self.M):
            A_t_minus_1 = phi+rho
            A_t_minus_1 += \
                (1-rho)*(1/(1+exp(-zeta*(
                        self.mZ_t[t-1, :, :]-beta-gamma*self.mZ_t[t-1, :, :]
                ))))*self.mW_1[t-1, :, :]

            _, s_A, _ = svd(A_t_minus_1)
            det_A = log(prod(s_A))

            residual_t = self.mY_t[t, :] - dot(A_t_minus_1, self.mY_t[t - 1, :])

            log_likelihood += det_A
            log_likelihood -= 0.5*dot(residual_t.T, dot(pinv(Sigma), residual_t))

        return log_likelihood + det_Sigma

    def neg_log_likelihood(self, params: list):
        return -self.likelihood_function(params=params)

    def fit(self) -> None:
        if self.type == "Normal":
            self.Normal()
        elif self.type == "DE":
            self.DifferentialEvolution()
        elif self.type == "SA":
            self.SimulatedAnnealing()
        elif self.type == "PSO":
            self.ParticleSwarmOptimization()
        elif self.type == "BH":
            self.BasinHopping()
        else:
            raise ValueError(f"Unknown method: {self.type}")
        return None

    def get_best_params(self) -> ndarray:
        return self.best_params

    def Normal(self):

        result = minimize(
            fun=self.neg_log_likelihood,
            x0=self.initial_params,
            bounds=self.bounds,
        )

        if result.fun < self.min_nll:
            self.min_nll = result.fun
            self.best_params = result.x

        return None

    def SimulatedAnnealing(self):

        result = dual_annealing(
            func=self.neg_log_likelihood,
            bounds=self.bounds,
        )

        self.min_nll = result.fun
        self.best_params = result.x
        return

    def ParticleSwarmOptimization(self):
        # Define the bounds in the form expected by pyswarms
        max_bounds = [bound[1] for bound in self.bounds]
        min_bounds = [bound[0] for bound in self.bounds]
        bounds = (min_bounds, max_bounds)

        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=10, dimensions=len(self.bounds), options=options, bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(self.neg_log_likelihood, iters=1000)

        self.min_nll = cost
        self.best_params = pos
        return

    def DifferentialEvolution(self):

        result = differential_evolution(
            func=self.neg_log_likelihood,
            bounds=self.bounds,
            disp=True
        )

        self.min_nll = result.fun
        self.best_params = result.x
        return

    def BasinHopping(self):

        result = basinhopping(
            func=self.neg_log_likelihood,
            x0=self.initial_params,
            minimizer_kwargs={'bounds': self.bounds},
            disp=True,
        )

        self.min_nll = result.fun
        self.best_params = result.x
        return
