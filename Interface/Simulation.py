from SpiPy.Models.SAR import *
from scipy.stats import multivariate_normal, norm
from numpy.random import seed, uniform, rand
from numpy import exp, zeros, eye, pi, cos, var, mean
from scipy.spatial.distance import cdist
from numpy.linalg import eigvals


seed(123)


N = 10
T = 500
rho = 0.5
alpha = 0.05
beta = 0.07
gamma = 0.25
zeta = 0.7

locations = rand(N, 2)
dist_matrix = cdist(locations, locations, 'euclidean')
W_dist = 1 / (dist_matrix + eye(N))
W_dist = W_dist / W_dist.sum(axis=1, keepdims=True)
errors = multivariate_normal.rvs(cov=eye(N), size=T)
wind_speed = norm.rvs(loc=10, scale=2, size=T)
wind_direction = uniform(0, 2*pi, size=T)
angles = uniform(0, 2*pi, size=(N, N))
Y = zeros((T, N))
Y_true = zeros((T, N))

tWind = zeros((T, N, N))

for t in range(1, T):
    wind_angle = (wind_direction[t] - angles + pi) % (2*pi) - pi
    wind_speed_decomp = wind_speed[t] * cos(wind_angle)
    W_dynamic = W_dist * wind_speed_decomp
    W_dynamic = W_dynamic / max(eigvals(W_dynamic))
    tWind[t, :, :] = W_dynamic
    transition_function = alpha + rho / (1 + exp(-zeta * (W_dynamic - beta - gamma * W_dynamic @ W_dynamic.T)))
    Y_true[t] = transition_function @ (W_dynamic @ Y[t-1].T)
    Y[t, :] = transition_function @ (W_dynamic @ Y[t-1].T) + errors[t]

vars = zeros(N)
means = zeros(N)
for i in range(N):
    vars[i] = var(Y[:, i])
    means[i] = mean(Y[:, i])

N = Y.shape[1]
initial_params = zeros(3*N+5)
initial_params[:N] = [0.9] * N                                           # Phi
initial_params[N:2*N] = list(vars)                                     # Sigma
initial_params[2*N:3*N] = list(means)                                  # Mu
initial_params[-5] = 0.25                                                # Alpha
initial_params[-4] = 0.8                                                 # Rho
initial_params[-3] = 0.2                                                 # Zeta
initial_params[-2] = 2.0                                                 # Beta
initial_params[-1] = 5.0                                                 # Gamma

bounds = [(None, None)] * N                                              # Phi
bounds += [(1, 1000)] * N                                                # Sigma
bounds += [(0, 1000)] * N                                                # Mu
bounds += [(None, None)]                                                 # Alpha
bounds += [(None, None)]                                                 # Rho
bounds += [(0, 1)]                                                       # Zeta
bounds += [(None, None)]                                                 # Beta
bounds += [(None, None)]                                                 # Gamma

optimizer = QMLEOptimizer(
    initial_params=initial_params,
    wind_tensor=tWind,
    exog=Y,
    bounds=bounds,
    ratio=tWind
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
