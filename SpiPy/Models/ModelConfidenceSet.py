from numpy.random import rand
from numpy import ix_
import pandas as pd
import numpy as np


def bootstrap_sample(data, B, w):
    t = len(data)
    p = 1 / w
    indices = np.zeros((t, B), dtype=int)
    indices[0, :] = np.ceil(t * rand(1, B))
    select = np.asfortranarray(rand(B, t).T < p)
    values = np.ceil(rand(1, np.sum(np.sum(select))) * t).astype(int)
    indices_flat = indices.ravel(order="F")
    indices_flat[select.ravel(order="F")] = values.ravel()
    indices = indices_flat.reshape([B, t]).T
    for i in range(1, t):
        indices[i, ~select[i, :]] = indices[i - 1, ~select[i, :]] + 1
    indices[indices > t] = indices[indices > t] - t
    indices -= 1
    return data[indices]


def compute_dij(losses, bs_data):
    t, M0 = losses.shape
    B = bs_data.shape[1]
    d_ij_bar = np.zeros((M0, M0))
    for j in range(M0):
        d_ij_bar[j, :] = np.mean(losses - losses[:, [j]], axis=0)

    d_ij_bar_star = np.zeros((B, M0, M0))
    for b in range(B):
        mean_work_data = np.mean(losses[bs_data[:, b], :], axis=0)
        for j in range(M0):
            d_ij_bar_star[b, j, :] = mean_work_data - mean_work_data[j]

    var_d_ij_bar = np.mean((d_ij_bar_star - np.expand_dims(d_ij_bar, 0)) ** 2, axis=0)
    var_d_ij_bar += np.eye(M0)

    return d_ij_bar, d_ij_bar_star, var_d_ij_bar


def calculate_PvalR(z, included, z_data_0):
    emp_dist_TR = np.max(np.max(np.abs(z), 2), 1)
    z_data = z_data_0[ix_(included - 1, included - 1)]
    TR = np.max(z_data)
    p_val = np.mean(emp_dist_TR > TR)
    return p_val


def calculate_PvalSQ(z, included, z_data_0):
    emp_dist_TSQ = np.sum(z ** 2, axis=1).sum(axis=1) / 2
    z_data = z_data_0[ix_(included - 1, included - 1)]
    TSQ = np.sum(z_data ** 2) / 2
    p_val = np.mean(emp_dist_TSQ > TSQ)
    return p_val


def iterate(d_ij_bar, d_ij_bar_star, var_d_ij_bar, alpha, algorithm="R"):
    B, M0, _ = d_ij_bar_star.shape
    z0 = (d_ij_bar_star - np.expand_dims(d_ij_bar, 0)) / np.sqrt(
        np.expand_dims(var_d_ij_bar, 0)
    )
    z_data_0 = d_ij_bar / np.sqrt(var_d_ij_bar)

    excludedR = np.zeros([M0, 1], dtype=int)
    p_values_R = np.ones([M0, 1])

    for i in range(M0 - 1):
        included = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
        m = len(included)
        z = z0[ix_(range(B), included - 1, included - 1)]

        if algorithm == "R":
            p_values_R[i] = calculate_PvalR(z, included, z_data_0)
        elif algorithm == "SQ":
            p_values_R[i] = calculate_PvalSQ(z, included, z_data_0)

        scale = m / (m - 1)
        d_i_bar = np.mean(d_ij_bar[ix_(included - 1, included - 1)], 0) * scale
        d_ib_star = np.mean(d_ij_bar_star[ix_(range(B), included - 1, included - 1)], 1) * (
                m / (m - 1)
        )
        vardi = np.mean((d_ib_star - d_i_bar) ** 2, axis=0)
        t = d_i_bar / np.sqrt(vardi)
        model_to_remove = np.argmax(t)
        excludedR[i] = included[model_to_remove]

    max_p_val = p_values_R[0]
    for i in range(1, M0):
        if p_values_R[i] < max_p_val:
            p_values_R[i] = max_p_val
        else:
            max_p_val = p_values_R[i]

    excludedR[-1] = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
    pl = np.argmax(p_values_R > alpha)
    includedR = excludedR[pl:]
    excludedR = excludedR[:pl]
    return includedR - 1, excludedR - 1, p_values_R


def MCS(losses, alpha, B, w, algorithm):
    t, M0 = losses.shape
    bs_data = bootstrap_sample(np.arange(t), B, w)
    d_ij_bar, d_ij_bar_star, var_d_ij_bar = compute_dij(losses, bs_data)
    includedR, excludedR, p_values_R = iterate(
        d_ij_bar, d_ij_bar_star, var_d_ij_bar, alpha, algorithm=algorithm
    )
    return includedR, excludedR, p_values_R


class ModelConfidenceSet(object):
    def __init__(self, data, alpha, B, w, algorithm="SQ", names=None):

        self.pvalues = None
        self.included = None
        self.excluded = None

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.names = data.columns.values if names is None else names
        elif isinstance(data, np.ndarray):
            self.data = data
            self.names = np.arange(data.shape[1]) if names is None else names

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be larger than zero and less than 1, found {alpha}"
            )
        if not isinstance(B, int):
            try:
                B = int(B)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap size B must be a integer, fail to convert", identifier
                )
        if B < 1:
            raise ValueError(f"Bootstrap size B must be larger than 1, found {B}")
        if not isinstance(w, int):
            try:
                w = int(w)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap block size w must be a integer, fail to convert",
                    identifier,
                )
        if w < 1:
            raise ValueError(f"Bootstrap block size w must be larger than 1, found {w}")

        if algorithm not in ["R", "SQ"]:
            raise TypeError(f"Only R and SQ algorithm supported, found {algorithm}")

        self.alpha = alpha
        self.B = B
        self.w = w
        self.algorithm = algorithm

    def run(self):
        included, excluded, p_values = MCS(
            self.data, self.alpha, self.B, self.w, self.algorithm
        )
        self.included = self.names[included].ravel().tolist()
        self.excluded = self.names[excluded].ravel().tolist()
        self.pvalues = pd.Series(p_values.ravel(), index=self.excluded + self.included)

        return self
