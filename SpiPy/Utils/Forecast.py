from __future__ import annotations
from numpy import ndarray, mean, abs, log, sqrt
from numpy.linalg import det


def MSE(y_true: ndarray, y_pred: ndarray) -> ndarray:
    return mean((y_true - y_pred) ** 2)


def RMSE(y_true: ndarray, y_pred: ndarray) -> float:
    return sqrt(mean((y_true - y_pred) ** 2))


def MAE(y_true: ndarray, y_pred: ndarray) -> ndarray:
    return mean(abs(y_true - y_pred))


def MAPE(y_true: ndarray, y_pred: ndarray) -> float:
    return mean(abs((y_true - y_pred) / y_true)) * 100


def SMAPE(y_true: ndarray, y_pred: ndarray) -> float:
    return mean(abs((y_true - y_pred)) / (abs(y_pred) + abs(y_true)) / 2) * 100


def AIC(residuals: ndarray) -> float:
    p = 1
    k = residuals.shape[1]
    n = len(residuals)
    log_det = log(det(residuals))
    return log_det + (2 * k * p ** 2) / (n - k * p - 1)
