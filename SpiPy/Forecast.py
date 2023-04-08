from __future__ import annotations
from numpy import ndarray
import numpy as np


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> ndarray:
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> ndarray:
    return np.mean(np.abs(y_true - y_pred))


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def SMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred)) / (np.abs(y_pred) + np.abs(y_true)) / 2) * 100


def AIC(residuals: np.ndarray) -> float:
    p = 1
    k = residuals.shape[1]
    n = len(residuals)
    logdet = np.log(np.linalg.det(residuals))
    return logdet + (2 * k * p ** 2) / (n - k * p - 1)
