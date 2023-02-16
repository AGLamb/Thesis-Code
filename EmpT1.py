from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import random
import math


def main(N: int, fBeta: float, iMu: int, iSigma: int, iBurnIn: int):
    vY = simulate_ar_process(N, fBeta, iMu, iSigma, iBurnIn)
    beta1 = beta1_estimation(vY)
    print(beta1)
    print(AutoReg(vY, lags=1, trend='n').fit().params[0])
    return


def simulate_ar_process(N: int, fBeta: float, iMu: int, iSigma: int, iBurnIn: int):
    vY = np.zeros(N)
    vY[0] = random.normalvariate(mu=iMu, sigma=math.sqrt(iSigma))
    for i in range(1, N):
        vY[i] = vY[i-1] * fBeta + random.normalvariate(mu=iMu, sigma=math.sqrt(iSigma))
    vY = vY[iBurnIn:]
    plt.plot(vY)
    plt.show()
    return vY


def beta1_estimation(vInput):
    vY_lag = np.roll(vInput, 1, axis=0)
    vY_lag[0] = 0
    return np.matmul(vInput.T, vY_lag) * 1/np.matmul(vY_lag.T, vY_lag)


if __name__ == "__main__":
    main(120, 0.9, 0, 2, 20)
