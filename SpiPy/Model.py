from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.api import AutoReg, VAR, VARMAX
import sklearn.metrics as skm
import pandas as pd


#
# WWY_lagged = np.roll(WWY, 1, axis=0)
# WWY_lagged[0, :] = 0
#
# SWVAR = VAR(filtered_pol, exog=WWY_lagged).fit(maxlags=1, trend='c')
# # print(SWVAR.summary())
#
# for key in LocDict:
#     R2 = skm.r2_score(SWVAR.fittedvalues[key] + SWVAR.resid[key], SWVAR.fittedvalues[key])
#     print(f'The R-Squared of {key} is: {R2 * 100:.2f}%')
