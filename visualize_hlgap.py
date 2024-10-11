import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
from sklearn.linear_model import LinearRegression
from math import sqrt


def wavelength2float(wstring):
    try:
        wstring = float(wstring)
        return wstring
    except:
        wstring = np.max([float(i) for i in wstring.split(";")])
        return wstring


def nm2ev(wv):
	return 1239.8 / wv

data = pd.read_csv("../data/pigments_with_hlgap.csv")

data["lambda_max"] = data["lambda_max"].apply(wavelength2float)
data["lambda_max"] = data["lambda_max"].apply(nm2ev)

m1 = LinearRegression().fit(data["hlgap"].values.reshape(-1, 1), data["lambda_max"].tolist())

plt.scatter(data["hlgap"], data["lambda_max"], marker='.', color='black')
plt.plot([1.6, 3.5], [1.6, 3.5], color='red')
plt.annotate(r'$R^2$' + f' = {str(r2_score(data["hlgap"], data["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data["hlgap"], data["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data["hlgap"].values.reshape(-1, 1), data["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data["hlgap"].values.reshape(-1, 1)), data["lambda_max"]))[0:5]}', xy=(1.5, 4.8),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
plt.xlim(0., 4.0)
plt.ylim(1.5, 5.0)
plt.tick_params(axis='both', which='major', labelsize = 12)
plt.xlabel("HOMO-LUMO gap calculated with GFN2-xTB, eV", fontsize = 12)
plt.ylabel("Energy of the most intensive absorption, eV", fontsize = 12)
plt.show()