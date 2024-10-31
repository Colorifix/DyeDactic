import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import wavelength2float, nm2ev, wavelength_to_rgb


pd.set_option('display.max_rows', 400)

# read pigments file and convert highest lambda max to eV
data = pd.read_csv("data/pigments.csv")
data["lambda_max"] = data["lambda_max"].apply(wavelength2float)
data["lambda_max"] = data["lambda_max"].apply(nm2ev)

# read the tddft calculation results, convert nm to eV
# and filter low intensity electron transitions
# taking the lowest energy transition
data_tddft = pd.read_csv("data/tddft_results.csv", sep=";")
data_tddft["wavelength"] = data_tddft["wavelength"].apply(nm2ev)
data_tddft = data_tddft[data_tddft["fosc"] >= 0.01]
data_tddft = data_tddft.groupby(['name', 'method'], as_index=False).agg({'wavelength':'min'})

# select energy values only for wPBEPP86 with solvation model applied
data_wpbepp86_solv = data_tddft[data_tddft["method"] == "wPBEPP86_solv"]

# merge with experimental data
data_wpbepp86_solv = pd.merge(data_wpbepp86_solv, data, on="name")

# Fit linear regression equations to get slope and intercept to remove systematic error
m16 = LinearRegression().fit(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1), data_wpbepp86_solv["lambda_max"].tolist())

# compute mean average error after removal of the systematic error
data_wpbepp86_solv["MAE"] = abs(data_wpbepp86_solv["lambda_max"] - m16.predict(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1)))



fig, ax = plt.subplots(1, figsize=(8, 8))
ax.scatter(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"], marker='.', color='black')

# draw only points with error > 1.0 eV
mask = np.abs(m16.predict(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1)) -  data_wpbepp86_solv["lambda_max"]) > 0.95

ax.scatter(data_wpbepp86_solv["wavelength"][mask], data_wpbepp86_solv["lambda_max"][mask], marker='s', edgecolor = 'black', color = 'red', s = 50)
for _, item in data_wpbepp86_solv[mask].iterrows():
    print(item)
    ax.annotate(item["name"], (item["wavelength"] - 0.6, item["lambda_max"] + 0.08), fontsize = 14)

# ax.set_title("wPBEPP86 - CPCM solvation")
ax.plot([1.5, 4.0], [1.5, 4.0], color='red')
ax.annotate(r'$R^2$' + f' = {str(r2_score(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m16.score(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1), data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m16.predict(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1)), data_wpbepp86_solv["lambda_max"]))[0:5]}', xy=(5.2, 2.2),
                  size=14, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax.set_xlim(0.4, 5.0)
ax.set_ylim(1.4, 6.0)
ax.tick_params(axis='both', which='major', labelsize = 14)
clim = (1.5, 3.6)
wl = np.arange(clim[0], clim[1], 0.005)
colorlist = [wavelength_to_rgb(w) for w in wl]
ax.barh(wl, width = 0.7, height = 0.005, left = 0.67, color = colorlist)
ax.text(0.45, 3.7, "Perceived colour\n     of the light", fontsize = 12)


#fig.tight_layout()
ax.set_ylabel("Experimental energy of strongest\n light absorption, eV", fontsize=16)
ax.set_xlabel("Predicted energy of strongest\n light absorption, eV", fontsize=16)
plt.show()
plt.savefig("figs/wpbepp86_main_text.png")


