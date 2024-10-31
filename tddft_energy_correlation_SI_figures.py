import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# select energy values for each used method
data_pbe0 = data_tddft[data_tddft["method"] == "PBE0"]
data_pbe0_solv = data_tddft[data_tddft["method"] == "PBE0_solv"]
data_wb97xd4 = data_tddft[data_tddft["method"] == "wB97XD4"]
data_wb97xd4_solv = data_tddft[data_tddft["method"] == "wB97XD4_solv"]
data_bmk = data_tddft[data_tddft["method"] == "BMK"]
data_bmk_solv = data_tddft[data_tddft["method"] == "BMK_solv"]
data_camb3lyp = data_tddft[data_tddft["method"] == "CAMB3LYP"]
data_camb3lyp_solv = data_tddft[data_tddft["method"] == "CAMB3LYP_solv"]
data_m062x = data_tddft[data_tddft["method"] == "M062X"]
data_m062x_solv = data_tddft[data_tddft["method"] == "M062X_solv"]
data_b2plyp = data_tddft[data_tddft["method"] == "B2PLYP"]
data_b2plyp_solv = data_tddft[data_tddft["method"] == "B2PLYP_solv"]
data_pbeqihd = data_tddft[data_tddft["method"] == "PBEQIHD"]
data_pbeqihd_solv = data_tddft[data_tddft["method"] == "PBEQIHD_solv"]
data_wpbepp86 = data_tddft[data_tddft["method"] == "wPBEPP86"]
data_wpbepp86_solv = data_tddft[data_tddft["method"] == "wPBEPP86_solv"]
data_wb97xd4_tda = data_tddft[data_tddft["method"] == "wB97XD4_TDA"]
data_wb97xd4_tda_solv = data_tddft[data_tddft["method"] == "wB97XD4_TDA_solv"]

# merge with experimental data
data_pbe0 = pd.merge(data_pbe0, data, on="name")
data_pbe0_solv = pd.merge(data_pbe0_solv, data, on="name")
data_wb97xd4 = pd.merge(data_wb97xd4, data, on="name")
data_wb97xd4_solv = pd.merge(data_wb97xd4_solv, data, on="name")
data_bmk = pd.merge(data_bmk, data, on="name")
data_bmk_solv = pd.merge(data_bmk_solv, data, on="name")
data_camb3lyp = pd.merge(data_camb3lyp, data, on="name")
data_camb3lyp_solv = pd.merge(data_camb3lyp_solv, data, on="name")
data_m062x = pd.merge(data_m062x, data, on="name")
data_m062x_solv = pd.merge(data_m062x_solv, data, on="name")
data_b2plyp = pd.merge(data_b2plyp, data, on="name")
data_b2plyp_solv = pd.merge(data_b2plyp_solv, data, on="name")
data_pbeqihd = pd.merge(data_pbeqihd, data, on="name")
data_pbeqihd_solv = pd.merge(data_pbeqihd_solv, data, on="name")
data_wpbepp86 = pd.merge(data_wpbepp86, data, on="name")
data_wpbepp86_solv = pd.merge(data_wpbepp86_solv, data, on="name")
data_wb97xd4_tda = pd.merge(data_wb97xd4_tda, data, on="name")
data_wb97xd4_tda_solv = pd.merge(data_wb97xd4_tda_solv, data, on="name")

# Fit linear regression equations to get slope and intercept to remove systematic error
m1 = LinearRegression().fit(data_pbe0["wavelength"].values.reshape(-1, 1), data_pbe0["lambda_max"].tolist())
m2 = LinearRegression().fit(data_pbe0_solv["wavelength"].values.reshape(-1, 1), data_pbe0_solv["lambda_max"].tolist())
m3 = LinearRegression().fit(data_wb97xd4["wavelength"].values.reshape(-1, 1), data_wb97xd4["lambda_max"].tolist())
m4 = LinearRegression().fit(data_wb97xd4_solv["wavelength"].values.reshape(-1, 1), data_wb97xd4_solv["lambda_max"].tolist())
m5 = LinearRegression().fit(data_bmk["wavelength"].values.reshape(-1, 1), data_bmk["lambda_max"].tolist())
m6 = LinearRegression().fit(data_bmk_solv["wavelength"].values.reshape(-1, 1), data_bmk_solv["lambda_max"].tolist())
m7 = LinearRegression().fit(data_camb3lyp["wavelength"].values.reshape(-1, 1), data_camb3lyp["lambda_max"].tolist())
m8 = LinearRegression().fit(data_camb3lyp_solv["wavelength"].values.reshape(-1, 1), data_camb3lyp_solv["lambda_max"].tolist())
m9 = LinearRegression().fit(data_m062x["wavelength"].values.reshape(-1, 1), data_m062x["lambda_max"].tolist())
m10 = LinearRegression().fit(data_m062x_solv["wavelength"].values.reshape(-1, 1), data_m062x_solv["lambda_max"].tolist())
m11 = LinearRegression().fit(data_b2plyp["wavelength"].values.reshape(-1, 1), data_b2plyp["lambda_max"].tolist())
m12 = LinearRegression().fit(data_b2plyp_solv["wavelength"].values.reshape(-1, 1), data_b2plyp_solv["lambda_max"].tolist())
m13 = LinearRegression().fit(data_pbeqihd["wavelength"].values.reshape(-1, 1), data_pbeqihd["lambda_max"].tolist())
m14 = LinearRegression().fit(data_pbeqihd_solv["wavelength"].values.reshape(-1, 1), data_pbeqihd_solv["lambda_max"].tolist())
m15 = LinearRegression().fit(data_wpbepp86["wavelength"].values.reshape(-1, 1), data_wpbepp86["lambda_max"].tolist())
m16 = LinearRegression().fit(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1), data_wpbepp86_solv["lambda_max"].tolist())
m17 = LinearRegression().fit(data_wb97xd4_tda["wavelength"].values.reshape(-1, 1), data_wb97xd4_tda["lambda_max"].tolist())
m18 = LinearRegression().fit(data_wb97xd4_tda_solv["wavelength"].values.reshape(-1, 1), data_wb97xd4_tda_solv["lambda_max"].tolist())

# compute mean average error after removal of the systematic error
data_pbe0["MAE"] = abs(data_pbe0["lambda_max"] - m1.predict(data_pbe0["wavelength"].values.reshape(-1, 1)))
data_pbe0_solv["MAE"] = abs(data_pbe0_solv["lambda_max"] - m2.predict(data_pbe0_solv["wavelength"].values.reshape(-1, 1)))
data_wb97xd4["MAE"] = abs(data_wb97xd4["lambda_max"] - m3.predict(data_wb97xd4["wavelength"].values.reshape(-1, 1)))
data_wb97xd4_solv["MAE"] = abs(data_wb97xd4_solv["lambda_max"] - m4.predict(data_wb97xd4_solv["wavelength"].values.reshape(-1, 1)))
data_bmk["MAE"] = abs(data_bmk["lambda_max"] - m5.predict(data_bmk["wavelength"].values.reshape(-1, 1)))
data_bmk_solv["MAE"] = abs(data_bmk_solv["lambda_max"] - m6.predict(data_bmk_solv["wavelength"].values.reshape(-1, 1)))
data_camb3lyp["MAE"] = abs(data_camb3lyp["lambda_max"] - m7.predict(data_camb3lyp["wavelength"].values.reshape(-1, 1)))
data_camb3lyp_solv["MAE"] = abs(data_camb3lyp_solv["lambda_max"] - m8.predict(data_camb3lyp_solv["wavelength"].values.reshape(-1, 1)))
data_m062x["MAE"] = abs(data_m062x["lambda_max"] - m9.predict(data_m062x["wavelength"].values.reshape(-1, 1)))
data_m062x_solv["MAE"] = abs(data_m062x_solv["lambda_max"] - m10.predict(data_m062x_solv["wavelength"].values.reshape(-1, 1)))
data_b2plyp["MAE"] = abs(data_b2plyp["lambda_max"] - m11.predict(data_b2plyp["wavelength"].values.reshape(-1, 1)))
data_b2plyp_solv["MAE"] = abs(data_b2plyp_solv["lambda_max"] - m12.predict(data_b2plyp_solv["wavelength"].values.reshape(-1, 1)))
data_pbeqihd["MAE"] = abs(data_pbeqihd["lambda_max"] - m13.predict(data_pbeqihd["wavelength"].values.reshape(-1, 1)))
data_pbeqihd_solv["MAE"] = abs(data_pbeqihd_solv["lambda_max"] - m14.predict(data_pbeqihd_solv["wavelength"].values.reshape(-1, 1)))
data_wpbepp86["MAE"] = abs(data_wpbepp86["lambda_max"] - m15.predict(data_wpbepp86["wavelength"].values.reshape(-1, 1)))
data_wpbepp86_solv["MAE"] = abs(data_wpbepp86_solv["lambda_max"] - m16.predict(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1)))
data_wb97xd4_tda["MAE"] = abs(data_wb97xd4_tda["lambda_max"] - m17.predict(data_wb97xd4_tda["wavelength"].values.reshape(-1, 1)))
data_wb97xd4_tda_solv["MAE"] = abs(data_wb97xd4_tda_solv["lambda_max"] - m18.predict(data_wb97xd4_tda_solv["wavelength"].values.reshape(-1, 1)))


# PBE0 correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_pbe0["wavelength"], data_pbe0["lambda_max"], marker='.', color='black')
ax[0].set_title("PBE0", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_pbe0["wavelength"], data_pbe0["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbe0["wavelength"], data_pbe0["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_pbe0["wavelength"].values.reshape(-1, 1), data_pbe0["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_pbe0["wavelength"].values.reshape(-1, 1)), data_pbe0["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_pbe0_solv["wavelength"], data_pbe0_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("PBE0 - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_pbe0_solv["wavelength"], data_pbe0_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbe0_solv["wavelength"], data_pbe0_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m2.score(data_pbe0_solv["wavelength"].values.reshape(-1, 1), data_pbe0_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m2.predict(data_pbe0_solv["wavelength"].values.reshape(-1, 1)), data_pbe0_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/pbe0.png")

# wB97XD4 correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_wb97xd4["wavelength"], data_wb97xd4["lambda_max"], marker='.', color='black')
ax[0].set_title("wB97XD4", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4["wavelength"], data_wb97xd4["lambda_max"]))[0:6]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4["wavelength"], data_wb97xd4["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m3.score(data_wb97xd4["wavelength"].values.reshape(-1, 1), data_wb97xd4["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m3.predict(data_wb97xd4["wavelength"].values.reshape(-1, 1)), data_wb97xd4["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_wb97xd4_solv["wavelength"], data_wb97xd4_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("wB97XD4 - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4_solv["wavelength"], data_wb97xd4_solv["lambda_max"]))[0:6]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4_solv["wavelength"], data_wb97xd4_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m4.score(data_wb97xd4_solv["wavelength"].values.reshape(-1, 1), data_wb97xd4_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m4.predict(data_wb97xd4_solv["wavelength"].values.reshape(-1, 1)), data_wb97xd4_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/wb97xd.png")

# BMK correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_bmk["wavelength"], data_bmk["lambda_max"], marker='.', color='black')
ax[0].set_title("BMK", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_bmk["wavelength"], data_bmk["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_bmk["wavelength"], data_bmk["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m5.score(data_bmk["wavelength"].values.reshape(-1, 1), data_bmk["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m5.predict(data_bmk["wavelength"].values.reshape(-1, 1)), data_bmk["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_bmk_solv["wavelength"], data_bmk_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("BMK - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_bmk_solv["wavelength"], data_bmk_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_bmk_solv["wavelength"], data_bmk_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m6.score(data_bmk_solv["wavelength"].values.reshape(-1, 1), data_bmk_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m6.predict(data_bmk_solv["wavelength"].values.reshape(-1, 1)), data_bmk_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)


fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/bmk.png")

# CAM-B3LYP correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_camb3lyp["wavelength"], data_camb3lyp["lambda_max"], marker='.', color='black')
ax[0].set_title("CAMB3LYP", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_camb3lyp["wavelength"], data_camb3lyp["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_camb3lyp["wavelength"], data_camb3lyp["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m7.score(data_camb3lyp["wavelength"].values.reshape(-1, 1), data_camb3lyp["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m7.predict(data_camb3lyp["wavelength"].values.reshape(-1, 1)), data_camb3lyp["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_camb3lyp_solv["wavelength"], data_camb3lyp_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("CAMB3LYP - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_camb3lyp_solv["wavelength"], data_camb3lyp_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_camb3lyp_solv["wavelength"], data_camb3lyp_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m8.score(data_camb3lyp_solv["wavelength"].values.reshape(-1, 1), data_camb3lyp_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m8.predict(data_camb3lyp_solv["wavelength"].values.reshape(-1, 1)), data_camb3lyp_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

#fig.tight_layout()
#fig.suptitle("Application of TD-DFT calculations to predict $\\lambda_{max}$ of natural colourants", fontsize=16)
fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/camb3lyp.png")

# M06-2X correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_m062x["wavelength"], data_m062x["lambda_max"], marker='.', color='black')
ax[0].set_title("M06-2X", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_m062x["wavelength"], data_m062x["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_m062x["wavelength"], data_m062x["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m9.score(data_m062x["wavelength"].values.reshape(-1, 1), data_m062x["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m9.predict(data_m062x["wavelength"].values.reshape(-1, 1)), data_m062x["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_m062x_solv["wavelength"], data_m062x_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("M06-2X - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_m062x_solv["wavelength"], data_m062x_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_m062x_solv["wavelength"], data_m062x_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m10.score(data_m062x_solv["wavelength"].values.reshape(-1, 1), data_m062x_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m10.predict(data_m062x_solv["wavelength"].values.reshape(-1, 1)), data_m062x_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

#fig.tight_layout()
#fig.suptitle("Application of TD-DFT calculations to predict $\\lambda_{max}$ of natural colourants", fontsize=16)
fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/m062x.png")

# B2PLYP correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_b2plyp["wavelength"], data_b2plyp["lambda_max"], marker='.', color='black')
ax[0].set_title("B2PLYP", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_b2plyp["wavelength"], data_b2plyp["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_b2plyp["wavelength"], data_b2plyp["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m11.score(data_b2plyp["wavelength"].values.reshape(-1, 1), data_b2plyp["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m11.predict(data_b2plyp["wavelength"].values.reshape(-1, 1)), data_b2plyp["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_b2plyp_solv["wavelength"], data_b2plyp_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("B2PLYP - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_b2plyp_solv["wavelength"], data_b2plyp_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_b2plyp_solv["wavelength"], data_b2plyp_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m12.score(data_b2plyp_solv["wavelength"].values.reshape(-1, 1), data_b2plyp_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m12.predict(data_b2plyp_solv["wavelength"].values.reshape(-1, 1)), data_b2plyp_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)


#fig.tight_layout()
#fig.suptitle("Application of TD-DFT calculations to predict $\\lambda_{max}$ of natural colourants", fontsize=16)
fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/b2plyp.png")

fig, ax = plt.subplots(2, figsize=(10, 8))


# PBEQIHD correlation figure
ax[0].scatter(data_pbeqihd["wavelength"], data_pbeqihd["lambda_max"], marker='.', color='black')
ax[0].set_title("PBEQIHD", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_pbeqihd["wavelength"], data_pbeqihd["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbeqihd["wavelength"], data_pbeqihd["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m13.score(data_pbeqihd["wavelength"].values.reshape(-1, 1), data_pbeqihd["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m13.predict(data_pbeqihd["wavelength"].values.reshape(-1, 1)), data_pbeqihd["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_pbeqihd_solv["wavelength"], data_pbeqihd_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("PBEQIHD - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_pbeqihd_solv["wavelength"], data_pbeqihd_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbeqihd_solv["wavelength"], data_pbeqihd_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m14.score(data_pbeqihd_solv["wavelength"].values.reshape(-1, 1), data_pbeqihd_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m14.predict(data_pbeqihd_solv["wavelength"].values.reshape(-1, 1)), data_pbeqihd_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)


#fig.tight_layout()
#fig.suptitle("Application of TD-DFT calculations to predict $\\lambda_{max}$ of natural colourants", fontsize=16)
fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/pbeqihd.png")

# wPBEPP86 correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_wpbepp86["wavelength"], data_wpbepp86["lambda_max"], marker='.', color='black')
ax[0].set_title("wPBEPP86", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_wpbepp86["wavelength"], data_wpbepp86["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wpbepp86["wavelength"], data_wpbepp86["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m15.score(data_wpbepp86["wavelength"].values.reshape(-1, 1), data_wpbepp86["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m15.predict(data_wpbepp86["wavelength"].values.reshape(-1, 1)), data_wpbepp86["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("wPBEPP86 - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wpbepp86_solv["wavelength"], data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m16.score(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1), data_wpbepp86_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m16.predict(data_wpbepp86_solv["wavelength"].values.reshape(-1, 1)), data_wpbepp86_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/wpbepp86.png")

# wB97XD4 - TDA  correlation figure
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_wb97xd4_tda["wavelength"], data_wb97xd4_tda["lambda_max"], marker='.', color='black')
ax[0].set_title("wB97XD4 - TDA", fontsize=16)
ax[0].plot([2.0, 4.0], [2.0, 4.0], color = 'red')
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4_tda["wavelength"], data_wb97xd4_tda["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4_tda["wavelength"], data_wb97xd4_tda["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m17.score(data_wb97xd4_tda["wavelength"].values.reshape(-1, 1), data_wb97xd4_tda["lambda_max"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m17.predict(data_wb97xd4_tda["wavelength"].values.reshape(-1, 1)), data_wb97xd4_tda["lambda_max"]))[0:5]}', xy=(2.5, 5.5), size=12, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].set_xlim(1.5, 6.0)
ax[0].set_ylim(1.5, 6.0)


ax[1].scatter(data_wb97xd4_tda_solv["wavelength"], data_wb97xd4_tda_solv["lambda_max"], marker='.', color='black')
ax[1].set_title("wB97XD4 - TDA - CPCM solvation", fontsize=16)
ax[1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4_tda_solv["wavelength"], data_wb97xd4_tda_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4_tda_solv["wavelength"], data_wb97xd4_tda_solv["lambda_max"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m18.score(data_wb97xd4_tda_solv["wavelength"].values.reshape(-1, 1), data_wb97xd4_tda_solv["lambda_max"]))[0:5]}\n\
' + r'$MAE$ = ' + f' = {str(mean_absolute_error(m18.predict(data_wb97xd4_tda_solv["wavelength"].values.reshape(-1, 1)), data_wb97xd4_tda_solv["lambda_max"]))[0:5]}', xy=(2.5, 5.5),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1].set_xlim(1.5, 6.0)
ax[1].set_ylim(1.5, 6.0)
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

#fig.tight_layout()
#fig.suptitle("Application of TD-DFT calculations to predict energy of an absorption peak $\\lambda_{max}$ of natural colourants", fontsize=16)
fig.supylabel("Experimental energy of strongest light absorption, eV", fontsize=16)
fig.supxlabel("Predicted energy of strongest light absorption, eV", fontsize=16)
plt.savefig("figs/wb97xd4_tda.png")




