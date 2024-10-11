import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import wavelength2float, nm2ev, epsilon2float

# convert wavelengths and extinction coefficients, remove empty strings
data = pd.read_csv("data/pigments.csv")
data["lambda_max"] = data["lambda_max"].apply(wavelength2float)
data["lambda_max"] = data["lambda_max"].apply(nm2ev)
data = data[~data["epsilon"].isna()]
data["epsilon"] = np.log10(data["epsilon"].apply(epsilon2float))

# remove weak transitions from calculation results and choose
# a pair: lowest absorption energy - oscillator strength
data_tddft = pd.read_csv("data/tddft_results.csv", sep=";")
data_tddft = data_tddft[data_tddft["fosc"] >= 0.01]
data_tddft["wavelength"] = data_tddft["wavelength"].apply(nm2ev)
data_tddft_temp = data_tddft.groupby(['name', 'method'], as_index=False).agg({'wavelength': 'min'})
data_tddft = pd.merge(data_tddft_temp, data_tddft, on = ['name', 'method', 'wavelength'])

# Select data and keep them separately
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

# try to build a simple linear regression between oscillator strength
# and extinction coefficient
m1 = LinearRegression().fit(data_pbe0["fosc"].values.reshape(-1, 1), data_pbe0["epsilon"].tolist())
m2 = LinearRegression().fit(data_pbe0_solv["fosc"].values.reshape(-1, 1), data_pbe0_solv["epsilon"].tolist())
m3 = LinearRegression().fit(data_wb97xd4["fosc"].values.reshape(-1, 1), data_wb97xd4["epsilon"].tolist())
m4 = LinearRegression().fit(data_wb97xd4_solv["fosc"].values.reshape(-1, 1), data_wb97xd4_solv["epsilon"].tolist())
m5 = LinearRegression().fit(data_bmk["fosc"].values.reshape(-1, 1), data_bmk["epsilon"].tolist())
m6 = LinearRegression().fit(data_bmk_solv["fosc"].values.reshape(-1, 1), data_bmk_solv["epsilon"].tolist())
m7 = LinearRegression().fit(data_camb3lyp["fosc"].values.reshape(-1, 1), data_camb3lyp["epsilon"].tolist())
m8 = LinearRegression().fit(data_camb3lyp_solv["fosc"].values.reshape(-1, 1), data_camb3lyp_solv["epsilon"].tolist())
m9 = LinearRegression().fit(data_m062x["fosc"].values.reshape(-1, 1), data_m062x["epsilon"].tolist())
m10 = LinearRegression().fit(data_m062x_solv["fosc"].values.reshape(-1, 1), data_m062x_solv["epsilon"].tolist())
m11 = LinearRegression().fit(data_b2plyp["fosc"].values.reshape(-1, 1), data_b2plyp["epsilon"].tolist())
m12 = LinearRegression().fit(data_b2plyp_solv["fosc"].values.reshape(-1, 1), data_b2plyp_solv["epsilon"].tolist())
m13 = LinearRegression().fit(data_pbeqihd["fosc"].values.reshape(-1, 1), data_pbeqihd["epsilon"].tolist())
m14 = LinearRegression().fit(data_pbeqihd_solv["fosc"].values.reshape(-1, 1), data_pbeqihd_solv["epsilon"].tolist())
m15 = LinearRegression().fit(data_wpbepp86["fosc"].values.reshape(-1, 1), data_wpbepp86["epsilon"].tolist())
m16 = LinearRegression().fit(data_wpbepp86_solv["fosc"].values.reshape(-1, 1), data_wpbepp86_solv["epsilon"].tolist())

# remove systematic error for fosc vs extinction coefficient
data_pbe0["MAE"] = abs(data_pbe0["epsilon"] - m1.predict(data_pbe0["fosc"].values.reshape(-1, 1)))
data_pbe0_solv["MAE"] = abs(data_pbe0_solv["epsilon"] - m2.predict(data_pbe0_solv["fosc"].values.reshape(-1, 1)))
data_wb97xd4["MAE"] = abs(data_wb97xd4["epsilon"] - m3.predict(data_wb97xd4["fosc"].values.reshape(-1, 1)))
data_wb97xd4_solv["MAE"] = abs(data_wb97xd4_solv["epsilon"] - m4.predict(data_wb97xd4_solv["fosc"].values.reshape(-1, 1)))
data_bmk["MAE"] = abs(data_bmk["epsilon"] - m1.predict(data_bmk["fosc"].values.reshape(-1, 1)))
data_bmk_solv["MAE"] = abs(data_bmk_solv["epsilon"] - m5.predict(data_bmk_solv["fosc"].values.reshape(-1, 1)))
data_camb3lyp["MAE"] = abs(data_camb3lyp["epsilon"] - m6.predict(data_camb3lyp["fosc"].values.reshape(-1, 1)))
data_camb3lyp_solv["MAE"] = abs(data_camb3lyp_solv["epsilon"] - m7.predict(data_camb3lyp_solv["fosc"].values.reshape(-1, 1)))
data_m062x["MAE"] = abs(data_m062x["epsilon"] - m8.predict(data_m062x["fosc"].values.reshape(-1, 1)))
data_m062x_solv["MAE"] = abs(data_m062x_solv["epsilon"] - m9.predict(data_m062x_solv["fosc"].values.reshape(-1, 1)))
data_b2plyp["MAE"] = abs(data_b2plyp["epsilon"] - m10.predict(data_b2plyp["fosc"].values.reshape(-1, 1)))
data_b2plyp_solv["MAE"] = abs(data_b2plyp_solv["epsilon"] - m11.predict(data_b2plyp_solv["fosc"].values.reshape(-1, 1)))
data_pbeqihd["MAE"] = abs(data_pbeqihd["epsilon"] - m12.predict(data_pbeqihd["fosc"].values.reshape(-1, 1)))
data_pbeqihd_solv["MAE"] = abs(data_pbeqihd_solv["epsilon"] - m13.predict(data_pbeqihd_solv["fosc"].values.reshape(-1, 1)))
data_wpbepp86["MAE"] = abs(data_wpbepp86["epsilon"] - m14.predict(data_wpbepp86["fosc"].values.reshape(-1, 1)))
data_wpbepp86_solv["MAE"] = abs(data_wpbepp86_solv["epsilon"] - m15.predict(data_wpbepp86_solv["fosc"].values.reshape(-1, 1)))


# fosc vs epsilon for PBE0
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_pbe0["fosc"], data_pbe0["epsilon"], marker='.', color='black')
ax[0].set_title("PBE0", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_pbe0["fosc"], data_pbe0["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbe0["fosc"], data_pbe0["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_pbe0["fosc"].values.reshape(-1, 1), data_pbe0["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_pbe0["fosc"].values.reshape(-1, 1)), data_pbe0["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_pbe0_solv["fosc"], data_pbe0_solv["epsilon"], marker='.', color='black')
ax[1].set_title("PBE0 - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_pbe0_solv["fosc"], data_pbe0_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbe0_solv["fosc"], data_pbe0_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_pbe0_solv["fosc"].values.reshape(-1, 1), data_pbe0_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_pbe0_solv["fosc"].values.reshape(-1, 1)), data_pbe0_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox = dict(boxstyle = 'round', fc = 'w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/pbe0_ext.png")

# fosc vs epsilon for wB97XD4
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_wb97xd4["fosc"], data_wb97xd4["epsilon"], marker='.', color='black')
ax[0].set_title("wB97XD4", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4["fosc"], data_wb97xd4["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4["fosc"], data_wb97xd4["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_wb97xd4["fosc"].values.reshape(-1, 1), data_wb97xd4["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_wb97xd4["fosc"].values.reshape(-1, 1)), data_wb97xd4["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_wb97xd4_solv["fosc"], data_wb97xd4_solv["epsilon"], marker='.', color='black')
ax[1].set_title("wB97XD4 - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_wb97xd4_solv["fosc"], data_wb97xd4_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wb97xd4_solv["fosc"], data_wb97xd4_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_wb97xd4_solv["fosc"].values.reshape(-1, 1), data_wb97xd4_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_wb97xd4_solv["fosc"].values.reshape(-1, 1)), data_wb97xd4_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox = dict(boxstyle = 'round', fc = 'w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/wb97xd_ext.png")

# fosc vs epsilon for BMK
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_bmk["fosc"], data_bmk["epsilon"], marker='.', color='black')
ax[0].set_title("BMK", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_bmk["fosc"], data_bmk["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_bmk["fosc"], data_bmk["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_bmk["fosc"].values.reshape(-1, 1), data_bmk["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_bmk["fosc"].values.reshape(-1, 1)), data_bmk["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_bmk_solv["fosc"], data_bmk_solv["epsilon"], marker='.', color='black')
ax[1].set_title("BMK - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_bmk_solv["fosc"], data_bmk_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_bmk_solv["fosc"], data_bmk_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_bmk_solv["fosc"].values.reshape(-1, 1), data_bmk_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_bmk_solv["fosc"].values.reshape(-1, 1)), data_bmk_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/bmk_ext.png")

# fosc vs epsilon for CAM-B3LYP
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_camb3lyp["fosc"], data_camb3lyp["epsilon"], marker='.', color='black')
ax[0].set_title("CAMB3LYP", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_camb3lyp["fosc"], data_camb3lyp["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_camb3lyp["fosc"], data_camb3lyp["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_camb3lyp["fosc"].values.reshape(-1, 1), data_camb3lyp["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_camb3lyp["fosc"].values.reshape(-1, 1)), data_camb3lyp["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_camb3lyp_solv["fosc"], data_camb3lyp_solv["epsilon"], marker='.', color='black')
ax[1].set_title("CAMB3LYP - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_camb3lyp_solv["fosc"], data_camb3lyp_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_camb3lyp_solv["fosc"], data_camb3lyp_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_camb3lyp_solv["fosc"].values.reshape(-1, 1), data_camb3lyp_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_camb3lyp_solv["fosc"].values.reshape(-1, 1)), data_camb3lyp_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/camb3lyp_ext.png")

# fosc vs epsilon for M06-2X
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_m062x["fosc"], data_m062x["epsilon"], marker='.', color='black')
ax[0].set_title("M06-2X", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_m062x["fosc"], data_m062x["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_m062x["fosc"], data_m062x["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_m062x["fosc"].values.reshape(-1, 1), data_m062x["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_m062x["fosc"].values.reshape(-1, 1)), data_m062x["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_m062x_solv["fosc"], data_m062x_solv["epsilon"], marker='.', color='black')
ax[1].set_title("M06-2X - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_m062x_solv["fosc"], data_m062x_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_m062x_solv["fosc"], data_m062x_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_m062x_solv["fosc"].values.reshape(-1, 1), data_m062x_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_m062x_solv["fosc"].values.reshape(-1, 1)), data_m062x_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/m062x_ext.png")


# fosc vs epsilon for B2PLYP
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_b2plyp["fosc"], data_b2plyp["epsilon"], marker='.', color='black')
ax[0].set_title("B2PLYP", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_b2plyp["fosc"], data_b2plyp["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_b2plyp["fosc"], data_b2plyp["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_b2plyp["fosc"].values.reshape(-1, 1), data_b2plyp["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_b2plyp["fosc"].values.reshape(-1, 1)), data_b2plyp["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_b2plyp_solv["fosc"], data_b2plyp_solv["epsilon"], marker='.', color='black')
ax[1].set_title("B2PLYP - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_b2plyp_solv["fosc"], data_b2plyp_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_b2plyp_solv["fosc"], data_b2plyp_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_b2plyp_solv["fosc"].values.reshape(-1, 1), data_b2plyp_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_b2plyp_solv["fosc"].values.reshape(-1, 1)), data_b2plyp_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/b2plyp_ext.png")

# fosc vs epsilon for PBEQIHD
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_pbeqihd["fosc"], data_pbeqihd["epsilon"], marker='.', color='black')
ax[0].set_title("PBEQIHD", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_pbeqihd["fosc"], data_pbeqihd["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbeqihd["fosc"], data_pbeqihd["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_pbeqihd["fosc"].values.reshape(-1, 1), data_pbeqihd["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_pbeqihd["fosc"].values.reshape(-1, 1)), data_pbeqihd["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_pbeqihd_solv["fosc"], data_pbeqihd_solv["epsilon"], marker='.', color='black')
ax[1].set_title("PBEQIHD - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_pbeqihd_solv["fosc"], data_pbeqihd_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_pbeqihd_solv["fosc"], data_pbeqihd_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_pbeqihd_solv["fosc"].values.reshape(-1, 1), data_pbeqihd_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_pbeqihd_solv["fosc"].values.reshape(-1, 1)), data_pbeqihd_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)
fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/pbeqihd_ext.png")

# fosc vs epsilon for wPBEPP86
fig, ax = plt.subplots(2, figsize=(10, 8))

ax[0].scatter(data_wpbepp86["fosc"], data_wpbepp86["epsilon"], marker='.', color='black')
ax[0].set_title("wPBEPP86", fontsize=16)
ax[0].annotate(r'$R^2$' + f' = {str(r2_score(data_wpbepp86["fosc"], data_wpbepp86["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wpbepp86["fosc"], data_wpbepp86["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_wpbepp86["fosc"].values.reshape(-1, 1), data_wpbepp86["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_wpbepp86["fosc"].values.reshape(-1, 1)), data_wpbepp86["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))

ax[1].scatter(data_wpbepp86_solv["fosc"], data_wpbepp86_solv["epsilon"], marker='.', color='black')
ax[1].set_title("wPBEPP86 - CPCM solvation", fontsize=16)
ax[1].annotate(r'$R^2$' + f' = {str(r2_score(data_wpbepp86_solv["fosc"], data_wpbepp86_solv["epsilon"]))[0:5]}\n\
' + r'$MAE$ = ' + f' {str(mean_absolute_error(data_wpbepp86_solv["fosc"], data_wpbepp86_solv["epsilon"]))[0:5]}\n\
' + r'$R^2_{sys}$' + f' = {str(m1.score(data_wpbepp86_solv["fosc"].values.reshape(-1, 1), data_wpbepp86_solv["epsilon"]))[0:5]}\n\
' + r'$MAE_{sys}$' + f' = {str(mean_absolute_error(m1.predict(data_wpbepp86_solv["fosc"].values.reshape(-1, 1)), data_wpbepp86_solv["epsilon"]))[0:5]}', xy=(3, 3), size=14, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
ax[0].tick_params(axis='both', which='major', labelsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize = 16)

fig.supylabel("Log(epsilon, 1/M/cm)", fontsize=16)
fig.supxlabel("Oscillator strength", fontsize=16)
plt.savefig("figs/wpbepp86_ext.png")
