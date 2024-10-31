# ElasticNet training produce many warnings; let's suppress them
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as R2
from scipy.stats import pearsonr
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from src.utils import wavelength2float, nm2ev


data = pd.read_csv("data/pigments.csv")
data["lambda_max"] = data["lambda_max"].apply(wavelength2float)
data["lambda_max"] = data["lambda_max"].apply(nm2ev)

# prepare subsets for training
data_ground = pd.read_csv("data/orbital_energies.csv", sep=";")
data_wb97xd4 = data_ground[data_ground["method"] == "wb97xd4"]
data_wb97xd4_solv = data_ground[data_ground["method"] == "wb97xd4_solv"]
data_wb97xd4 = pd.merge(data_wb97xd4, data, on="name")
data_wb97xd4_solv = pd.merge(data_wb97xd4_solv, data, on="name")


# remove calculated transitions with too low oscillator strength
data_tddft = pd.read_csv("data/tddft_results.csv", sep=";")
data_tddft["wavelength"] = data_tddft["wavelength"].apply(nm2ev)
data_tddft = data_tddft[data_tddft["fosc"] >= 0.01]
data_tddft = data_tddft.groupby(['name', 'method'], as_index=False).agg({'wavelength':'min'})


# remove data points with too big difference between experimental and calculated data
data_wb97xd4_solv_tddft = data_tddft[data_tddft["method"] == "wB97XD4_solv"]
data_wb97xd4_solv_tddft = pd.merge(data_wb97xd4_solv_tddft, data, on="name")
m4 = LinearRegression().fit(data_wb97xd4_solv_tddft["wavelength"].values.reshape(-1, 1), data_wb97xd4_solv_tddft["lambda_max"].tolist())
data_wb97xd4_solv_tddft["MAE"] = abs(data_wb97xd4_solv_tddft["lambda_max"] - m4.predict(data_wb97xd4_solv_tddft["wavelength"].values.reshape(-1, 1)))
reliable_mols = data_wb97xd4_solv_tddft[data_wb97xd4_solv_tddft["MAE"] < 0.5]["name"]

# filter only reliable molecules
data_wb97xd4 = pd.merge(reliable_mols, data_wb97xd4, on="name")
data_wb97xd4_solv = pd.merge(reliable_mols, data_wb97xd4_solv, on="name")

# dependent variable and descriptors
X, y = data_wb97xd4[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, data_wb97xd4["lambda_max"].values
X_solv, y_solv = data_wb97xd4_solv[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, data_wb97xd4_solv["lambda_max"].values

# CV - external test split
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_solv_cv, X_solv_test, y_solv_cv, y_solv_test = train_test_split(X_solv, y_solv, test_size=0.15, random_state=42)

# values L1 penalty to explore during CV
l1_ratio = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

# Define elastic net model objects and fit them
elastic_net_regression = ElasticNetCV(cv = 10,
                                      l1_ratio = l1_ratio,
                                      alphas = np.logspace(-5, 2, 100),
                                      max_iter = 10000,
                                      random_state = 42)

elastic_net_regression_solv = ElasticNetCV(cv = 10,
                                           l1_ratio = l1_ratio,
                                           alphas = np.logspace(-5, 2, 100),
                                           max_iter = 10000,
                                           random_state = 42)


elastic_net_regression.fit(X_cv, y_cv)
elastic_net_regression_solv.fit(X_solv_cv, y_solv_cv)


# best alpha , l1_ratio
alpha, l1_ratio = elastic_net_regression.alpha_, elastic_net_regression.l1_ratio_
alpha_solv, l1_ratio_solv = elastic_net_regression_solv.alpha_, elastic_net_regression_solv.l1_ratio_


predicted_cv_accum = []
y_cv_accum = []
predicted_cv_solv_accum = []
y_cv_solv_accum = []


# do cross-validation with best parameters to get error estimation for the whole dataset
kf = KFold(n_splits = 10, shuffle = True, random_state = 123)
for train, test in kf.split(X_cv):
    train_X, test_X, train_y, test_y = X_cv[train], X_cv[test], y_cv[train], y_cv[test]
    train_X_solv, test_X_solv, train_y_solv, test_y_solv = X_solv_cv[train], X_solv_cv[test], y_solv_cv[train], y_solv_cv[test]
    elastic_net_fold = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)  
    elastic_net_solv_fold = ElasticNet(alpha = alpha_solv, l1_ratio = l1_ratio_solv)
    elastic_net_fold.fit(train_X, train_y)
    elastic_net_solv_fold.fit(train_X_solv, train_y_solv)
    predicted_cv_accum.extend(elastic_net_fold.predict(test_X))
    predicted_cv_solv_accum.extend(elastic_net_solv_fold.predict(test_X_solv))
    y_cv_accum.extend(test_y)
    y_cv_solv_accum.extend(test_y_solv)


fig, ax = plt.subplots(2, 2)

ax[0, 0].scatter(y_cv_accum, predicted_cv_accum, marker='.', color='black')
ax[0, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 0].annotate(f"MAE = {str(MAE(predicted_cv_accum, y_cv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_accum, y_cv_accum)**0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_accum, y_cv_accum))[:4]}", xy=(2.6, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0, 0].set_title("10-fold CV (wB97XD4)")


ax[1, 0].scatter(y_cv_solv_accum, predicted_cv_solv_accum, marker='.', color='black')
ax[1, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 0].annotate(f"MAE = {str(MAE(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_solv_accum, y_cv_solv_accum)**0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}", xy=(2.6, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1, 0].set_title("10-fold CV (wB97XD4 + CPCM)")


y_ext_pred = elastic_net_regression.predict(X_test)
y_ext_pred_solv = elastic_net_regression_solv.predict(X_solv_test)


ax[0, 1].scatter(y_test, y_ext_pred, marker='.', color='black')
ax[0, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 1].annotate(f"MAE = {str(MAE(y_ext_pred, y_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred, y_test)**0.5)[:4]}\n" + 
                  f"R2 = {str(R2(y_ext_pred, y_test))[:4]}", xy = (2.75, 4.0),
                  size = 12, ha='right', va='top',
                  bbox = dict(boxstyle = 'round', fc = 'w'))
ax[0, 1].set_title(f"External test (wB97XD4)")


ax[1, 1].scatter(y_solv_test, y_ext_pred_solv, marker='.', color='black')
ax[1, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 1].annotate(f"MAE = {str(MAE(y_ext_pred_solv, y_solv_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred_solv, y_solv_test)**0.5)[:4]}\n" + 
                  F"R2 = {str(R2(y_ext_pred_solv, y_solv_test))[:4]}", xy = (2.75, 4.0),
                  size = 12, ha = 'right', va = 'top',
                  bbox = dict(boxstyle='round', fc='w'))
ax[1, 1].set_title(f"External test (wB97XD4 + CPCM)")
fig.supxlabel("Experimental energy of the most intensive light absorption, eV")
fig.supylabel("Calculated vertical transition energy, eV")

fig.tight_layout(pad=0.1)
plt.savefig("figs/gs_model.png")
plt.show()


# Do the same thing to largest subclasses of colourants
indoles_solv = data_wb97xd4_solv[data_wb97xd4_solv["class"] == "indole_heterocycles"]
indoles = data_wb97xd4[data_wb97xd4_solv["class"] == "indole_heterocycles"]
aq_solv = data_wb97xd4_solv[data_wb97xd4_solv["class"] == "anthraquinones"]
aq = data_wb97xd4[data_wb97xd4_solv["class"] == "anthraquinones"]


# generate some descriptor/dependent variable correlation information: Pearson r itself for the whole database
print(f"""
| Descriptor |  Full database (gas)  |  Full database (solvent)  |  Indole heterocycles (gas) | Indole heterocycles (solvent)|  Anthraquinones (gas) | Anthraquinones (solvent) |
|:------------------------------------------:|:----------------:|:----------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| HOMO                                       | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['lambda_max'])[0])[:5]} | {str(pearsonr(data_wb97xd4_solv['homo'], data_wb97xd4_solv['lambda_max'])[0])[:5]} |  {str(pearsonr(indoles['homo'], indoles['lambda_max'])[0])[:5]} | {str(pearsonr(indoles_solv['homo'], indoles_solv['lambda_max'])[0])[:5]} | {str(pearsonr(aq['homo'], aq['lambda_max'])[0])[:5]} | {str(pearsonr(aq_solv['homo'], aq_solv['lambda_max'])[0])[:5]} |
| LUMO                                       | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['lambda_max'])[0])[:5]} |  {str(pearsonr(data_wb97xd4_solv['lumo'], data_wb97xd4_solv['lambda_max'])[0])[:5]} | {str(pearsonr(indoles['lumo'], indoles['lambda_max'])[0])[:5]} |  {str(pearsonr(indoles_solv['lumo'], indoles_solv['lambda_max'])[0])[:5]} |{str(pearsonr(aq['lumo'], aq['lambda_max'])[0])[:5]} |  {str(pearsonr(aq_solv['lumo'], aq_solv['lambda_max'])[0])[:5]} |
| Dipole moment                              | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['lambda_max'])[0])[:5]} | {str(pearsonr(data_wb97xd4_solv['dm'], data_wb97xd4_solv['lambda_max'])[0])[:5]} | {str(pearsonr(indoles['dm'], indoles['lambda_max'])[0])[:5]} | {str(pearsonr(indoles_solv['dm'], indoles_solv['lambda_max'])[0])[:5]} | {str(pearsonr(aq['dm'], aq['lambda_max'])[0])[:5]} | {str(pearsonr(aq_solv['dm'], aq_solv['lambda_max'])[0])[:5]} |
| Electronegativity ($chi$)                 | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['lambda_max'])[0])[:5]} | {str(pearsonr(data_wb97xd4_solv['chi'], data_wb97xd4_solv['lambda_max'])[0])[:5]} |{str(pearsonr(indoles['chi'], indoles['lambda_max'])[0])[:5]} | {str(pearsonr(indoles_solv['chi'], indoles_solv['lambda_max'])[0])[:5]} | {str(pearsonr(aq['chi'], aq['lambda_max'])[0])[:5]} | {str(pearsonr(aq_solv['chi'], aq_solv['lambda_max'])[0])[:5]} |
| Hardness  ($eta$)                         | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['lambda_max'])[0])[:5]} | {str(pearsonr(data_wb97xd4_solv['eta'], data_wb97xd4_solv['lambda_max'])[0])[:5]} | {str(pearsonr(indoles['eta'], indoles['lambda_max'])[0])[:5]} | {str(pearsonr(indoles_solv['eta'], indoles_solv['lambda_max'])[0])[:5]} | {str(pearsonr(aq['eta'], aq['lambda_max'])[0])[:5]} | {str(pearsonr(aq_solv['eta'], aq_solv['lambda_max'])[0])[:5]} |
| Electrophilicity ($omega$)                | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['lambda_max'])[0])[:5]} {str(pearsonr(data_wb97xd4_solv['omega'], data_wb97xd4_solv['lambda_max'])[0])[:5]} |  {str(pearsonr(indoles['omega'], indoles['lambda_max'])[0])[:5]} {str(pearsonr(indoles_solv['omega'], indoles_solv['lambda_max'])[0])[:5]} |  {str(pearsonr(aq['omega'], aq['lambda_max'])[0])[:5]} {str(pearsonr(aq_solv['omega'], aq_solv['lambda_max'])[0])[:5]} |
| Mean polarisability ($alpha$)             | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['lambda_max'])[0])[:5]} {str(pearsonr(data_wb97xd4_solv['alpha'], data_wb97xd4_solv['lambda_max'])[0])[:5]} | {str(pearsonr(indoles['alpha'], indoles['lambda_max'])[0])[:5]} {str(pearsonr(indoles_solv['alpha'], indoles_solv['lambda_max'])[0])[:5]} |  {str(pearsonr(aq['alpha'], aq['lambda_max'])[0])[:5]} {str(pearsonr(aq_solv['alpha'], aq_solv['lambda_max'])[0])[:5]} |
| Anisotropic polarisability ($Deltaalpha$)| {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['lambda_max'])[0])[:5]} | {str(pearsonr(data_wb97xd4_solv['delta_alpha'], data_wb97xd4_solv['lambda_max'])[0])[:5]} | {str(pearsonr(indoles['delta_alpha'], indoles['lambda_max'])[0])[:5]} | {str(pearsonr(indoles_solv['delta_alpha'], indoles_solv['lambda_max'])[0])[:5]} | {str(pearsonr(aq['delta_alpha'], aq['lambda_max'])[0])[:5]} | {str(pearsonr(aq_solv['delta_alpha'], aq_solv['lambda_max'])[0])[:5]} |
""")

# generate some descriptor/dependent variable correlation information: P-value for the whole database
print(f"""
| Descriptor |  Full database (gas)  |  Full database (solvent)  |  Indole heterocycles (gas) | Indole heterocycles (solvent)|  Anthraquinones (gas) | Anthraquinones (solvent) |
|:------------------------------------------:|:----------------:|:----------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| HOMO                                       | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4_solv['homo'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(indoles['homo'], indoles['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles_solv['homo'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq['homo'], aq['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq_solv['homo'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| LUMO                                       | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(data_wb97xd4_solv['lumo'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles['lumo'], indoles['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(indoles_solv['lumo'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} |{str(pearsonr(aq['lumo'], aq['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(aq_solv['lumo'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Dipole moment                              | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4_solv['dm'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles['dm'], indoles['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles_solv['dm'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq['dm'], aq['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq_solv['dm'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Electronegativity ($chi$)                 | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4_solv['chi'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} |{str(pearsonr(indoles['chi'], indoles['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles_solv['chi'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq['chi'], aq['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq_solv['chi'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Hardness  ($eta$)                         | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4_solv['eta'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles['eta'], indoles['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles_solv['eta'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq['eta'], aq['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq_solv['eta'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Electrophilicity ($omega$)                | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(data_wb97xd4_solv['omega'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(indoles['omega'], indoles['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(indoles_solv['omega'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(aq['omega'], aq['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(aq_solv['omega'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Mean polarisability ($alpha$)             | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(data_wb97xd4_solv['alpha'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles['alpha'], indoles['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(indoles_solv['alpha'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} |  {str(pearsonr(aq['alpha'], aq['lambda_max'])[1] < 0.05)[:5]} {str(pearsonr(aq_solv['alpha'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
| Anisotropic polarisability ($Deltaalpha$)| {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4_solv['delta_alpha'], data_wb97xd4_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles['delta_alpha'], indoles['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(indoles_solv['delta_alpha'], indoles_solv['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq['delta_alpha'], aq['lambda_max'])[1] < 0.05)[:5]} | {str(pearsonr(aq_solv['delta_alpha'], aq_solv['lambda_max'])[1] < 0.05)[:5]} |
""")

# generate some descriptor cross-correlation information: Pearson r itself for the whole database
print(f"""
|      | HOMO | LUMO | Dipole moment | $chi$ | $eta$ | $omega$ | $alpha$ | $Deltaalpha$ | 
|   :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| HOMO | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['homo'])[0])[:5]}  | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['homo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['homo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['homo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['homo'])[0])[:5]}  | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['homo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['homo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['homo'])[0])[:5]} | 
| LUMO | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['lumo'])[0])[:5]}  | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['lumo'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['lumo'])[0])[:5]} | 
| Dipole moment | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['dm'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['dm'])[0])[:5]} | 
| $chi$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['chi'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['chi'])[0])[:5]} | 
| $eta$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['eta'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['eta'])[0])[:5]} | 
| $omega$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['omega'])[0])[:5]}   | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['omega'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['omega'])[0])[:5]} |
| $alpha$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['alpha'])[0])[:5]}   | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['alpha'])[0])[:5]} |
| $Deltaalpha$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['delta_alpha'])[0])[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['delta_alpha'])[0])[:5]} | 

""")

# generate some descriptor cross-correlation information: Pearson r itself for the whole database
print(f"""
|      | HOMO | LUMO | Dipole moment | $chi$ | $eta$ | $omega$ | $alpha$ | $Deltaalpha$ | 
|   :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| HOMO | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['homo'])[1] < 0.05)[:5]}  | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['homo'])[1] < 0.05)[:5]}  | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['homo'])[1] < 0.05)[:5]} | 
| LUMO | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['lumo'])[1] < 0.05)[:5]}  | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['lumo'])[1] < 0.05)[:5]} | 
| Dipole moment | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['dm'])[1] < 0.05)[:5]} | 
| $chi$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['chi'])[1] < 0.05)[:5]} | 
| $eta$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['eta'])[1] < 0.05)[:5]} | 
| $omega$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['omega'])[1] < 0.05)[:5]}   | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['omega'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['omega'])[1] < 0.05)[:5]} |
| $alpha$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['alpha'])[1] < 0.05)[:5]}   | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['alpha'])[1] < 0.05)[:5]} |
| $Deltaalpha$ | {str(pearsonr(data_wb97xd4['homo'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['lumo'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['dm'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['chi'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['eta'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['omega'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['alpha'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | {str(pearsonr(data_wb97xd4['delta_alpha'], data_wb97xd4['delta_alpha'])[1] < 0.05)[:5]} | 
""")

# prepare the separate models and information for indole-containing colourants
X, y = indoles[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, indoles["lambda_max"].values
X_solv, y_solv = indoles_solv[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, indoles_solv["lambda_max"].values

X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_solv_cv, X_solv_test, y_solv_cv, y_solv_test = train_test_split(X_solv, y_solv, test_size=0.15, random_state=42)


l1_ratio = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]


elastic_net_regression = ElasticNetCV(cv = 10,
                                      l1_ratio = l1_ratio,
                                      alphas = np.logspace(-5, 2, 100),
                                      max_iter = 10000,
                                      random_state = 42)

elastic_net_regression_solv = ElasticNetCV(cv = 10,
                                           l1_ratio = l1_ratio,
                                           alphas = np.logspace(-5, 2, 100),
                                           max_iter = 10000,
                                           random_state = 42)


elastic_net_regression.fit(X_cv, y_cv)
elastic_net_regression_solv.fit(X_solv_cv, y_solv_cv)


# best alpha , l1_ratio 
alpha, l1_ratio = elastic_net_regression.alpha_, elastic_net_regression.l1_ratio_
alpha_solv, l1_ratio_solv = elastic_net_regression_solv.alpha_, elastic_net_regression_solv.l1_ratio_

print(elastic_net_regression.coef_, elastic_net_regression.intercept_)
print(elastic_net_regression_solv.coef_, elastic_net_regression_solv.intercept_)
print(alpha, l1_ratio)
print(alpha_solv, l1_ratio_solv)


predicted_cv_accum = []
y_cv_accum = []
predicted_cv_solv_accum = []
y_cv_solv_accum = []


# do cross-validation with best parameters to get error estimation
kf = KFold(n_splits = 10, shuffle = True, random_state = 123)
for train, test in kf.split(X_cv):
    train_X, test_X, train_y, test_y = X_cv[train], X_cv[test], y_cv[train], y_cv[test]
    train_X_solv, test_X_solv, train_y_solv, test_y_solv = X_solv_cv[train], X_solv_cv[test], y_solv_cv[train], y_solv_cv[test]
    elastic_net_fold = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
    elastic_net_solv_fold = ElasticNet(alpha = alpha_solv, l1_ratio = l1_ratio_solv)
    elastic_net_fold.fit(train_X, train_y)
    elastic_net_solv_fold.fit(train_X_solv, train_y_solv)
    predicted_cv_accum.extend(elastic_net_fold.predict(test_X))
    predicted_cv_solv_accum.extend(elastic_net_solv_fold.predict(test_X_solv))
    y_cv_accum.extend(test_y)
    y_cv_solv_accum.extend(test_y_solv)


fig, ax = plt.subplots(2, 2)


ax[0, 0].scatter(y_cv_accum, predicted_cv_accum, marker='.', color='black')
ax[0, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 0].annotate(f"MAE = {str(MAE(predicted_cv_accum, y_cv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_accum, y_cv_accum) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_accum, y_cv_accum))[:4]}", xy=(3.0, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0, 0].set_title("10-fold CV (wB97XD4)")


ax[1, 0].scatter(y_cv_solv_accum, predicted_cv_solv_accum, marker='.', color='black')
ax[1, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 0].annotate(f"MAE = {str(MAE(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_solv_accum, y_cv_solv_accum) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}", xy=(3.0, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1, 0].set_title("10-fold CV (wB97XD4 + CPCM)")


y_ext_pred = elastic_net_regression.predict(X_test)
y_ext_pred_solv = elastic_net_regression_solv.predict(X_solv_test)


ax[0, 1].scatter(y_test, y_ext_pred, marker='.', color='black')
ax[0, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 1].annotate(f"MAE = {str(MAE(y_ext_pred, y_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred, y_test) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(y_ext_pred, y_test))[:4]}", xy=(3.0, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0, 1].set_title(f"External test (wB97XD4)")


ax[1, 1].scatter(y_solv_test, y_ext_pred_solv, marker='.', color='black')
ax[1, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 1].annotate(f"MAE = {str(MAE(y_ext_pred_solv, y_solv_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred_solv, y_solv_test) ** 0.5)[:4]}\n" + 
                  F"R2 = {str(R2(y_ext_pred_solv, y_solv_test))[:4]}", xy=(3.0, 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1, 1].set_title(f"External test (wB97XD4 + CPCM)")
fig.supxlabel("Experimental energy of the most intensive light absorption, eV")
fig.supylabel("Calculated vertical transition energy, eV")

fig.tight_layout(pad=0.1)
plt.savefig("figs/indole_gs_model.png")
plt.show()


# prepare the separate models and information for anthraquinones
X, y = aq[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, aq["lambda_max"].values
X_solv, y_solv = aq_solv[['homo', 'lumo', 'chi', 'eta', 'omega', 'dm', 'alpha', 'delta_alpha']].values, aq_solv["lambda_max"].values

X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_solv_cv, X_solv_test, y_solv_cv, y_solv_test = train_test_split(X_solv, y_solv, test_size=0.15, random_state=42)


l1_ratio = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]


elastic_net_regression = ElasticNetCV(cv = 10,
                                      l1_ratio = l1_ratio,
                                      alphas = np.logspace(-5, 2, 100),
                                      max_iter = 10000,
                                      random_state = 42)

elastic_net_regression_solv = ElasticNetCV(cv = 10,
                                           l1_ratio = l1_ratio,
                                           alphas = np.logspace(-5, 2, 100),
                                           max_iter = 10000,
                                           random_state = 42)

elastic_net_regression.fit(X_cv, y_cv)
elastic_net_regression_solv.fit(X_solv_cv, y_solv_cv)


# best alpha , l1_ratio 
alpha, l1_ratio = elastic_net_regression.alpha_, elastic_net_regression.l1_ratio_
alpha_solv, l1_ratio_solv = elastic_net_regression_solv.alpha_, elastic_net_regression_solv.l1_ratio_


predicted_cv_accum = []
y_cv_accum = []
predicted_cv_solv_accum = []
y_cv_solv_accum = []


# do cross-validation with best parameters to get error estimation
kf = KFold(n_splits = 10, shuffle = True, random_state = 123)
for train, test in kf.split(X_cv):
    train_X, test_X, train_y, test_y = X_cv[train], X_cv[test], y_cv[train], y_cv[test]
    train_X_solv, test_X_solv, train_y_solv, test_y_solv = X_solv_cv[train], X_solv_cv[test], y_solv_cv[train], y_solv_cv[test]
    elastic_net_fold = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)  
    elastic_net_solv_fold = ElasticNet(alpha = alpha_solv, l1_ratio = l1_ratio_solv)
    elastic_net_fold.fit(train_X, train_y)
    elastic_net_solv_fold.fit(train_X_solv, train_y_solv)
    predicted_cv_accum.extend(elastic_net_fold.predict(test_X))
    predicted_cv_solv_accum.extend(elastic_net_solv_fold.predict(test_X_solv))
    y_cv_accum.extend(test_y)
    y_cv_solv_accum.extend(test_y_solv)


fig, ax = plt.subplots(2, 2)


ax[0, 0].scatter(y_cv_accum, predicted_cv_accum, marker='.', color='black')
ax[0, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 0].annotate(f"MAE = {str(MAE(predicted_cv_accum, y_cv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_accum, y_cv_accum) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_accum, y_cv_accum))[:4]}", xy=(3., 4.),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0, 0].set_title("10-fold CV (wB97XD4)")


ax[1, 0].scatter(y_cv_solv_accum, predicted_cv_solv_accum, marker='.', color='black')
ax[1, 0].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 0].annotate(f"MAE = {str(MAE(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}\n" + 
                  f"RMSE = {str(MSE(predicted_cv_solv_accum, y_cv_solv_accum) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(predicted_cv_solv_accum, y_cv_solv_accum))[:4]}", xy=(3., 4.),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[1, 0].set_title("10-fold CV (wB97XD4 + CPCM)")


y_ext_pred = elastic_net_regression.predict(X_test)
y_ext_pred_solv = elastic_net_regression_solv.predict(X_solv_test)


ax[0, 1].scatter(y_test, y_ext_pred, marker='.', color='black')
ax[0, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[0, 1].annotate(f"MAE = {str(MAE(y_ext_pred, y_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred, y_test) ** 0.5)[:4]}\n" + 
                  f"R2 = {str(R2(y_ext_pred, y_test))[:4]}", xy=(3., 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))
ax[0, 1].set_title(f"External test (wB97XD4)")


ax[1, 1].scatter(y_solv_test, y_ext_pred_solv, marker='.', color='black')
ax[1, 1].plot([2.0, 4.0], [2.0, 4.0], color='red')
ax[1, 1].annotate(f"MAE = {str(MAE(y_ext_pred_solv, y_solv_test))[:4]}\n" + 
                  f"RMSE = {str(MSE(y_ext_pred_solv, y_solv_test) ** 0.5)[:4]}\n" + 
                  F"R2 = {str(R2(y_ext_pred_solv, y_solv_test))[:4]}", xy=(3., 4.0),
                  size=12, ha='right', va='top',
                  bbox=dict(boxstyle='round', fc='w'))

fig.supxlabel("Experimental energy of the most intensive light absorption, eV")
fig.supylabel("Calculated vertical transition energy, eV")
ax[1, 1].set_title(f"External test (wB97XD4 + CPCM)")

fig.tight_layout(pad=0.1)
plt.savefig("figs/anthraquinone_gs_model.png")
plt.show()

