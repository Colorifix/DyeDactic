import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
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

# group by classes and filter out minor classes
s = data.groupby(['class']).size()
large_classes = s[s > 5].reset_index()
large_classes = large_classes['class'].tolist() 
large_classes.remove("anthracyclinones")

data_pbe0 = data_pbe0[data_pbe0["class"].isin(large_classes)]
data_pbe0_solv = data_pbe0_solv[data_pbe0_solv["class"].isin(large_classes)]
data_wb97xd4 = data_wb97xd4[data_wb97xd4["class"].isin(large_classes)]
data_wb97xd4_solv = data_wb97xd4_solv[data_wb97xd4_solv["class"].isin(large_classes)]
data_bmk = data_bmk[data_bmk["class"].isin(large_classes)]
data_bmk_solv = data_bmk_solv[data_bmk_solv["class"].isin(large_classes)]
data_camb3lyp = data_camb3lyp[data_camb3lyp["class"].isin(large_classes)]
data_camb3lyp_solv = data_camb3lyp_solv[data_camb3lyp_solv["class"].isin(large_classes)]
data_m062x = data_m062x[data_m062x["class"].isin(large_classes)]
data_m062x_solv = data_m062x_solv[data_m062x_solv["class"].isin(large_classes)]
data_b2plyp = data_b2plyp[data_b2plyp["class"].isin(large_classes)]
data_b2plyp_solv = data_b2plyp_solv[data_b2plyp_solv["class"].isin(large_classes)]
data_pbeqihd = data_pbeqihd[data_pbeqihd["class"].isin(large_classes)]
data_pbeqihd_solv = data_pbeqihd_solv[data_pbeqihd_solv["class"].isin(large_classes)]
data_wpbepp86 = data_wpbepp86[data_wpbepp86["class"].isin(large_classes)]
data_wpbepp86_solv = data_wpbepp86_solv[data_wpbepp86_solv["class"].isin(large_classes)]

# get mean absolute errors for each class
mae_data_pbe0 = data_pbe0.groupby(['class']).apply(lambda x: mean_absolute_error(m1.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_pbe0_solv = data_pbe0_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m2.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_wb97xd4 = data_wb97xd4.groupby(['class']).apply(lambda x: mean_absolute_error(m3.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_wb97xd4_solv = data_wb97xd4_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m4.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_bmk = data_bmk.groupby(['class']).apply(lambda x: mean_absolute_error(m5.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_bmk_solv = data_bmk_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m6.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_camb3lyp = data_camb3lyp.groupby(['class']).apply(lambda x: mean_absolute_error(m7.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_camb3lyp_solv = data_camb3lyp_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m8.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_m062x = data_m062x.groupby(['class']).apply(lambda x: mean_absolute_error(m9.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_m062x_solv = data_m062x_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m10.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_b2plyp = data_b2plyp.groupby(['class']).apply(lambda x: mean_absolute_error(m11.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_b2plyp_solv = data_b2plyp_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m12.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_pbeqihd = data_pbeqihd.groupby(['class']).apply(lambda x: mean_absolute_error(m13.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_pbeqihd_solv = data_pbeqihd_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m14.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_wpbepp86 = data_wpbepp86.groupby(['class']).apply(lambda x: mean_absolute_error(m15.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))
mae_data_wpbepp86_solv = data_wpbepp86_solv.groupby(['class']).apply(lambda x: mean_absolute_error(m16.predict(x["fosc"].values.reshape(-1, 1)), x["epsilon"]))

# compute Pearson correlation coefficients for all major classes of colourants
pearsonr_data_pbe0 = data_pbe0.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_pbe0_solv = data_pbe0_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_wb97xd4 = data_wb97xd4.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_wb97xd4_solv = data_wb97xd4_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_bmk = data_bmk.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_bmk_solv = data_bmk_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_camb3lyp = data_camb3lyp.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_camb3lyp_solv = data_camb3lyp_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_m062x = data_m062x.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_m062x_solv = data_m062x_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_b2plyp = data_b2plyp.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_b2plyp_solv = data_b2plyp_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_pbeqihd = data_pbeqihd.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_pbeqihd_solv = data_pbeqihd_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_wpbepp86 = data_wpbepp86.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])
pearsonr_data_wpbepp86_solv = data_wpbepp86_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[0])

# compute P-values for Pearson correlation coefficients for all major classes of colourants
pearsonr_data_pbe0_pval = data_pbe0.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_pbe0_solv_pval = data_pbe0_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_wb97xd4_pval = data_wb97xd4.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_wb97xd4_solv_pval = data_wb97xd4_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_bmk_pval = data_bmk.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_bmk_solv_pval = data_bmk_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_camb3lyp_pval = data_camb3lyp.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_camb3lyp_solv_pval = data_camb3lyp_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_m062x_pval = data_m062x.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_m062x_solv_pval = data_m062x_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_b2plyp_pval = data_b2plyp.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_b2plyp_solv_pval = data_b2plyp_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_pbeqihd_pval = data_pbeqihd.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_pbeqihd_solv_pval = data_pbeqihd_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_wpbepp86_pval = data_wpbepp86.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])
pearsonr_data_wpbepp86_solv_pval = data_wpbepp86_solv.groupby(['class']).apply(lambda x: pearsonr(x["fosc"], x["epsilon"])[1])

maes = pd.DataFrame({"pbe0": mae_data_pbe0,
                     "pbe0_solv": mae_data_pbe0_solv,
                     "wb97xd4": mae_data_wb97xd4,
                     "wb97xd4_solv": mae_data_wb97xd4_solv,
                     "bmk": mae_data_bmk,
                     "bmk_solv": mae_data_bmk_solv,
                     "camb3lyp": mae_data_camb3lyp,
                     "camb3lyp_solv": mae_data_camb3lyp_solv,
                     "m062x": mae_data_m062x,
                     "m062x_solv": mae_data_m062x_solv,
                     "b2plyp": mae_data_b2plyp,
                     "b2plyp_solv": mae_data_b2plyp_solv,
                     "pbeqihd": mae_data_pbeqihd,
                     "pbeqihd_solv": mae_data_pbeqihd_solv,
                     "wpbepp86": mae_data_wpbepp86,
                     "wpbepp86_solv": mae_data_wpbepp86_solv,
})

print(maes.to_markdown())
pearsons = pd.DataFrame({"pbe0": pearsonr_data_pbe0,
                         "pbe0_solv": pearsonr_data_pbe0_solv,
                         "wb97xd4": pearsonr_data_wb97xd4,
                         "wb97xd4_solv": pearsonr_data_wb97xd4_solv,
                         "bmk": pearsonr_data_bmk,
                         "bmk_solv": pearsonr_data_bmk_solv,
                         "camb3lyp": pearsonr_data_camb3lyp,
                         "camb3lyp_solv": pearsonr_data_camb3lyp_solv,
                         "m062x": pearsonr_data_m062x,
                         "m062x_solv": pearsonr_data_m062x_solv,
                         "b2plyp": pearsonr_data_b2plyp,
                         "b2plyp_solv": pearsonr_data_b2plyp_solv,
                         "pbeqihd": pearsonr_data_pbeqihd,
                         "pbeqihd_solv": pearsonr_data_pbeqihd_solv,
                         "wpbepp86": pearsonr_data_wpbepp86,
                         "wpbepp86_solv": pearsonr_data_wpbepp86_solv,
})  

pearsons_pval = pd.DataFrame({"pbe0": pearsonr_data_pbe0_pval,
                              "pbe0_solv": pearsonr_data_pbe0_solv_pval,
                              "wb97xd4": pearsonr_data_wb97xd4_pval,
                              "wb97xd4_solv": pearsonr_data_wb97xd4_solv_pval,
                              "bmk": pearsonr_data_bmk_pval,
                              "bmk_solv": pearsonr_data_bmk_solv_pval,
                              "camb3lyp": pearsonr_data_camb3lyp_pval,
                              "camb3lyp_solv": pearsonr_data_camb3lyp_solv_pval,
                              "m062x": pearsonr_data_m062x_pval,
                              "m062x_solv": pearsonr_data_m062x_solv_pval,
                              "b2plyp": pearsonr_data_b2plyp_pval,
                              "b2plyp_solv": pearsonr_data_b2plyp_solv_pval,
                              "pbeqihd": pearsonr_data_pbeqihd_pval,
                              "pbeqihd_solv": pearsonr_data_pbeqihd_solv_pval,
                              "wpbepp86": pearsonr_data_wpbepp86_pval,
                              "wpbepp86_solv": pearsonr_data_wpbepp86_solv_pval,
})

pearsons = pearsons.applymap(lambda x: str(x)[:4])
pearsons[pearsons_pval < 0.05] = pearsons[pearsons_pval < 0.05].apply(lambda x: '<div class="green">' + x + '</div>')
print("""%%html
<style>
    .green {
        background-color: #00FF00;
    }
</style>""")
print(pearsons.to_markdown())




