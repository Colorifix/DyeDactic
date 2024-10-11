import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import cm
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib as mpl
from src.utils import wavelength_to_rgb

plt.rcParams.update({'font.size': 16, 'figure.constrained_layout.use': True})

with open("hyperopt/best_params/verbose.log", "r") as f:
    training_data = f.readlines()

# Here I am taking information from the trqining log
# about losses and learning rate
fold_losses = {}
fold_lr = {}
fold_validation_mse = {}
losses = []
lr = []
loss_temp = []
lr_temp = []

for line in training_data:
    # check current fold and empty list of losses and lrs
    if line.startswith("Fold"):
        fold = line.split()[1]
        fold_losses[fold] = []
        fold_lr[fold] = []
        fold_validation_mse[fold] = []
        # if we switch to the next fold put the data collected for the previous folds into dicts
        if fold != '0': 
            fold_losses[str(int(fold) - 1)].append(loss_temp)
            fold_lr[str(int(fold) - 1)].append(lr_temp)

    # ordinary line: put data into current loss and lr lists
    if line.startswith("Loss"):
        spl_line = line.split()
        loss_temp.append(float(spl_line[2][:-1])**0.5)
        lr_temp.append(float(spl_line[11]))

    # if the epoch ends put the collected data to storage dict and empty the temporary lists
    if line.startswith("Epoch") and ("Epoch 0" not in line):
        fold_losses[fold].append(loss_temp)
        fold_lr[fold].append(lr_temp)
        loss_temp = []
        lr_temp = []

    # a line containing validation error for the epoch
    if "Validation rmse" in line:
        spl_line = line.split()
        fold_validation_mse[fold].append(float(spl_line[3]))

# add temporary losses to the storage dictionary which have not been adde
fold_losses[fold].append(loss_temp)
fold_lr[fold].append(lr_temp)

# delete fold number 1 as it demonstrates problems convergence
del fold_losses['3']
del fold_lr['3']
del fold_validation_mse['3']


# compute average values from learning rates and losses for each epoch
for fold in fold_losses.keys():
    fold_losses[fold] = [np.mean(losses) for losses in fold_losses[fold]]
    fold_lr[fold] = [np.mean(lrs) for lrs in fold_lr[fold]]


n_folds = len(fold_validation_mse)
n_epochs = len(fold_validation_mse[list(fold_losses.keys())[0]])


upper_bound_losses = []
lower_bound_losses = []
upper_bound_validation_mse = []
lower_bound_validation_mse = []
mean_losses = []
mean_validation_mse = []

# to estimate the variability of training error calculate averages an standard deviation for each epoch based on 4 folds
# upper and lower bounds are taken as 3 standard deviations divided by square root of number of folds (4 in this case)
for i in range(n_epochs-1):
    mean_losses.append(np.mean([fold_losses[j][i] for j in fold_losses.keys()]))
    mean_validation_mse.append(np.mean([fold_validation_mse[j][i] for j in fold_losses.keys()]))
    upper_bound_losses.append(mean_losses[i] + 3 * np.std([fold_losses[j][i] for j in fold_losses.keys()]) / n_folds**0.5)
    lower_bound_losses.append(mean_losses[i] - 3 * np.std([fold_losses[j][i] for j in fold_losses.keys()]) / n_folds**0.5)
    upper_bound_validation_mse.append(mean_validation_mse[i] + 3 * np.std([fold_validation_mse[j][i] for j in fold_losses.keys()]))
    lower_bound_validation_mse.append(mean_validation_mse[i] - 3 * np.std([fold_validation_mse[j][i] for j in fold_losses.keys()]))


# validation and test data
exp_val = pd.read_csv('data/test_artificial.csv')
pred_val = pd.read_csv('data/preds_artificial_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')
exp_test = pd.read_csv('data/test_natural.csv')
pred_test_fold0 = pd.read_csv('data/preds_natural_fold0_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')
pred_test_fold1 = pd.read_csv('data/preds_natural_fold1_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')
pred_test_fold2 = pd.read_csv('data/preds_natural_fold2_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')
# Fold 3 training did not work as expected after several epochs the error started to grow quickly
# this is more a MPNN technical problem rather than fold issue. This behavoir can be observed some times during training
# I exclude this fold from prediction as it is undertrained.
#pred_test_fold3 = pd.read_csv('preds_natural_fold3_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')
pred_test_fold4 = pd.read_csv('data/preds_natural_fold4_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv')

s = pd.DataFrame([pred_test_fold0.peakwavs_max, pred_test_fold1.peakwavs_max, pred_test_fold2.peakwavs_max, pred_test_fold4.peakwavs_max])


fig = plt.figure()

ax = []

ax.append(fig.add_subplot(3, 1, 1))

ln1 = ax[0].plot(np.arange(n_epochs - 1), mean_validation_mse, lw = 2, label = 'Mean validation RMSE', color = 'red')
ln2 = ax[0].fill_between(np.arange(n_epochs - 1), lower_bound_validation_mse, upper_bound_validation_mse, facecolor='red', alpha=0.2, label='3 sigma range of validation RMSE')
ln3 = ax[0].plot(np.arange(n_epochs - 1), mean_losses, lw=2, label='Mean training RMSE', color='blue')
ln4 = ax[0].fill_between(np.arange(n_epochs - 1), lower_bound_losses, upper_bound_losses, facecolor='blue', alpha=0.2, label='3 sigma range of training RMSE')
               
ax2 = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
ax2.set_ylabel('learning rate')  # we already handled the x-label with ax1
ln5 = ax2.plot(np.arange(n_epochs-1), fold_lr['0'][:-1], lw=3, ls='--', label = "learning rate", color='darkgreen')
ax2.tick_params(axis='y')

lines, labels = ax[0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc = 'upper right', fontsize = 13)

ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('RMSE, eV')
ax[0].set_ylim([0.1, 0.7])
ax[0].grid()

x, y = exp_val["peakwavs_max"], pred_val["peakwavs_max"]
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
#z = np.floor(z / min(z))

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

ax.append(fig.add_subplot(3, 1, 2))
ax[1].scatter(x, y, c = z, s = 15, label = 'validation set of artificial dyes')

norm = mpl.colors.Normalize(vmin=1., vmax=1./min(z))
cb = cm.ScalarMappable(norm=norm, cmap='viridis')
cbar = fig.colorbar(cb, ax = ax[1])
cbar.set_label('Number of points in the vicinity')
ax[1].set_xticks([2.0, 3.0, 4.0, 5.0, 6.0])
ax[1].set_yticks([2.0, 3.0, 4.0, 5.0, 6.0])
ax[1].set_xlabel('Lowest light absorption energy\n maximum (Experiment), eV')
ax[1].set_ylabel('Lowest light absorption energy\n maximum (Predicted), eV')
ax[1].legend(fontsize = 13, loc='upper left')
#ax[1].scatter(x, y, c = z, s = 15, label = 'validation set of artificial dyes')

ax.append(fig.add_subplot(3, 1, 3))
print(mean_absolute_error(exp_test["peakwavs_max"], s.mean(0)))
print(r2_score(exp_test["peakwavs_max"], s.mean(0)))
ax[2].errorbar(exp_test["peakwavs_max"], s.mean(0), yerr = s.std(0),
               fmt='o', color = 'magenta', ecolor="red", markeredgecolor="red",
               capsize = 3, label = 'test set of natural dyes')
ax[2].set_xlabel('Lowest light absorption energy\n maximum (Experiment), eV')
ax[2].set_ylabel('Lowest light absorption energy\n maximum (Predicted), eV')
ax[2].set_xlim([1.5, 5.0])
ax[2].set_ylim([1.5, 4.12])
ax[2].set_xticks([2.0, 3.0, 4.0, 5.0])
ax[2].set_yticks([2.0, 3.0, 4.0])
ax[2].legend(fontsize = 13, loc='upper left')         


# Colour palette
clim = (1.6, 3.4)
wl = np.arange(clim[0], clim[1], 0.005)
colorlist = [wavelength_to_rgb(w) for w in wl]
ax[2].barh(wl, width = 0.5, height=0.005, left = 4.2, color = colorlist)
ax[2].text(3.9, 3.5, "Perceived colour\n     of the light", fontsize = 13)
ax[2].plot([2., 3.7], [2., 3.7], c = "black", linewidth = 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.3)
fig = plt.gcf()
fig.set_size_inches(8, 15)
plt.savefig("nn_fig_v3.png", dpi=300)
plt.show()


    

    
    
    
    
    
