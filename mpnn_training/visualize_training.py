import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import cm
from sklearn.metrics import mean_absolute_error, r2_score
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.stats import pearsonr
import os, traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def wavelength_to_rgb(wavelength_eV, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    
    NB:
    wavelength is taken in eV then converted to nm
    '''
    wavelength_eV = float(wavelength_eV)
    wavelength = 1239.8 / wavelength_eV
    
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


plt.rcParams.update({'font.size': 16, 'figure.constrained_layout.use': True})

tb_dfs = [tflog2pandas(fn) for fn in os.listdir() if fn.startswith('events.')]


train_losses = np.zeros(500).reshape(1, -1)
val_losses = np.zeros(500).reshape(1, -1)


for data in tb_dfs:
    train_loss = data[data['metric'] == 'train_loss_epoch']['value'].to_numpy().reshape(1, -1)
    train_loss = train_loss ** 0.5
    val_loss = data[data['metric'] == 'val/rmse']['value'].to_numpy().reshape(1, -1)
    
    train_losses = np.concatenate((train_losses, train_loss), 0)
    val_losses = np.concatenate((val_losses, val_loss), 0)


train_losses = train_losses[1:,:]
val_losses = val_losses[1:,:]

mean_train_losses = np.mean(train_losses, 0)
std_train_losses = np.std(train_losses, 0)
mean_val_losses = np.mean(val_losses, 0)
std_val_losses = np.std(val_losses, 0)

upper_bound_train_loss = mean_train_losses + 3 * std_train_losses
lower_bound_train_loss = mean_train_losses - 3 * std_train_losses
upper_bound_val_loss = mean_val_losses + 3 * std_val_losses
lower_bound_val_loss = mean_val_losses - 3 * std_val_losses


lr_init = 0.0010170618974826723
lr_final = 0.0008745233959951397
lr_max = 0.008213119797581435
warmup_epochs = 5
total_epochs = 500


lr_final = lr_final * lr_max
lr_init = lr_init * lr_max

warm_up_steps = np.arange(0, warmup_epochs)
exp_steps = np.arange(warmup_epochs, total_epochs)
delta = (lr_max - lr_init) / warmup_epochs
gamma = (exp_steps - warmup_epochs) / (total_epochs - warmup_epochs)
warm_up_steps = lr_init + delta * warm_up_steps
exp_steps = lr_max * (lr_final / lr_max) ** gamma
lr = np.concatenate((warm_up_steps, exp_steps))


# validation and test data
exp_val = pd.read_csv('test_artificial1.csv')
pred_val = pd.read_csv('preds_artificial_fold1.csv')
exp_test1 = pd.read_csv('test_natural1.csv')
exp_test2 = pd.read_csv('test_natural2.csv')
exp_test3 = pd.read_csv('test_natural3.csv')
exp_test4 = pd.read_csv('test_natural4.csv')
exp_test5 = pd.read_csv('test_natural5.csv')
pred_test_fold1_model0 = pd.read_csv('preds_natural_fold1_model0.csv')
pred_test_fold1_model1 = pd.read_csv('preds_natural_fold1_model1.csv')
pred_test_fold1_model2 = pd.read_csv('preds_natural_fold1_model2.csv')
pred_test_fold1_model3 = pd.read_csv('preds_natural_fold1_model3.csv')
pred_test_fold2_model0 = pd.read_csv('preds_natural_fold2_model0.csv')
pred_test_fold2_model1 = pd.read_csv('preds_natural_fold2_model1.csv')
pred_test_fold2_model2 = pd.read_csv('preds_natural_fold2_model2.csv')
pred_test_fold2_model3 = pd.read_csv('preds_natural_fold2_model3.csv')
pred_test_fold3_model0 = pd.read_csv('preds_natural_fold3_model0.csv')
pred_test_fold3_model1 = pd.read_csv('preds_natural_fold3_model1.csv')
pred_test_fold3_model2 = pd.read_csv('preds_natural_fold3_model2.csv')
pred_test_fold3_model3 = pd.read_csv('preds_natural_fold3_model3.csv')
pred_test_fold4_model0 = pd.read_csv('preds_natural_fold4_model0.csv')
pred_test_fold4_model1 = pd.read_csv('preds_natural_fold4_model1.csv')
pred_test_fold4_model2 = pd.read_csv('preds_natural_fold4_model2.csv')
pred_test_fold4_model3 = pd.read_csv('preds_natural_fold4_model3.csv')
pred_test_fold5_model0 = pd.read_csv('preds_natural_fold5_model0.csv')
pred_test_fold5_model1 = pd.read_csv('preds_natural_fold5_model1.csv')
pred_test_fold5_model2 = pd.read_csv('preds_natural_fold5_model2.csv')
pred_test_fold5_model3 = pd.read_csv('preds_natural_fold5_model3.csv')



s_pred_model0 = pd.concat([pred_test_fold1_model0.peakwavs_max, pred_test_fold2_model0.peakwavs_max, pred_test_fold3_model0.peakwavs_max, pred_test_fold4_model0.peakwavs_max, pred_test_fold5_model0.peakwavs_max])
s_pred_model1 = pd.concat([pred_test_fold1_model1.peakwavs_max, pred_test_fold2_model1.peakwavs_max, pred_test_fold3_model1.peakwavs_max, pred_test_fold4_model1.peakwavs_max, pred_test_fold5_model1.peakwavs_max])
s_pred_model2 = pd.concat([pred_test_fold1_model2.peakwavs_max, pred_test_fold2_model2.peakwavs_max, pred_test_fold3_model2.peakwavs_max, pred_test_fold4_model2.peakwavs_max, pred_test_fold5_model2.peakwavs_max])
s_pred_model3 = pd.concat([pred_test_fold1_model3.peakwavs_max, pred_test_fold2_model3.peakwavs_max, pred_test_fold3_model3.peakwavs_max, pred_test_fold4_model3.peakwavs_max, pred_test_fold5_model3.peakwavs_max])
s_pred = pd.DataFrame({'model0' : s_pred_model0, 'model1' : s_pred_model1, 'model2' : s_pred_model2, 'model3' : s_pred_model3})
s_exp = pd.concat([exp_test1.peakwavs_max, exp_test2.peakwavs_max, exp_test3.peakwavs_max, exp_test4.peakwavs_max, exp_test5.peakwavs_max])

exp_test = pd.concat([exp_test1, exp_test2, exp_test3, exp_test4, exp_test5])
print(exp_test.loc[s_pred_model0 - s_exp > 1.1])

fig = plt.figure()

ax = []

ax.append(fig.add_subplot(3, 1, 1))

mean_train_losses = np.mean(train_losses, 0)
std_train_losses = np.std(train_losses, 0)
mean_val_losses = np.mean(val_losses, 0)
std_val_losses = np.std(val_losses, 0)

upper_bound_train_loss = mean_train_losses + 3 * std_train_losses
lower_bound_train_loss = mean_train_losses - 3 * std_train_losses
upper_bound_val_loss = mean_val_losses + 3 * std_val_losses
lower_bound_val_loss = mean_val_losses - 3 * std_val_losses


ln1 = ax[0].plot(np.arange(total_epochs), mean_val_losses, lw = 2, label = 'Mean validation RMSE', color = 'red')
ln2 = ax[0].fill_between(np.arange(total_epochs), lower_bound_val_loss, upper_bound_val_loss, facecolor='red', alpha=0.2, label='3 sigma range of validation RMSE')
ln3 = ax[0].plot(np.arange(total_epochs), mean_train_losses, lw=2, label='Mean training RMSE', color='blue')
ln4 = ax[0].fill_between(np.arange(total_epochs), lower_bound_train_loss, upper_bound_train_loss, facecolor='blue', alpha=0.2, label='3 sigma range of training RMSE')
               
ax2 = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
ax2.set_ylabel('learning rate')  # we already handled the x-label with ax1
ln5 = ax2.plot(np.arange(total_epochs), lr, lw=3, ls='--', label = "learning rate", color='darkgreen')
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


ax[1].set_xlabel('Lowest light absorption energy\n maximum (Experiment), eV')
ax[1].set_ylabel('Lowest light absorption energy\n maximum (Predicted), eV')
ax[1].legend(fontsize = 13, loc='upper left')
ax[1].set_xlim([1.5, 6.0])
ax[1].set_ylim([1.5, 6.0])
#ax[1].scatter(x, y, c = z, s = 15, label = 'validation set of artificial dyes')


xy = np.vstack([s_exp, s_pred.mean(1)])
density = gaussian_kde(xy)(xy)

ax.append(fig.add_subplot(3, 1, 3))
print(mean_absolute_error(s_exp, s_pred.mean(1)))
print(r2_score(s_exp, s_pred.mean(1)))
ax[2].scatter(s_exp, s_pred.mean(1), c = density, s = 15,
               #ecolor="red", markeredgecolor="red",
              label = 'test set of natural dyes')

ax[2].plot([1.6, 4.5], [1.6, 4.5], c = "red", linewidth = 2)
ax[2].set_xlabel('Lowest light absorption energy\n maximum (Experiment), eV')
ax[2].set_ylabel('Lowest light absorption energy\n maximum (Predicted), eV')
ax[2].set_xlim([1.5, 6.0])
ax[2].set_ylim([1.5, 6.0])
ax[2].legend(fontsize = 13, loc='upper left')         


# Colour palette
clim = (1.6, 3.4)
wl = np.arange(clim[0], clim[1], 0.005)
colorlist = [wavelength_to_rgb(w) for w in wl]
ax[2].barh(wl, width = 1.2, height=0.005, left = 4.3, color = colorlist)
ax[2].text(4.25, 3.5, "Perceived colour\n     of the light", fontsize = 13)
ax[1].plot([1.6, 5.0], [1.6, 5.0], c = "red", linewidth = 2)

#plt.subplots_adjust(wspace=0.4, hspace=0.3)
#fig = plt.gcf()
fig.set_size_inches(8, 15)
plt.savefig("nn_fig_v4.png", dpi=300)
plt.show()




