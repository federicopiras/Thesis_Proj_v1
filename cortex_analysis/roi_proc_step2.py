"""
===============
Description
===============

* Making cortex Grand Average
* Plotting in time the Grand Average by LED.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, ellip, ellipord
import pickle
import os
import glob
from matplotlib.ticker import MultipleLocator

valid_modes = ['mean', 'pca_bst', 'pca_sk']
mode = 'pca_sk'
if mode not in valid_modes:
    raise ValueError(f"invalid mode for mode={mode}. mode should be one in {valid_modes}")

# Pick up sub epochs
# Relative path may differ: this is the path where the dataset folder containing the .pkl files is located
relative_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/cortex'
folder_path = glob.glob(os.path.join(relative_path, mode + '*'))[0]
plot_path = os.path.join(folder_path, 'plot')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
print(f"folder_path: {folder_path}")
print(f"plot_path: {plot_path}")
file_paths = glob.glob(os.path.join(folder_path, '*.pkl'))
file_paths = sorted(file_paths)

# Pick up epochs and averaging

# Allocate memory
all_epochs_ = []
all_targets = []

for file_path in file_paths[:1]:
    with open(file_path, 'rb') as f:
        dict_ = pickle.load(f)
    epochs = dict_['epochs']
    info = dict_['info']
    ival = dict_['ival']
    roi_info = dict_['roi_info']
    run_labels = dict_['run_labels']
    targets = dict_['targets']
    srate = dict_['srate']
    all_epochs_.append(epochs)
    all_targets.append(targets)

all_epochs_ = np.concatenate(all_epochs_, axis=0)
all_targets = np.concatenate(all_targets)
print(f"all_epochs_.shape: {all_epochs_.shape}")
print(f"all_targets.shape: {all_targets.shape}")

# Filtering
# LP
srate = 512
wp = 10 / (srate / 2)
ws = 15 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_lp, a_lp = ellip(ord, gpass, gstop, wn, btype='low')
# HP
srate = 512
wp = 0.5 / (srate / 2)
ws = 0.01 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_hp, a_hp = ellip(ord, gpass, gstop, wn, btype='high')

for j, epoch in enumerate(all_epochs_):
    epoch = filtfilt(b_lp, a_lp, epoch)
    epoch = filtfilt(b_hp, a_hp, epoch)
    # baseline = np.mean(epoch[:, :512], axis=-1)
    # baseline = baseline.reshape(all_epochs_.shape[1], 1)
    all_epochs_[j, :, :] = epoch #- baseline

# Grand Average
grand_average = np.zeros(shape=(5, np.shape(all_epochs_)[1], np.shape(all_epochs_)[-1]))
for target_id in np.unique(targets):
    grand_average[target_id-1, ...] = np.mean(all_epochs_[all_targets==target_id, ...], axis=0)

grand_average = grand_average * 1e12
fmin = np.min(grand_average)
fmax = np.max(grand_average)
fmin = fmin + 0.25 * fmin
fmax = fmax + 0.25 * fmax
# if np.abs(fmin) < np.abs(fmax):
#    fmin = -fmax
# else:
#    fmax = -fmin
print(f"MAX grand_average: {fmax}")
print(f"MIN grand_average: {fmin}")

# Some useful parameters
roi_names = roi_info['roi_names'].tolist()
pre_time = ival[0]
post_time = ival[1]
t = np.linspace(pre_time, post_time, epochs.shape[-1])

# --------------------------
# Plotting left hemisphere
# --------------------------
left_roi_names = [left_roi_name for left_roi_name in roi_names if 'lh' in left_roi_name[-2:]]

fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(30, 30))
fig.suptitle('ROI LH', fontsize=20, fontweight='bold')
for roi, ax in zip(left_roi_names, axs.ravel()):
    idx_roi = roi_names.index(roi)
    for led in range(grand_average.shape[0]):
        ax.plot(t, grand_average[led, idx_roi, :], label='LED' + str(led + 1))
    ax.set_title(roi)
    ax.legend(prop={'size': 6})
    ax.grid()
    ax.set_ylim([fmin, fmax])
    ax.set_xlim([t[0], t[-1]])

    ## Cambio colore ai ticks che mi interessano
    # ax.get_xticklabels()[1].set_color('red')
    # ax.get_xticklabels()[3].set_color('red')
    # ax.get_xticklabels()[0].set_color('red')
    # ax.get_xticklabels()[-1].set_color('red')

    # Aggiungo vertical lines
    # ax.vlines(t[513], ymin=fmin, ymax=fmax, linestyles='--', linewidth=2)
    # ax.vlines(t[1537], ymin=fmin, ymax=fmax, linestyles='--', linewidth=2)

# Elimino gli ultimi 4 assi che sono in più.
fig.delaxes(axs[-1, -1])
fig.delaxes(axs[-1, -2])
fig.tight_layout()
fig.show()
# fig.savefig(os.path.join(plot_path, 'roi_lh.pdf'))

# --------------------------
# Plotting right hemisphere
# --------------------------
right_roi_names = [right_roi_name for right_roi_name in roi_names if 'rh' in right_roi_name[-2:]]

fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(30, 30))
fig.suptitle('ROI LH', fontsize=20, fontweight='bold')

for roi, ax in zip(left_roi_names, axs.ravel()):
    idx_roi = roi_names.index(roi)
    for led in range(grand_average.shape[0]):
        ax.plot(t, grand_average[led, idx_roi, :], label='LED' + str(led + 1))
    ax.set_title(roi)
    ax.legend(prop={'size': 6})
    ax.grid()
    ax.set_ylim([fmin, fmax])
    ax.set_xlim([t[0], t[-1]])

    ## Cambio colore ai ticks che mi interessano
    # ax.get_xticklabels()[1].set_color('red')
    # ax.get_xticklabels()[3].set_color('red')
    # ax.get_xticklabels()[0].set_color('red')
    # ax.get_xticklabels()[-1].set_color('red')

    # Aggiungo vertical lines
    # ax.vlines(t[513], ymin=fmin, ymax=fmax, linestyles='--', linewidth=2)
    # ax.vlines(t[1537], ymin=fmin, ymax=fmax, linestyles='--', linewidth=2)

# Elimino gli ultimi 4 assi che sono in più.
fig.delaxes(axs[-1, -1])
fig.delaxes(axs[-1, -2])
fig.tight_layout()
fig.show()
#fig.savefig(os.path.join(plot_path, 'roi_rh.pdf'))

# ---------------------------------------------------------------------------------------
# Plot signal of same lh and rh ROI in a (1, 2) subplot: (1,1)-->time series of left roi
#                                                        (1,2)-->time series of right ROI
# ---------------------------------------------------------------------------------------

# Need a df to use the lambda x
roi_names = roi_info['roi_names']
# Take just the ROI name (without the final lh-rh)
x = lambda x: x[:-3]
roi_names_ = roi_names.apply(x)
# Take unique roi names
roi_names_ = roi_names_.tolist()
unique_roi_names = np.unique(roi_names_).tolist()


# Dichiaro le ROI_names:
for roi in unique_roi_names:

    # Take lh ROI index
    lh_idx = np.where([roi in roi_name for roi_name in roi_names])[0][0]
    # Take rh ROI index
    rh_idx = np.where([roi in roi_name for roi_name in roi_names])[0][1]

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22, 4))
    # Plot lh ROI in the left subplot and rh ROI in the right subplot
    for led in range(grand_average.shape[0]):
        axs[0].plot(t, grand_average[led, lh_idx, :], label='LED' + str(led + 1))
        axs[1].plot(t, grand_average[led, rh_idx, :], label='LED' + str(led + 1))

    # Aggiungo titolo, label, griglia, ylim ai singoli subplots
    axs[0].set_title(roi_names[lh_idx])
    axs[0].set_ylim([fmin, fmax])
    axs[0].set_xlim([t[0], t[-1]])
    axs[0].set_ylabel('pA*m')
    axs[0].set_xlabel('time [s]')
    axs[0].legend()
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0].get_xticklabels()[2].set_color('red')
    axs[0].get_xticklabels()[4].set_color('red')
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[0].grid(which='major')
    axs[0].grid(which='minor', linestyle='--', alpha=0.5)

    axs[1].set_title(roi_names[rh_idx])
    axs[1].set_ylim([fmin, fmax])
    axs[1].set_xlim([t[0], t[-1]])
    axs[1].set_ylabel('pA*m')
    axs[1].set_xlabel('time [s]')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].get_xticklabels()[2].set_color('red')
    axs[1].get_xticklabels()[4].set_color('red')
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1].grid(which='major')
    axs[1].grid(which='minor', linestyle='--', alpha=0.5)

    # Aggiungo vertical lines
    axs[0].vlines(t[511], ymin=fmin, ymax=fmax, linestyles='-', linewidth=2, color='k')
    axs[0].vlines(t[1535], ymin=fmin, ymax=fmax, linestyles='-', linewidth=2, color='k')
    axs[1].vlines(t[511], ymin=fmin, ymax=fmax, linestyles='-', linewidth=2, color='k')
    axs[1].vlines(t[1535], ymin=fmin, ymax=fmax, linestyles='-', linewidth=2, color='k')

    ## Cambio colore ai ticks che mi interessano
    # axs[0].get_xticklabels()[1].set_color('red')
    # axs[0].get_xticklabels()[3].set_color('red')
    # axs[1].get_xticklabels()[1].set_color('red')
    # axs[1].get_xticklabels()[3].set_color('red')

    fig.show()
    # if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    # fig.savefig(os.path.join(save_path, roi_names[lh_idx][:-3] + '.pdf'))
