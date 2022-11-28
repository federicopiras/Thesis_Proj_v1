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

valid_modes = ['mean', 'pca_bst', 'pca_sk']
mode = 'mean'
if mode not in valid_modes:
    raise ValueError(f"invalid mode for mode={mode}. mode should be one in {valid_modes}")

# Pick up sub epochs
# Relative path may differ
relative_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/cortex'
folder_path = glob.glob(os.path.join(relative_path, mode + '*'))[0]
print(folder_path)
file_paths = glob.glob(os.path.join(folder_path, '*.pkl'))
file_paths = sorted(file_paths)

# Pick up epochs and averaging

# Allocate memory
all_epochs_ = []
all_targets = []

for file_path in file_paths:
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
roi_names = roi_info['roi_names'].tolist()

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

all_epochs_ = filtfilt(b_lp, a_lp, all_epochs_)
all_epochs_ = filtfilt(b_hp, a_hp, all_epochs_)
for j, epoch in enumerate(all_epochs_):
    baseline = np.mean(epoch[:, :512], axis=-1)
    baseline = baseline.reshape(all_epochs_.shape[1], 1)
    all_epochs_[j, :, :] = epoch - baseline

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


# Plotting left hemisphere
pre_time = ival[0]
post_time = ival[1]
t = np.linspace(pre_time, post_time, epochs.shape[-1])
left_roi_names = [left_roi_name for left_roi_name in roi_names if 'lh' in left_roi_name[-2:]]

fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(30, 30))
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

