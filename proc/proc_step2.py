import os
import glob
import pickle
import numpy as np
from scipy.signal import filtfilt, ellip, ellipord, spectrogram, welch
import matplotlib.pyplot as plt
import mne
import cv2
from pdf2image import convert_from_path
from matplotlib.ticker import MultipleLocator


def plot_total_grand_average(grand_average_array, info, srate, window, param, save=None, spath=''):
    """
    Plots the total grand average on the scalp.
    :param grand_average_array: array containing grand average for LED
    :param srate: sampling frequency
    :param window: window of time to compute the average of the signal centered in the desired sample
    :param param: dictionary of useful data to plot the grand average
    :param save: bool. default None. Must be True to save the figure to spath
    :param spath: path to save the figure.
    :return: figure
    """

    if save and not spath:
        raise ValueError('spath must be a path if save is True')

    seconds = param['seconds']
    v_min = param['v_min']
    v_max = param['v_max']
    samples = seconds * srate
    if samples[0] < 0:
        samples = samples - samples[0]
    samples[1:] = samples[1:] - 1

    fig, axes = plt.subplots(nrows=5, ncols=samples.shape[0], figsize=(24, 11), gridspec_kw=dict(top=0.9))
    fig.suptitle('Total Grand Average', fontsize=20, fontweight='bold')

    # Define colorbar
    colormap = plt.cm.get_cmap('jet')
    sm = plt.cm.ScalarMappable(cmap=colormap)
    vmin = v_min * 1e6
    vmax = v_max * 1e6
    sm.set_clim(vmin=vmin, vmax=vmax)

    for row, ga in enumerate(grand_average_array):
        for col, sample in enumerate(samples):
            if sample == 0:
                start = int(sample)
                stop = int(sample) + int(srate * window)
                im, cm = mne.viz.plot_topomap(np.mean(ga[:, start:stop], axis=1),
                                              info,
                                              vmin=v_min,
                                              vmax=v_max,
                                              sphere=0.08,
                                              cmap='jet',
                                              # extrapolate='local',
                                              # border='mean',
                                              axes=axes[row][col])
                # fig.colorbar(im, ax=axes[row][col], fraction=0.06, aspect=20)

            elif sample == samples[-1]:
                start = int(sample) - int(srate * window)
                stop = int(sample)
                im, cm = mne.viz.plot_topomap(np.mean(ga[:, start:stop], axis=1),
                                              info,
                                              vmin=v_min,
                                              vmax=v_max,
                                              sphere=0.08,
                                              cmap='jet',
                                              # extrapolate='local',
                                              # border='mean',
                                              axes=axes[row][col])
                # fig.colorbar(im, ax=axes[row][col], fraction=0.06, aspect=20)

            else:
                start = int(sample) - int(srate * 0.5 * window)
                stop = int(sample) + int(srate * 0.5 * window)
                im, cm = mne.viz.plot_topomap(np.mean(ga[:, start:stop], axis=1),
                                              info,
                                              vmin=v_min,
                                              vmax=v_max,
                                              sphere=0.08,
                                              cmap='jet',
                                              # extrapolate='local',
                                              # border='mean',
                                              axes=axes[row][col])
                # fig.colorbar(im, ax=axes[row][col], fraction=0.06, aspect=20)

    seconds = list(seconds)
    # seconds[4] = 'Cue'
    cols = ['{}s'.format(col) for col in seconds]
    rows = ['Led{}'.format(row) for row in [1, 2, 3, 4, 5]]
    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontweight='bold', fontsize=18)
        ax.xaxis.set_label_coords(0.55, -0.7)
        if col == '0.0s' or col == '2.0s':
            ax.xaxis.label.set_color('red')
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, y=0.3, rotation=0, labelpad=43.0, fontweight='bold', fontsize=20)

    cbar_ax = fig.add_axes([0.92, 0.075, 0.01, 0.8])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=25)
    cb.ax.set_title('µV', fontsize=25)

    fig.subplots_adjust(left=0.07,
                        bottom=0.1,
                        right=0.90,
                        top=0.95,
                        wspace=0.2,
                        hspace=0.01)
    fig.show()
    if save:
        if not os.path.exists(spath):
            os.makedirs(spath)
        path1 = os.path.join(save_path, 'total_grand_average.png')
        fig.savefig(path1)
    return fig


def select_subjects(path):
    """
    Pick up all the valid subjects path.
    :param path: path to the folder containing the experiment directories
    :return: list of all subject paths
    """
    sbjs = []
    experiments_path = sorted(glob.glob(os.path.join(path, 'experiment*')))
    for ep in experiments_path:
        if os.path.basename(ep) == 'experiment1':
            sbjs.append(sorted(glob.glob(os.path.join(ep, 'sub*')))[1:])
        else:
            sbjs.append(sorted(glob.glob(os.path.join(ep, 'sub*')))[1:])
    sbjs = sorted([s for l in sbjs for s in l])
    return sbjs


root_path = '/Users/federico/University/Magistrale/00.TESI/data_original'
sbj_paths = select_subjects(path=root_path)

experiment = 'both'
if experiment == 'experiment0':
    sbj_paths = sbj_paths[:9]
if experiment == 'experiment1':
    sbj_paths = sbj_paths[9:]

# Dichiaro un array dove conterrò tutte le epoche
# Dichiaro un array dove conterrà tutti i targets
all_epochs = []
all_targets = []

root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets'
epochs_path = glob.glob(os.path.join(root_path, 'scalp_hard_[-1,2', '*.pkl'))
epochs_path = sorted(epochs_path)

for epoch_path in epochs_path:
    # Pick up pickle epochs file path:
    with open(epoch_path, 'rb') as f:
        sub_dict = pickle.load(f)

    epochs = sub_dict['epochs']
    info = sub_dict['info']
    targets = sub_dict['targets']
    srate = sub_dict['srate']
    pre, post = sub_dict['ival']
    ch_names = sub_dict['ch_names']

    all_epochs.append(epochs)
    all_targets.append(targets)

all_epochs = np.concatenate(all_epochs)
all_targets = np.concatenate(all_targets)
# CAR
epochs_array = mne.EpochsArray(all_epochs, info)
mne.set_eeg_reference(epochs_array, ref_channels='average', projection=True)
epochs_array.apply_proj()
all_epochs = epochs_array.get_data()
print(f"all_epochs.shape:  {all_epochs.shape}")
print(f"all_targets.shape: {all_targets.shape}")

# Dichiaro i parametri del filtro
# Passa basso
srate = 512
wp = 10 / (srate / 2)
ws = 15 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_lp, a_lp = ellip(ord, gpass, gstop, wn, btype='low')

# Passa alto
srate = 512
wp = 0.5 / (srate / 2)
ws = 0.01 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_hp, a_hp = ellip(ord, gpass, gstop, wn, btype='high')

# Filtro i dati e rimuovo la baseline filtrata
m, n, p = all_epochs.shape
epochs_filt = np.zeros((m, n, p))
for j, epoch_ in enumerate(all_epochs):
    #epoch = filtfilt(b_hp, a_hp, epoch_)
    epoch_lp = filtfilt(b_lp, a_lp, epoch_)
    epoch = filtfilt(b_hp, a_hp, epoch_lp)
    baseline = np.mean(epoch[:, :512], axis=-1)
    baseline = baseline.reshape(epoch.shape[0], 1)
    epochs_filt[j, : , :] = epoch# - baseline
print(f"epoch_filt.shape: {epochs_filt.shape}")

# Faccio il grand average
m, n, p = all_epochs.shape
m = np.unique(all_targets).shape[0]
grand_average = np.zeros((m, n, p))
for j, target in enumerate(np.unique(all_targets)):
    grand_average[j, : , :] = np.mean(epochs_filt[all_targets == target, :, :], axis=0)

# Stampiamo alcune info utili su grand average
fmin = np.min(grand_average)
fmax = np.max(grand_average)
fmin = fmin + 0.25 * fmin
fmax = fmax + 0.25 * fmax
if np.abs(fmin) < np.abs(fmax):
    fmin = -fmax
else:
    fmax = -fmin

print(f"MAX grand_average: {fmax}")
print(f"MIN grand_average: {fmin}")

fmin = np.round(fmin * 1e6)
fmax = np.round(fmax * 1e6)


# Plot nel tempo diviso per LED
nsample = (np.abs(pre) + np.abs(post)) * srate
time = np.linspace(pre, post, nsample)

for ch in ch_names:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for led in range(grand_average.shape[0]):
        ax.plot(time, grand_average[led, ch_names.index(ch), :] * 1e6, label='LED' + str(led + 1), linewidth=1)
    ax.set_title(ch, fontsize=15)
    ax.legend()
    # plt.axvspan(0.25, 0.6, color='red', alpha=0.08)
    # plt.annotate('P300', xy=(0.15, 5), xycoords='data', rotation=90)
    ax.set_xlim([pre, post])
    ax.set_ylim([fmin, fmax])
    ax.set_xlabel('time [s]', fontsize=15)
    ax.set_ylabel('µV', fontsize=20)

    ax.vlines(time[511], ymin=-8, ymax=8, linestyles='-', linewidth=1.8, color='k')
    # ax.vlines(time[1535], ymin=-8, ymax=8, linestyles='-', linewidth=1.8, color='k')
    ax.hlines(y=0, xmin=-1, xmax=4, linestyles='--', linewidth=2, color='gray', alpha=0.4)

    # Cambio colore ai ticks che mi interessano
    ticks = np.arange(pre, post + 0.1, 0.25)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.get_xticklabels()[3].set_color('red')
    ax.get_xticklabels()[7].set_color('red')
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(which='major', linestyle='-', alpha=0.4)
    ax.grid(which='minor', linestyle='--', alpha=0.2)

    plt.show()
    save_path = os.path.join(
        '/Users/federico/University/Magistrale/00.TESI/data_original/THESIS_IMAGES/scalp_hard_[-1,2/CAR/bline_off',
        ch + '.pdf')
    fig.savefig(save_path)

pre_time = -1
post_time = 4
plot_param = dict(seconds=np.arange(pre_time, post_time + 0.1, 0.25),
                  v_min=-4 * 1e-6,
                  v_max=4 * 1e-6)
srate_in = 512
w = 0.25
fig = plot_total_grand_average(grand_average_array=grand_average,
                               info=info,
                               srate=srate_in,
                               window=w,
                               param=plot_param,
                               save=False)
