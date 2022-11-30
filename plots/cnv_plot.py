"""
A simple script to plot the within subject average in a subplot

"""

import matplotlib.pyplot as plt
import os
import glob
import pickle
import numpy as np
import re

root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/scalp'
epochs_paths = glob.glob(os.path.join(root_path, '*.pkl'))
epochs_paths = sorted(epochs_paths)
grand_average = []
for epochs_path in epochs_paths:
    with open(epochs_path, 'rb') as f:
        m_dict = pickle.load(f)
    epochs = m_dict['epochs']
    grand_average.append(np.mean(epochs, axis=0))
grand_average = np.array(grand_average)*1e5
fmin = np.min(grand_average) + 0.25*np.min(grand_average)
fmax = np.max(grand_average) + 0.25*np.max(grand_average)
ch_names = m_dict['ch_names']
pre, post = m_dict['ival']
srate = m_dict['srate']
t = np.linspace(pre, post, srate*(np.abs(pre)+np.abs(post)))
zline_channels = ['Fpz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
for zline_channel in zline_channels:
    idx = ch_names.index(zline_channel)
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 12))
    fig.suptitle(zline_channel)
    for ga, epochs_path, ax in zip(grand_average, epochs_paths, axs.ravel()):
        ax.plot(t, ga[idx, :])
    ax.set_title(re.split('[/ _]', epochs_path)[-3])
    ax.set_ylim([fmin, fmax])
    ax.legend()
    fig.show()