"""
DESCRIPTION

First stage of preprocessing:
* Filtering with a 0.5-50Hz elliptic bandpass filter
* Detrend EEG data
* Crop data to remove initial and final bad data segment
* Apply ICA and save IC figures to a path

IMPORTANT: path may differ for other users
"""


import pandas as pd
import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
import pickle
from scipy.signal import ellip, ellipord, filtfilt, detrend
from pyprep.find_noisy_channels_array import NoisyChannels
from mne_icalabel import label_components
from mne.preprocessing import ICA


def create_probab_df(raw, ica, fpath):
    """
    Creates csv file with automatic classification of IC component and probabilities.

    :param raw: Raw Object on which ICA is applied
    :param ica: ICA Object fitted to raw
    :param fpath: path to save the .csv
    :return:
    """

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    ic_number = 62 - len(raw.info['bads'])

    columns_name = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise',
                    'other']

    labels = label_components(inst=raw.pick(picks=['eeg'], exclude='bads'), ica=ica, method='iclabel')
    df_labels = pd.DataFrame(list(zip(labels['labels'], list(range(0, ic_number)))),
                             columns=['IC type', 'IC number'])
    df_prob = pd.DataFrame(labels['labels_pred_proba'], columns=columns_name)
    df_prob = pd.concat([df_labels, df_prob], axis=1)

    # swap 1st and 2nd column
    columns_names = list(df_prob.columns)
    columns_names[0], columns_names[1] = columns_names[1], columns_names[0]
    df_prob = df_prob[columns_names]

    # save
    df_prob.to_csv(os.path.join(fpath, 'IC_df_proba.csv'), sep=',', index=False)
    df = pd.DataFrame(dict(IC=[]))
    df.to_csv(os.path.join(fpath, 'IC_to_remove.csv'), sep=',', index=False)


def subplot_ica(ica, data, sfreq, psd, srate_in, time_duration, i, fpath):
    """
    Makes a plot of ICA signals on the scalp, in time and PSD
    :param ica: ICA object containing un-mixing matrix
    :param data: data in time
    :param sfreq: frequency to plot
    :param psd: Power spectral density
    :param srate_in: sampling frequency
    :param time_duration: int: time duration of the plot (in seconds)
    :param i: int: IC pick to plot
    :param fpath: path to save the image
    :return:
    """

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # Save and load ICA component image
    f = ica.plot_components(picks=i, cmap='jet', res=128, sphere=0.08)
    f.savefig(os.path.join(fpath, 'img.jpg'))
    im = plt.imread(os.path.join(fpath, 'img.jpg'))
    os.remove(os.path.join(fpath, 'img.jpg'))

    # Create a figure to plot different images
    fig = plt.figure(figsize=(40, 21))

    # First subplot: ICA component image
    fig.add_subplot(2 ,2, 1)
    plt.imshow(im)
    plt.axis('off')

    # Second subplot: PSD image
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(sfreq, psd, 'k', linewidth=2.5)
    ax.set_xlim([0, 55])
    ax.set_xlabel('frequency [Hz]', fontsize=15)
    ax.set_ylabel('$µV^2$ / Hz', fontsize=15)
    ax.grid()

    # Third subplot: First 100 seconds of IC in time
    t = np.arange(0, srate_in*time_duration) / srate_in
    ax = fig.add_subplot(2, 2, (3, 4))
    ax.plot(t, data[0:srate_in*time_duration], 'k', linewidth=2.5)
    ax.set_xlim([0, time_duration])
    ax.set_xlabel('time [s]', fontsize=15)
    ax.set_ylim([-25, 25])
    ax.set_xticks(np.linspace(0,100,11))
    ax.set_yticks(np.linspace(-25, 25, 5))
    ax.grid(which='both')

    # fig.show()
    fig.suptitle('IC_' + str(i), fontsize=20, fontweight='bold')
    fig.savefig(os.path.join(fpath, 'ICA_' + str(i) + '.png'))


"""""""""""""""""""""
SCRIPT STARTS HERE
"""""""""""""""""""""
# Select sub_paths
root_folder = '/Users/federico/University/Magistrale/00.TESI/data_original'
experiment_folders = sorted(glob.glob(os.path.join(root_folder, 'experiment*')))
relative_path = os.path.join('ses-01', 'eeg')
sbj_paths = []

for experiment_folder in experiment_folders:
    if os.path.basename(experiment_folder) == 'experiment1':
        sbj_paths_ = sorted(glob.glob(os.path.join(experiment_folder, 'sub*')))[1:]
    else:
        sbj_paths_ = sorted(glob.glob(os.path.join(experiment_folder, 'sub*')))[1:]
    sbj_paths.append(sbj_paths_)

sbj_paths = [item for sublist in sbj_paths for item in sublist]

# Define filters
# LP
srate = 512
wp = 50 / (srate / 2)
ws = 60 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_lp, a_lp = ellip(ord, gpass, gstop, wn, btype='low')
print(ord)
# HP
srate = 512
wp = 0.5 / (srate / 2)
ws = 0.01 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_hp, a_hp = ellip(ord, gpass, gstop, wn, btype='high')
print(ord)

srate_in = 512
srate_out = srate_in

for sbj_path in sbj_paths:

    # Bad channel for each run
    bad_channels_ = []

    # Container for the filtered and detrended runs
    conc_data_ = []

    # Find run paths & remove ocular paths:
    unwanted = ['OA', 'OC']
    run_paths = sorted(glob.glob(os.path.join(sbj_path, relative_path, '*.set')))
    run_paths = [rp for rp in run_paths if not any(word in rp for word in unwanted)]

    # Find event paths:
    event_paths = sorted(glob.glob(os.path.join(sbj_path, relative_path, '*events.csv')))

    # Array that contains the single run lengths. It will be saved in a .pkl file together with the RAW concatenated
    # + filtered runs & a np.array of the conc runs. In that way, I can split the concatenated runs back in the single
    # runs. The array has Nrun elements. The N-th element is the length of the N-th run.
    runs_length = np.zeros(len(event_paths))

    # **************************************************************
    # Processing for each run & for each subject
    # **************************************************************

    # Cycling through each event path, run path:
    for i, (event_path, run_path) in enumerate(zip(event_paths, run_paths)):

        # Load raw, data, info. Dropping auricular and ocular channels.
        raw = mne.io.read_raw_eeglab(run_path)
        raw = raw.drop_channels(['A1', 'A2', 'F9', 'F10'])
        data_ = raw.get_data()
        info = raw.info

        # Load event_csv:
        df = pd.read_csv(event_path)

        # Conto i LED OR
        ledor_count = np.sum(df['trial_type'] == 'LED OR')

        # ******************************
        # IF 200 LED OR: take them all
        # ******************************
        if ledor_count == 200:
            print(event_path)

            # Select sample to crop data
            idx_start = np.where([df['trial_type'] == 'LED OR'])[1][0]
            sample_start = df['sample'].iloc[idx_start]
            print(f"idx_start: {idx_start}")
            print(f"sample_start: {sample_start}")
            idx_stop = np.where([df['trial_type'] == 'LED OR'])[1][-1]
            sample_stop = df['sample'].iloc[idx_stop]
            print(f"idx_start: {idx_stop}")
            print(f"sample_start: {sample_stop}")
            sample_start = sample_start - 3 * srate_in
            sample_stop = sample_stop + 3 * srate_in

        # ***********************************************************
        # IF > 200 LED OR: take the last 200.
        # Crop data referring to the first valid LED OR
        # ***********************************************************

        if ledor_count != 200:
            print(event_path)

            # Select sample to crop data
            idx_ledor = np.where([df['trial_type'] == 'LED OR'])[1][-200:]
            idx_start = idx_ledor[0]
            sample_start = df['sample'].iloc[idx_start]
            print(f"idx_start: {idx_start}")
            print(f"sample_start: {sample_start}")
            idx_stop = np.where([df['trial_type'] == 'LED OR'])[1][-1]
            sample_stop = df['sample'].iloc[idx_stop]
            print(f"idx_stop: {idx_stop}")
            print(f"sample_stop: {sample_stop}")
            sample_start = sample_start - 3 * srate_in
            sample_stop = sample_stop + 3 * srate_in

        # Add run length
        runs_length[i] = sample_stop - sample_start

        # Change montage
        ch_names = info.ch_names
        ch_positions = raw._get_channel_positions()
        info = mne.create_info(ch_names=ch_names, sfreq=srate_out, ch_types='eeg')
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage)

        # Filter & detrend data (LP, HP)
        data_ = filtfilt(b_lp, a_lp, data_)
        data_ = filtfilt(b_hp, a_hp, data_)
        data = detrend(data_)

        # Crop data
        data = data[:, sample_start:sample_stop]

        # Concatenate data
        conc_data_.append(data)

        # Bad channel identification
        NC = NoisyChannels(data, srate_in, ch_names, ch_positions)
        bad_idx_by_ransac, channel_correlations, bad_window_fractions = NC.find_bad_by_ransac(n_samples=100)
        bad_channels_.append(bad_idx_by_ransac)

    # Make a list containing all the single bad channel for each subject
    path = os.path.join(sbj_path, 'data_preproc3', 'bad_channels')
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"bad_channels_: {bad_channels_}")
    bad_channels = np.unique(np.concatenate(bad_channels_))
    print(f"bad_channels: {bad_channels}")
    save_path = os.path.join(path, 'bad_channels.txt')
    np.savetxt(save_path, bad_channels, fmt='%s')

    # np array of concatenated data
    conc_data = np.concatenate(conc_data_, axis=-1)
    print(f"conc_data.shape: {conc_data.shape}")

    # raw object of concatenated data
    raw_conc = mne.io.RawArray(conc_data, info)
    raw_conc.set_montage(montage)

    # dict with concatenated (filtered, detrended and cropped) data and their length
    m_dict = dict()
    m_dict['conc_runs'] = conc_data
    m_dict['raw_conc_runs'] = raw_conc
    m_dict['runs_length'] = runs_length

    save_path = os.path.join(sbj_path, 'data_preproc3', 'preprocessing', '01.filtering')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'conc_filt.pkl'), 'wb') as handle:
        pickle.dump(m_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Upload bad channel to raw
    bad_file_path = os.path.join(path, 'bad_channels.txt')
    raw_conc.load_bad_channels(bad_file=bad_file_path)

    # Make ICA on single sub concatenated runs
    ica = ICA(random_state=23)
    ica.fit(raw_conc)

    save_path = os.path.join(sbj_path, 'data_preproc3', 'preprocessing', '02.ICA')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ica.save(os.path.join(save_path, sbj_path[-7:] + '_ses-01_ica.fif'), overwrite=True)

    # Take single components
    components = ica.get_sources(raw_conc)
    ica_datas = components.get_data()
    save_path = os.path.join(sbj_path, 'data_preproc3', 'preprocessing', '02.ICA')

    for i, ica_data in enumerate(ica_datas):
        psd, sfreq = mne.time_frequency.psd_array_welch(x=ica_data, fmax=60, n_per_seg=5 * srate_in,
                                                        n_fft=5 * srate_in * 2,
                                                        n_overlap=int(5 * srate_in / 2),
                                                        sfreq=512)
        subplot_ica(ica, ica_data, sfreq, psd, srate_in, time_duration=100, i=i, fpath=save_path)

    create_probab_df(raw=raw_conc, ica=ica, fpath=save_path)