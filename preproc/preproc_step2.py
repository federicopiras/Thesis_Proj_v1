"""
* Read raw concatenated files, ICA mixing/unmixing matrices, bad_channels and manually labelled bad IC components
* Apply ICA without bad components
* Interpolate bad channels
* Set new EEG reference (without apply)
* Split in single runs and save
"""


import os
import glob
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


def create_path(path):
    """
    A very simple function to create a non existing path
    :param path: path to create
    :return: nothing
    """
    if not os.path.exists(path):
        os.makedirs(path)


"""
SCRIPT STARTS HERE
"""
root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/data/preprocessing'
data_folder = os.path.join(root_path, '01.filtering')
bad_ch_folder = os.path.join(root_path, 'bad_channels')
data_paths = glob.glob(os.path.join(data_folder, '*.pkl'))
data_paths = sorted(data_paths)
bad_ch_paths = glob.glob(os.path.join(bad_ch_folder, '*.txt'))
bad_ch_paths = sorted(bad_ch_paths)
ica_sub_folders = glob.glob(os.path.join(root_path, 'ICA', 'sub*'))
ica_sub_folders = sorted(ica_sub_folders)

for bad_ch_path, file_path, ica_sub_folder in zip(bad_ch_paths, data_paths, ica_sub_folders):
    with open(file_path, 'rb') as f:
        m_dict = pickle.load(f)

    with open(bad_ch_path) as f:
        bad_channels = f.read().splitlines()

    # Read raw
    raw = m_dict['raw_conc_runs']
    data = raw.get_data()
    info = raw.info
    ch_names = info.ch_names
    info['bads'] = bad_channels

    # LOAD ICA file
    ica_path = glob.glob(os.path.join(ica_sub_folder, '*.fif'))[0]
    ica = mne.preprocessing.read_ica(ica_path)

    # LOAD bad components file
    ica_bad_path = glob.glob(os.path.join(ica_sub_folder, '*remove.csv'))[0]
    bad_components = pd.read_csv(ica_bad_path)
    bad_components = list(bad_components['IC'])

    # Apply ICA
    raw_processed = ica.apply(raw, exclude=bad_components)

    # Interpolate bad channels after ICA
    raw_processed = raw_processed.interpolate_bads(method=dict(eeg='spline'))

    # Re-referencing
    raw_processed = raw_processed.copy().set_eeg_reference(ref_channels='average', projection=True)
    data = raw_processed.get_data()
    ch_names = raw_processed.info.ch_names

    # *****************************
    # Split in single runs
    # *****************************
    # Load run length
    runs_length = m_dict['runs_length']
    runs_length = runs_length.astype(int)
    run = []

    # Split single concatenated RUN back in single RUNS.
    # First run (j=0): take the conc_run from 0:run_length[0], where inizio=0 and fine=run_length[j].
    # j-th RUN (j!=0): take the conc_run from inizio=run_length[j-1] to fine=run_length[j]+inizio

    for j, element in enumerate(runs_length):
        # print(f"element: {element}")
        if j == 0:
            inizio = 0
            fine = element
        if j != 0:
            inizio = fine
            fine = inizio + element

        run = data[:, inizio:fine]

        # Save processed data
        sub_name = ica_sub_folder.split(sep='/')[-1]
        run_name = '_ses-01_run' + str(j + 1) + '.pkl'

        save_path = os.path.join(root_path, 'preproc_runs', sub_name)
        create_path(path=save_path)
        m_dict = dict()
        m_dict['data'] = run
        m_dict['info'] = info


        with open(os.path.join(save_path, sub_name + run_name), 'wb') as handle:
            pickle.dump(m_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)