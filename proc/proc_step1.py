"""
EPOCHING

* Load preproc runs, original csv and target files
*
"""


import os
import glob
import pickle
import re
import pandas as pd
import numpy as np


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

root_path = '/Users/federico/University/Magistrale/00.TESI/data_original'

# Pick up preprocessed run paths
load_path = os.path.join(root_path, 'data/preprocessing/preproc_runs')
run_folders = glob.glob(os.path.join(load_path, 'sub*'))
run_folders = sorted(run_folders)

#
experiment_paths = glob.glob(os.path.join(root_path, 'experiment*'))
experiment_paths = sorted(experiment_paths)
sbj_paths = []
for experiment_path in experiment_paths:
    sbj_paths.extend(glob.glob(os.path.join(experiment_path, 'sub*')))
sbj_paths = [sbj_path for sbj_path in sbj_paths if 'sub-001' not in sbj_path]
sbj_paths = sorted(sbj_paths)


# Parametri che definiscono epoca
# sec --> durata epoca
# srate --> frequenza di campionamento dei dati
# pre_time --> secondi antecedenti al CUE
# post_time --> secondi successivi al CUE
# pre --> campioni antecedenti all'epoca
# post --> campioni successivi all'epoca
srate = 512
pre_time = -1
post_time = 4
sec = np.abs(pre_time) + np.abs(post_time)  # 5
pre = pre_time * srate
post = post_time * srate

# For each subject:
#     Load preproc_runs
#     Load raw .csv
#     Load targets .csv
for sbj_path, run_folder in zip(sbj_paths[:1], run_folders[:1]):

    # list of .csv events file names:
    csv_paths = sorted(glob.glob(os.path.join(sbj_path, 'ses-01',
                                              'eeg', '*events.csv')))

    # list of preprocessed runs
    run_paths = sorted(glob.glob(os.path.join(run_folder, '*.pkl')))

    # list of .csv targets file names:
    sequence_paths = sorted(glob.glob(os.path.join(sbj_path, 'ses-01',
                                                   'eeg', '*sequence.csv')))

    # Define a list the will contain: * all subject epochs
    #                                 * sequence of target for the 6/7 runs
    #                                 * run reference list
    # sub_epochs --> list that contains all sub epochs
    # sub_targets --> list that contains targets sequence for Nrun
    # run_labels --> list that contains the run id for the epoch [1, 1, 1, .... 6, 6, 6]
    sub_epochs = []
    sub_targets = []
    run_labels = []
    baseline_matrix = []
    sub_dict = dict()

    for csv_path, run_path, sequence_path in zip(csv_paths, run_paths, sequence_paths):
        print(csv_path)
        print(run_path)

        # Load events.csv dataframe
        df = pd.read_csv(csv_path)

        # Load preprocessed data
        with open(run_path, 'rb') as f:
            m_dict = pickle.load(f)

        data = m_dict['data']
        info = m_dict['info']

        # Select last 200 LEDOR
        idx_ledor = np.where([df['trial_type'] == 'LED OR'])[1][-200:]

        # Select start and cue idx
        idx_start = idx_ledor[0]
        idx_cue = idx_ledor[::4]

        # Select start and cue sample
        sample_start = df['sample'].iloc[idx_start]
        sample_cue = df['sample'].iloc[idx_cue]

        # The run you are loading now starts at the sample_start - 3s respect to the original ones.
        # So you need to remove the offset from the CUE
        sample_start = sample_start - 3 * srate
        sample_cue = sample_cue - sample_start

        # Select targets
        targets = pd.read_csv(sequence_path, header=None)

        targets = np.array(targets[0].tolist())

        # Allocate array for epochs
        epochs = np.zeros((sample_cue.shape[0], data.shape[0], sec * srate))
        for j, sample in enumerate(sample_cue):
            start = sample + pre # that works both for positive and negative pre
            stop = sample + post
            epoch_ = data[:, start:stop]
            start_bline = sample - 512
            baseline_ = data[:, start_bline:sample]
            baseline = np.mean(baseline_, axis=1).reshape(data.shape[0], 1)
            epoch = epoch_ - baseline
            epochs[j, :, :] = epoch
            baseline_matrix.append(baseline)

        # Lists containing all subject epochs and targets
        sub_epochs.append(epochs)
        sub_targets.append(targets)

        # Calc run id
        run_id = int(run_path.replace('.', '_').split(sep='_')[-2][-1])
        run_labels.append((run_id) * np.ones(epochs.shape[0]))

    # Create a numpy array from list of lists
    sub_targets = np.concatenate(sub_targets)
    sub_targets = sub_targets.astype(int)
    run_labels = np.concatenate(run_labels).astype(int)
    sub_epochs = np.concatenate(sub_epochs, axis=0)

    # Save dictionary with useful data
    sub_dict['baseline'] = np.array(baseline_matrix)
    sub_dict['baseline_ival'] = [-1, 0]
    sub_dict['ch_names'] = info['ch_names']
    sub_dict['epochs'] = sub_epochs
    sub_dict['info'] = info
    sub_dict['ival'] = [-1, 4]
    sub_dict['srate'] = srate
    sub_dict['run_labels'] = run_labels
    sub_dict['targets'] = sub_targets

    # Saving
    save_path = os.path.join(root_path, 'datasets', 'scalp1')
    create_path(save_path)
    file_name = run_path.split(sep='/')[-1]
    with open(os.path.join(save_path, file_name), 'wb') as handle:
        pickle.dump(sub_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # epoch_folder_name = f'epoched_-{pre_time}-{post_time}_' + direction + '_' + all_or_valid
    # relative_path = os.path.join('data_preproc3', 'processing', epoch_folder_name)
    # save_path = os.path.join(sbj_path, relative_path)
    # if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    # with open(os.path.join(save_path, 'conc_epochs.pkl'), 'wb') as handle:
    #    pickle.dump(sub_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)