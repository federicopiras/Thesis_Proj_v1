"""
===============
Description
===============

Testing different scouting functions to aggregate multiple dipoles time series in a ROI in a single time series, that
is representative for the selected ROI.

* mean,     [-1, 4]
* pca_bst,  [-1, 2]
* pca_sk,   [-1, 2]
* mean_pca, [-1, 4]

"""


import mne
import os
import glob
import pandas as pd
import pickle
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.source_space import label_src_vertno_sel
import time
from sklearn.decomposition import PCA


def del_invalid_labels(labels_, word_list_):
    """
    Removes unwanted labels and create df with ROI label name and ROI number of vertices
    """
    labels_ = [label_ for label_ in labels_ if not any(word in label_.name for word in word_list_)]
    labels_list = []
    vertex_list = []
    for label_ in labels_:
        labels_list.append(label_.name)
        vertex_list.append(len(label_.vertices))
    df_ = pd.DataFrame(list(zip(labels_list, vertex_list)), columns=['roi_names', 'roi_n_vertices'])
    return labels_, df_


def pca_bst(F, reconcile_sign=True):
    """
    Scouting function. Aggregates signal by PCA, and flips sign
    :param F: data. (Nx3)xT
    :param reconcile_sign: if True, flips the sign
    :return: single time series, representing all the time series inside the ROI
    """
    import scipy

    if F.ndim > 1:
        f_mean = np.mean(F, axis=1)
        f_mean = f_mean.reshape(f_mean.shape[0], 1)
        f = F - f_mean
    elif F.ndim <= 1:
        f_mean = np.mean(F, axis=0)
        f = (F - f_mean).reshape((F.shape[0], 1))
        f_mean = f_mean.reshape((1, 1))

    # Signal decomposition
    u, s, vh = scipy.linalg.svd(f, full_matrices=False)
    s = s.reshape(s.shape[0], 1)
    vh = np.transpose(vh)
    # print(f"f_mean.shape: {f_mean.shape}")
    # print(f"f_mean: {f_mean}")
    # print(f"u.shape: {u.shape}")
    # print(f"s.shape: {s.shape}")
    # print(f"vh.shape: {vh.shape}")
    # print(np.transpose(u[:, 0].reshape(u[:, 0].shape[0], 1)).shape)

    explained = s[0][0] ** 2 / np.sum(s ** 2)

    # Find where the first component projects the most over original dimensions
    tmp_ = np.max(np.abs(u[:, 0]))
    nmax = np.argmax(u[:, 0])

    # What's the sign of absolute max amplitude along this dimension?
    tmp_ = np.max(np.abs(f[nmax, :]))
    i_omaxx = np.argmax(np.abs(f[nmax, :]))
    sign_omaxx = np.sign(f[nmax, i_omaxx])

    # Sign of maximum in first component time series
    Vmaxx = np.max(np.abs(vh[:, 0]))
    i_Vmaxx = np.argmax(np.abs(vh[:, 0]))
    sign_Vmaxx = np.sign(vh[i_Vmaxx, 0])

    if reconcile_sign:
        f_new = sign_Vmaxx * sign_omaxx * \
                np.matmul(s[0].reshape(s[0].shape[0], 1), np.transpose(vh[:, 0].reshape(vh[:, 0].shape[0], 1)))
        # print(f"f_new.shape: {f_new.shape}")
        f_new = f_new + sign_Vmaxx * sign_omaxx * np.matmul(np.transpose(u[:, 0].reshape(u[:, 0].shape[0], 1)), f_mean)


    else:

        f_new = np.matmul(s[0].reshape(s[0].shape[0], 1), np.transpose(vh[:, 0].reshape(vh[:, 0].shape[0], 1)))
        f_new = f_new + np.matmul(np.transpose(u[:, 0].reshape(u[:, 0].shape[0], 1)), f_mean)

    return f_new


def scouting(data_, mode_):
    # Takes as input a Nx3xT
    """
    Scouting methods. Input must be Nx3xT
    :param data_: time series inside ROI to be aggregated (Nx3xT)
    :param mode_: scouting function
    :return: single time series, representative of the ROI
    """

    valid_modes = ['mean', 'mean_pca', 'pca_sk', 'pca_bst']
    if mode_ not in valid_modes:
        raise ValueError(f"Invalid mode. valid modes are: {valid_modes}")

    if mode_ == 'mean':
        # Mean in time between all channels
        data_ = data_.reshape(data_.shape[0] * data_.shape[1], data_.shape[2])
        f_new_ = np.mean(data_, axis=0)

    if mode_ == 'mean_pca':
        pca = PCA(n_components=1, svd_solver='arpack')
        # Mean between three direction. Input is Nx3xT, output is 3xT
        data_ = np.mean(data_, axis=0)
        data_ = np.transpose(data_)
        data_ = pca.fit_transform(data_)
        f_new_ = np.transpose(data_)[0, :]

    elif mode_ == 'pca_sk':
        # PCA between the Nx3 components
        pca = PCA(n_components=1, svd_solver='arpack')
        data_ = data_.reshape(data_.shape[0] * data_.shape[1], data_.shape[2])
        data_ = np.transpose(data_)
        data_ = pca.fit_transform(data_)
        f_new_ = np.transpose(data_)[0, :]

    elif mode_ == 'pca_bst':
        data_ = data_.reshape(data_.shape[0] * data_.shape[1], data_.shape[2])
        try:
            f_new_ = pca_bst(data_.astype(np.float32))
        except:
            f_new_ = pca_bst(data_)
        f_new_ = f_new_[0, :]

    return f_new_


def select_subjects(path):
    """
    Pick up all the valid subjects path.
    :param path: path to the folder containing the experiment directories
    :return: list of all subject paths
    """
    sbjs = []
    experiments_path = sorted(glob.glob(os.path.join(path, 'experiment*')))
    for ep in experiments_path:
        sbjs.append(sorted(glob.glob(os.path.join(ep, 'sub*')))[1:])
    sbjs = sorted([s for l in sbjs for s in l])
    return sbjs


"""""""""""""""""""""
SCRIPT STARTS HERE
"""""""""""""""""""""
# select sub_paths
root_path = '/Users/federico/University/Magistrale/00.TESI/data_original'
sbj_paths = select_subjects(path=root_path)

# Load forward model
load_path = os.path.join('/Users/federico/University/Magistrale/00.TESI/data_original/experiment0/common_data/'
                         'forward_model/biosemi', 'experiment0-fwd.fif')
fwd = mne.read_forward_solution(load_path)

# Load atlas
# Removal of bad labels
# Creation of df containing label_name of the ROI and numbero of vertices of the ROI
labels_name = 'aparc'
labels = mne.read_labels_from_annot('fsaverage', labels_name)
labels, df = del_invalid_labels(labels_=labels, word_list_=['unknown', '?'])

# Inverse operator parameters
loose = 1
fixed = False

# Apply inverse epochs operator parameters
method = 'eLORETA'
snr = 3.
lambda2 = 1. / snr ** 2
pick_ori = 'vector'
srate = 512

# Scouting function
mode = 'pca_bst'

# Interval
ival = [-1, 2]

# saving path
root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/cortex'
relative_path = mode + '_' + str(ival)[:-1]
save_path = os.path.join(root_path, relative_path)
print(f"relative_path:  {relative_path}")

# Print inversion infos
print("======================")
print("INVERSION INFOS")
print("======================")
print("{:40} {:40}".format("Field", "Value"))
print('------------------------------------------------')
print("{:40} {:40}".format("atlas", labels_name))
print("{:40} {}".format("fixed", fixed))
print("{:40} {}".format("ival", ival))
print("{:40} {}".format("loose", loose))
print("{:40} {:40}".format("method", method))
print("{:40} {:40}".format("mode", mode))
print("{:40} {:40}".format("pick_ori", pick_ori))
print("{:40} {:40}".format("scouting_function", "mean"))
print("{:40} {}".format("srate", srate))

# Automatic crop rule:
# t = -1s --> sample = 0
# t = 0s --> sample = 512
# t = 1s --> sample = 1024
# t = 2s --> sample = 1536
# t = 3s --> sample = 2048
# t = 4s --> sample = 2560

# OR in time
# t = -1s --> t_out = 0s
# t = 0s --> t_out = 1s
# t = 1s --> t_out = 2s
# t = 2s --> t_out = 3s
# t = 3s --> t_out = 4s
# t = 4s --> t_out = 5s

# This is a function: y=(x+1), where x=t and y=t_out


for sbj_f_path in sbj_paths:

    print('=======================================================================================')
    print(sbj_f_path)
    print('=======================================================================================')

    # Data path
    load_path = os.path.join(sbj_f_path, 'data_preproc3', 'processing', 'epoched_-1-4_forward_all')
    data_path = os.path.join(load_path, 'conc_epochs.pkl')

    with open(data_path, 'rb') as f:
        m_dict = pickle.load(f)

    # Loading epochs, info, targets, run_labels
    epochs = m_dict['epochs']
    info = m_dict['info']
    targets = m_dict['targets']
    run_labels = m_dict['run_label']

    # Transforming epochs to epochs array and setting EEG Reference (CAR)
    epochs = mne.EpochsArray(epochs, info)
    epochs = epochs.copy().set_eeg_reference(ref_channels='average', projection=True)
    epochs.apply_proj()

    # Create identity noise cov
    noise_cov = mne.make_ad_hoc_cov(info, std=1)

    # Make inverse operator
    inverse_operator = make_inverse_operator(info=info,
                                             forward=fwd,
                                             noise_cov=noise_cov,
                                             loose=loose,
                                             fixed=fixed,
                                             verbose=False)

    # Allocate array for reconstruction
    tcs = np.zeros(shape=(run_labels.shape[0], len(labels), srate * (np.abs(ival[0]) + np.abs(ival[1]))))

    for i in np.arange(len(epochs)) :
        epoch = epochs[i]
        # Down-sample logic
        if srate != 512:
            epoch = epoch.resample(srate)
        # Crop logic
        if (ival[0] != -1) or (ival[-1] != 4):
            t_crop_start = (ival[0] + 1)
            t_crop_stop = (ival[-1] + 1)
            epoch = epoch.crop(t_crop_start, t_crop_stop, include_tmax=False)

        t = time.time()

        # Dipole activities
        stc = apply_inverse_epochs(epochs=epoch,
                                   inverse_operator=inverse_operator,
                                   lambda2=lambda2,
                                   method=method,
                                   pick_ori=pick_ori,
                                   verbose=False)

        print(f'Extracting {sbj_f_path[-7:]}, run {run_labels[i]}, trial {i % 50 + 1}')

        # Taking the single label
        for j, label in enumerate(labels):
            # print(j, label.name)
            _, idx = label_src_vertno_sel(label, inverse_operator['src'])
            data = stc[0].data[idx, :, :]
            # print(f"data.shape: {data.shape}")
            f_new = scouting(data_=data, mode_=mode)
            # print(f"data.shape: {data.shape}")
            # print(f"data_.shape: {data_.shape}")
            # print(f"f_new.shape: {f_new.shape}")
            tcs[i, j, :] = f_new

        elapsed = time.time() - t
        print(f'Elapsed time for trial #{i % 50 + 1} = {elapsed} seconds')

    info = dict(atlas_fname=labels_name,
                inv_loose=loose,
                inv_fixed=fixed,
                solver_method=method,
                solver_lambda2=lambda2,
                solver_pick_ori=pick_ori)

    epochs_dict = dict(baseline_ival=[-1, 0],
                       epochs=tcs,
                       info=info,
                       ival=ival,
                       roi_info=df,
                       run_labels=run_labels,
                       targets=targets,
                       srate=srate)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    experiment = sbj_f_path.split(sep='/')[-2]
    if experiment[-1] == '0':
        file_name = sbj_f_path[-7:] + '_ses-01_eeg.pkl'
    if experiment[-1] == '1':
        file_name = sbj_f_path[-7:-3] + '1' + sbj_f_path[-2:] + '_ses-01_eeg.pkl'

    with open(os.path.join(save_path, file_name), 'wb') as handle:
        pickle.dump(epochs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
