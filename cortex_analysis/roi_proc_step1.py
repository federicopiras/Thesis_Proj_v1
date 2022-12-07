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
import scipy
import os
import glob
import pandas as pd
import pickle
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.source_space import label_src_vertno_sel
import time
from sklearn.decomposition import PCA


def create_path(path):
    """
    A very simple function to create a non existing path
    :param path: path to create
    :return: nothing
    """
    if not os.path.exists(path):
        os.makedirs(path)


def del_invalid_labels(labels_):
    """
    Removes unwanted labels and create df with ROI label name and ROI number of vertices
    :param labels_: list of Labels.
    :return labels_: list of Labels without unwanted labels
    :return df: DataFrame containing Label names and Label number of vertices
    """
    word_list_ = ['unknown', '?']
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


def print_inversion_infos(labels_name_, fixed_, ival_, loose_, method_, mode_, pick_ori_, srate_):
    """
    Prints the inversion info in console

    Parameters
    __________
    labels_name_: str.
        string of the name of the ATLAS.
    fixed_ : bool.
        Use fixed source orientations normal to the cortical mantle.
    ival_: list.
        First element is time before stimulus, second element is time after stimulus
    loose_: float.
        Value that weights the source variances of the dipole components that are parallel (tangential)
        to the cortical surface.
    method_: str.
        Use minimum norm, dSPM, sLORETA, or eLORETA.
    mode_: str.
        Scouting function to aggregate multiple time series in a ROI in a single time series
    pick_ori_: str.
    srate_: int.
        Sampling frequency of the data.
    """
    # Print inversion infos
    print("======================")
    print("INVERSION INFOS")
    print("======================")
    print("{:40} {:40}".format("Field", "Value"))
    print('------------------------------------------------')
    print("{:40} {:40}".format("atlas", labels_name_))
    print("{:40} {}".format("fixed", fixed_))
    print("{:40} {}".format("ival", ival_))
    print("{:40} {}".format("loose", loose_))
    print("{:40} {:40}".format("method", method_))
    print("{:40} {:40}".format("mode", mode_))
    print("{:40} {:40}".format("pick_ori", str(pick_ori_)))
    print("{:40} {}".format("srate", srate_))


def scouting(data_, mode_, src_=None, label_=None):
    """
    Scouting methods. Input can be Nx3xT if dipoles are modelled as "free" or NxT if dipoles are modelled as normal
    to the cortex

    Parameters
    __________
    data_: np.array.
        The np.array containing the time series inside a ROI. Could be (Nvoxel, 3, T) or (Nvoxel, T)
    mode_: 'mean' | 'mean_pca' | 'pca_sk' | 'pca_bst' | 'mean_flip' | 'weight_mean'
        Scouting function.
        Options:
        * mean
            Returns a single time series, that is the mean of all the time series in a ROI.
        * mean_pca
            First compute the mean between the 3 directions --> (3, T). At this point, apply PCA to
            aggregate the 3 series in a single time series. Data must be 3D.
        * pca_sk
            Apply PCA among all the time series in the ROI. Returns a single series representative of the ROI.
            Data can be both 2D or 3D.
        * pca_bst
            Self implemented PCA (it implements the flip_sign too). Apply PCA among all the time series in the ROI.
            Returns a single series representative of the ROI. Data can be both 2D or 3D.
        * mean_flip
            First compute the main direction of the dipoles in the ROI. Then fips the sign of all the dipole series
            which direction is opposite to the main direction. Lastly, apply the mean between all series.
            src_ and label_ must not be None if you choose this method.
        * weight_mean
            Weighted mean of the time series. First compute the norm of the data along the time direction, that is
            here considered as the weight of the time series. Multiply each time series for its norm. Sum all the
            weighted time series and divide by the sum of the norm.
    src_: SourceSpaces.
        The source space over which the label is defined. If mode = 'mean_flip' must not be None.
    label_: Label.
        If mode = 'mean_flip' must not be None.

    Returns
    _______
    f_new, array.
        Single time series representative of the ROI
    """

    valid_modes = ['mean', 'mean_pca', 'pca_sk', 'pca_bst', 'mean_flip', 'weight_mean']
    if mode_ not in valid_modes:
        raise ValueError(f"Invalid mode. valid modes are: {valid_modes}")

    if (data_.ndim == 3) and ((mode_ != 'mean_pca') and (mode_ != 'weight_mean')):
        data_ = data_.reshape(data_.shape[0] * data_.shape[1], data_.shape[2])

    if mode_ == 'mean':
        # Mean in time between all channels
        f_new_ = np.mean(data_, axis=0)

    elif mode_ == 'mean_flip':
        if (src_ is None) or (label_ is None):
            raise Exception('src and label must not be none if mode = mean_flip')

        flip_mask = sign_flip(src_=src_, label_=label_)
        f_new_ = data_*(flip_mask.reshape(flip_mask.shape[0], 1))
        f_new_ = np.mean(f_new_, axis=0)
        return f_new_

    if mode_ == 'mean_pca':
        if data_.ndim == 2:
            raise ValueError(f'data must be (Nvox, 3, T). Your data dimension is: {data_.shape}')
        pca = PCA(n_components=1, svd_solver='arpack')
        # Mean between three direction. Input is Nx3xT, output is 3xT
        data_ = np.mean(data_, axis=0)
        data_ = np.transpose(data_)
        data_ = pca.fit_transform(data_)
        f_new_ = np.transpose(data_)[0, :]

    elif mode_ == 'pca_sk':
        # PCA between the Nx3 components
        pca = PCA(n_components=1, svd_solver='arpack')
        data_ = np.transpose(data_)
        data_ = pca.fit_transform(data_)
        f_new_ = np.transpose(data_)[0, :]

    elif mode_ == 'pca_bst':
        try:
            f_new_ = pca_bst(data_.astype(np.float32))
        except:
            f_new_ = pca_bst(data_)
        f_new_ = f_new_[0, :]

    elif mode == 'weight_mean':
        if data_.ndim == 3:
            #raise ValueError(f'data must be (Nvox, 3, T). Your data dimension is: {data_.shape}')
            n_vox, n_dim, n_time = data_.shape
            weights = np.linalg.norm(data_, axis=-1).reshape(n_vox, n_dim, 1)
            w_data_ = data_ * weights
            f_new_ = np.sum(w_data_, axis=(0, 1)) / np.sum(weights)
        if data_.ndim == 2:
            n_vox, n_time = data_.shape
            weights = np.linalg.norm(data_, axis=-1).reshape(n_vox, 1)
            w_data_ = data_ * weights
            f_new_ = np.sum(w_data_, axis=0) / np.sum(weights)


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


def sign_flip(src_, label_):
    """
    Flips the sign of the time series of the dipoles which direction is opposed to the main direction
    :param src_: SourceSpaces. The source space over which the label is defined.
    :param label_: Label.
    :return: flip_mask, array. Sign filp vector
    """

    lh_vertno = src_[0]['vertno']
    rh_vertno = src_[0]['vertno']

    ori = list()
    if label_.hemi == 'lh':
        vertices = label.vertices
        vertno_sel = np.intersect1d(lh_vertno, vertices)
        ori.append(src_[0]['nn'][vertno_sel])
    if label_.hemi == 'rh':
        vertices = label.vertices
        vertno_sel = np.intersect1d(rh_vertno, vertices)
        ori.append(src_[1]['nn'][vertno_sel])

    # Here you have the (Nvertex, 3) normal versors in a ROI
    ori = np.concatenate(ori, axis=0)

    # Here you have the principal orientations of the dipoles
    u, _, _ = scipy.linalg.svd(ori, full_matrices=False)
    flip_mask = np.sign(u[:, 0])

    # Now you have to flip all the dipoles activity which direction is opposed to the main direction.
    if (np.count_nonzero(flip_mask > 0)) < (np.count_nonzero(flip_mask < 0)):
        flip_mask = - flip_mask
    #print(f"Flipped the sign of {np.count_nonzero(flip_mask == -1)} sources.")
    return flip_mask


"""""""""""""""""""""
SCRIPT STARTS HERE
"""""""""""""""""""""
# select epoch paths
root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/scalp1'
epoch_paths = glob.glob(os.path.join(root_path, '*.pkl'))
epoch_paths = sorted(epoch_paths)

# Load forward model
load_path = os.path.join('/Users/federico/University/Magistrale/00.TESI/data_original/experiment0/common_data/'
                         'forward_model/biosemi', 'experiment0-fwd.fif')
fwd = mne.read_forward_solution(load_path)

# Load atlas
# Removal of bad labels
# Creation of df containing label_name of the ROI and numbero of vertices of the ROI
labels_name = 'aparc'
labels = mne.read_labels_from_annot('fsaverage', labels_name)
labels, df = del_invalid_labels(labels_=labels)

# Inverse operator parameters
loose = 1  # 0
fixed = False  # True

# Apply inverse epochs operator parameters
method = 'eLORETA'
snr = 3.
lambda2 = 1. / snr ** 2
pick_ori = 'vector'  # 'normal',  # None
srate = 512

# Scouting function
mode = 'weight_mean'  # mean, pca, mean_pca, pca_bst

# Interval
ival = [-1.5, 4]

# saving path
root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/cortex'
relative_path = mode + '_' + str(ival)[:-1]
save_path = os.path.join(root_path, 'normal_' + relative_path)
print(f"relative_path:  {relative_path}")

# PRINT INVERSION INFOS
print_inversion_infos(labels_name_=labels_name,
                      fixed_=fixed,
                      ival_=ival,
                      loose_=loose,
                      method_=method,
                      mode_=mode,
                      pick_ori_=pick_ori,
                      srate_=srate)

for epoch_path in epoch_paths:

    print('=======================================================================================')
    print(epoch_path.split(sep='/')[-1].upper())
    print('=======================================================================================')

    with open(epoch_path, 'rb') as f:
        m_dict = pickle.load(f)

    # Loading epochs, info, targets, run_labels
    epochs = m_dict['epochs']
    info = m_dict['info']
    targets = m_dict['targets']
    run_labels = m_dict['run_labels']
    pre, post = m_dict['ival']
    bline_ival = m_dict['baseline_ival']

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
    tcs = np.zeros(shape=(run_labels.shape[0], len(labels), int(srate * (np.abs(ival[0]) + np.abs(ival[1])))))

    for i in np.arange(len(epochs)):
        epoch = epochs[i]
        # Down-sample logic
        if srate != 512:
            epoch = epoch.resample(srate)
        # Crop logic
        if (ival[0] != pre) or (ival[-1] != post):
            t_crop_start = ival[0] + np.abs(pre)
            t_crop_stop = ival[-1] + np.abs(pre)
            epoch = epoch.crop(t_crop_start, t_crop_stop, include_tmax=False)

        t = time.time()

        # Dipole activities
        stc = apply_inverse_epochs(epochs=epoch,
                                   inverse_operator=inverse_operator,
                                   lambda2=lambda2,
                                   method=method,
                                   pick_ori=pick_ori,
                                   verbose=False)

        sub_id = epoch_path.split(sep='/')[-1].split(sep='_')[0]
        print(f'Extracting {sub_id}, run {run_labels[i]}, trial {i % 50 + 1}')

        # Taking the single label
        for j, label in enumerate(labels):
            # Uncomment if you want to use self declared scouting methods
            _, idx = label_src_vertno_sel(label, inverse_operator['src'])
            data = stc[0].data[idx, ...]
            f_new = scouting(data_=data, mode_=mode)

            # # MNE scouting method
            # f_new = mne.extract_label_time_course(stcs=stc,
            #                                       labels=label,
            #                                       src=inverse_operator['src'],
            #                                       mode=mode,
            #                                       verbose=False)[0][0]
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

    create_path(save_path)
    file_name = epoch_path.split(sep='/')[-1]

    with open(os.path.join(save_path, file_name), 'wb') as handle:
        pickle.dump(epochs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
