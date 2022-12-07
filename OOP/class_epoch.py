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


class GrandAverage:
    """
    Container for grand average array.

    Parameters
    __________

    """

    def __init__(self, grand_average_, kind_, divide_by_led_):
        self.grand_average_ = grand_average_
        self.kind_ = kind_
        self.divide_by_led_ = divide_by_led_

    def view_in_time(self):
        if self.kind == 'total' and self.divide_by_led_:
            pass
        pass

class MyEpochs:
    """
    Container for all epochs

    Parameters
    __________
    epochs_: np.array.
        Epoch matrix, dimension is (Nepochs, Nchan, T)
    targets_: np.array.
        Sequence of targets, dimension is (Nepochs,)
    sub_id_ : np.array.
        Array of dimension (Nepochs, ). It contains the sequence of the id of the sub. If the first M epochs are
        relative to sub-002, then the first M element of the sub_id array will be 002.
    ch_names: list.
        List of channel names.
    ival_: list.
        List of two elements. First element is the time before the presentation of the stimulus respect to the
        epoch is selected. Second element is the time after the presentation of the stimulus respect to the epoch
        is selected.
    srate: int
        Data sampling frequency.

    """

    def __init__(self, epochs_, targets_, info_=None, sub_id_=None, ch_names_=None, ival=None, srate=None):
        self.epochs_ = epochs_
        self.targets_ = targets_
        self.info_ = info_
        self.sub_id_ = sub_id_
        self.ch_names_ = ch_names_
        self.ival = ival
        self.srate = srate


    def apply_car(self, inplace=True):
        """
        Apply common average reference to epochs.
        :return:
        """
        print('--------------------------------------------------------------------')
        print("You are applying the Common Average Reference. Changes occur inplace")
        print('--------------------------------------------------------------------')
        epochs_object = mne.EpochsArray(self.epochs_, self.info_)
        mne.set_eeg_reference(epochs_object,
                              ref_channels='average',
                              projection=True)
        epochs_object.apply_proj()
        if inplace:
            self.epochs_ = epochs_object.get_data()
        else:
            return MyEpochs(epochs_=self.epochs_, targets_=self.targets_, info_=self.info_, sub_id_=self.sub_id_,
                            ch_names_=self.ch_names_, ival=self.ival, srate=self.srate)

    def filter_data(self, remove_filtered_baseline=True, baseline_ival=[-1.5, -0.25], inplace=True):
        """
        Filter the epochs in the (Nepochs, Nchannels, T) matrix. By default, filtering occurs inplace.
        Filtering is performed by applying in sequence an elliptic low pass and high pass filter. Pass-band
        is 0.5-10Hz

        Parameters
        __________
        remove_filtered_baseline: bool.
            If true, removes the filtered baseline from data
        :param remove_filtered_baseline: bool
        :param baseline_ival:
        :return:
        """

        if remove_filtered_baseline and (self.srate is None):
            raise Exception("srate must not be None if you want to remove the baseline from filtered data")

        # Low Pass
        wp = 10 / (self.srate / 2)
        ws = 15 / (self.srate / 2)
        gpass = 0.1
        gstop = 40
        ord, wn = ellipord(wp, ws, gpass, gstop)
        b_lp, a_lp = ellip(ord, gpass, gstop, wn, btype='low')

        # High Pass
        wp = 0.5 / (self.srate / 2)
        ws = 0.01 / (self.srate / 2)
        gpass = 0.1
        gstop = 40
        ord, wn = ellipord(wp, ws, gpass, gstop)
        b_hp, a_hp = ellip(ord, gpass, gstop, wn, btype='high')

        n_epochs, n_channels, n_time_sample = self.epochs_.shape
        epochs_filt = np.zeros((n_epochs, n_channels, n_time_sample))
        for j, epoch in enumerate(self.epochs_):
            epoch = filtfilt(b_lp, a_lp, epoch)
            epoch = filtfilt(b_hp, a_hp, epoch)
            if remove_filtered_baseline:
                pre_, post_ = baseline_ival
                bline_duration = int(np.abs(pre_ - post_) * self.srate)
                baseline = np.mean(epoch[:bline_duration, ...], axis=-1)
                baseline = baseline.reshape(n_channels, 1)
                epochs_filt[j, ...] = epoch - baseline
            else:
                epochs_filt[j, ...] = epoch
        if inplace:
            self.epochs_ = epochs_filt
        else:
            return MyEpochs(epochs_=epochs_filt, targets_=self.targets_, info_=self.info_, sub_id_=self.sub_id_,
                            ch_names_=self.ch_names_, ival=self.ival, srate=self.srate)

    def total_grand_average(self, divide_by_led=False):
        """
        Averaging technique between epochs

        Parameters
        __________
        divide_by_led: bool.
            If True, averaging epochs produced by the same stimulus; in this case the method
            returns a 5x60xT matrix. If False, averaging all epochs; the method returns a 60xT matrix.

        Returns
        _______
        array of grand average
        """
        if type(self.targets_) == list:
            self.targets_ = np.array(self.targets_)

        if self.epochs_.shape[0] != self.targets_.shape[0]:
            raise ValueError('epochs and targets must have the same length')
        grand_average_ = []

        if divide_by_led:
            if self.targets_ is None:
                raise ValueError('If divide_by_led is True targets must not be None')
            for u in np.unique(self.targets_):
                grand_average_.append(np.mean(self.epochs_[self.targets_ == u, ...], axis=0))
            grand_average_ = np.array(grand_average_)
        else:
            grand_average_ = np.mean(self.epochs_, axis=0)
            grand_average_ = np.array(grand_average_)
        return GrandAverage(grand_average_, kind_='total', divide_by_led_=divide_by_led)

    def within_subject_average(self, divide_by_led=False):
        """
        Within subject averaging technique between epochs

        Parameters
        __________
        divide_by_led: bool.
            If True, averaging epochs produced by the same stimulus; in this case the method
            returns a (n_sub x n_led x n_chan x T) matrix. If False, averaging all epochs;
            the method returns a (n_sub x n_chan x T) matrix.
        sub_id: np.array.
            sequence of numbers indicating the subject id epochs (001, 001, .... 101..., 112)

        Returns
        _______
        within subject average
        """
        if type(self.targets_) == list:
            self.targets_ = np.array(self.targets_)
        if self.epochs_.shape[0] != self.targets_.shape[0]:
            raise Exception('epochs and targets must have the same length')
        if self.sub_id_ is None:
            raise Exception('sub_id_ must not be None if you want to perform within subject averaging')

        if divide_by_led:
            ws_average = np.zeros(shape=(np.unique(self.sub_id_).shape[0],
                                         np.unique(self.targets_).shape[0],
                                         self.epochs_.shape[1],
                                         self.epochs_.shape[-1]))
            for k, id in enumerate(np.unique(self.sub_id_)):
                sub_targets = self.targets_[self.sub_id_ == id]
                sub_epochs = self.epochs_[self.sub_id_ == id, ...]
                for j, u in enumerate(np.unique(sub_targets)):
                    ws_average[k, j, ...] = np.mean(sub_epochs[u == sub_targets, ...], axis=0)
        else:
            ws_average = np.zeros(shape=(np.unique(self.sub_id_).shape[0],
                                         self.epochs_.shape[1],
                                         self.epochs_.shape[-1]))
            for k, id in enumerate(np.unique(self.sub_id_)):
                ws_average[k, ...] = np.mean(self.epochs_[id == self.sub_id_, ...], axis=0)
        return GrandAverage(ws_average, kind_='ws', divide_by_led_=divide_by_led)


def get_sub_id(path):
    """
    Takes subject id from path. If file path is sub-002_ses-01_eeg.pkl it returns 001.
    :param path: path of epochs.pkl file
    :return: sub_id, int.
    """
    sub_id_ = [item for item in path.split(sep='_') if 'sub' in item][0]
    sub_id_ = [f for f in sub_id_.split(sep='-') if 'sub' not in f][0]
    return int(sub_id_)


"""
SCRIPT STARTS HERE
"""
all_epochs = []
all_targets = []
sub_id = []

root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets'
epochs_path = glob.glob(os.path.join(root_path, 'scalp', '*.pkl'))
epochs_path = sorted(epochs_path)

for epoch_path in epochs_path[11:13]:

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
    sub_id.extend(np.ones(shape=targets.shape[0]).astype(int) * get_sub_id(epoch_path))

all_epochs = np.concatenate(all_epochs)
all_targets = np.concatenate(all_targets)
sub_id = np.array(sub_id)
# CAR
epochs_array = mne.EpochsArray(all_epochs, info)
mne.set_eeg_reference(epochs_array, ref_channels='average', projection=True)
epochs_array.apply_proj()
all_epochs = epochs_array.get_data()
print(f"all_epochs.shape:  {all_epochs.shape}")
print(f"all_targets.shape: {all_targets.shape}")

all_epochs = MyEpochs(all_epochs, all_targets)

