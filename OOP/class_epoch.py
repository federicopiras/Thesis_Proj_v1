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

    def __init__(self, grand_average_, kind_, divide_by_led_, dict_, sub_id):
        self.grand_average_ = grand_average_
        self.kind_ = kind_
        self.divide_by_led_ = divide_by_led_
        self.sub_id = sub_id
        for key, value in dict_.items():
            if key != ('epochs' or 'targets'):
                setattr(self, key, value)

    def get_min_max(self):
        fmin = np.min(self.grand_average_)
        fmax = np.max(self.grand_average_)
        return fmin, fmax

    def get_pre_post(self):
        pre_, post_ = self.ival
        return pre_, post_

    def view_in_time(self):
        fmin, fmax = self.get_min_max()
        fmin = (fmin + 0.25 * fmin) * 1e6
        fmax = (fmax + 0.25 * fmax) * 1e6
        pre_, post_ = self.get_pre_post()
        nsample = int((np.abs(pre) + np.abs(post)) * self.srate)
        time = np.linspace(pre_, post_, nsample)

        if self.kind_ == 'total' and self.divide_by_led_:
            for ch_ in self.ch_names:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                for led in range(self.grand_average_.shape[0]):
                    ax.plot(time, self.grand_average_[led, self.ch_names.index(ch_)] * 1e6,
                            label='LED'+str(led+1), linewidth=1)
                ax.set_title(ch_, fontsize=20)
                ax.legend()
                ax.set_xlim([pre_, post_])
                ax.set_ylim([fmin, fmax])
                ax.set_xlabel('time [s]', fontsize=15)
                ax.set_ylabel('µV', fontsize=15)

                ax.vlines(time[int(self.srate * np.abs(pre_))],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.vlines(time[int(self.srate * np.abs(pre_) + 2)],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.hlines(y=0,
                          xmin=pre_, xmax=post_,
                          linestyles='--', linewidth=2, color='gray', alpha=0.4)

                # Cambio colore ai ticks che mi interessano
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.get_xticklabels()[3].set_color('red')
                ax.get_xticklabels()[7].set_color('red')
                ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid(which='major', linestyle='-', alpha=0.4)
                ax.grid(which='minor', linestyle='--', alpha=0.2)
                fig.show()

        elif self.kind_ == 'total' and not self.divide_by_led_:
            for ch_ in self.ch_names:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.plot(time, self.grand_average_[self.ch_names.index(ch_)] * 1e6,
                        linewidth=1)
                ax.set_title(ch_, fontsize=20)
                ax.set_xlim([pre_, post_])
                ax.set_ylim([fmin, fmax])
                ax.set_xlabel('time [s]', fontsize=15)
                ax.set_ylabel('µV', fontsize=15)

                ax.vlines(time[int(self.srate * np.abs(pre_))],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.vlines(time[int(self.srate * np.abs(pre_) + 2*self.srate)],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.hlines(y=0,
                          xmin=pre_, xmax=post_,
                          linestyles='--', linewidth=2, color='gray', alpha=0.4)

                # Cambio colore ai ticks che mi interessano
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.get_xticklabels()[3].set_color('red')
                ax.get_xticklabels()[7].set_color('red')
                ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid(which='major', linestyle='-', alpha=0.4)
                ax.grid(which='minor', linestyle='--', alpha=0.2)
                fig.show()

        elif self.kind_ == 'ws' and self.divide_by_led_:
            if self.sub_id is None:
                raise Exception('sub id must not be None when performing within sub avg')
            for k, subject_average in enumerate(self.grand_average_):
                fig, axs = plt.subplots(nrows=10, ncols=6, sharex=True, sharey=True, figsize=(20, 10))
                fig.suptitle(self.sub_id[k])
                for ch_, ax in zip(self.ch_names, axs.ravel()):
                    for led in range(subject_average.shape[0]):
                        ax.plot(time, subject_average[led, self.ch_names.index(ch_)] * 1e6,
                                linewidth=1, label='LED'+str(led))
                    ax.set_title(ch_)
                    ax.legend()
                    ax.set_xlim([pre_, post_])
                    ax.set_ylim([fmin, fmax])
                    ax.set_xlabel('time [s]')
                    ax.set_ylabel('µV')

                    ax.vlines(time[int(self.srate * np.abs(pre_))],
                              ymin=fmin, ymax=fmax,
                              linestyles='-', linewidth=1.8, color='k')
                    ax.vlines(time[int(self.srate * np.abs(pre_) + 2*self.srate)],
                              ymin=fmin, ymax=fmax,
                              linestyles='-', linewidth=1.8, color='k')
                    ax.hlines(y=0,
                              xmin=pre_, xmax=post_,
                              linestyles='--', linewidth=2, color='gray', alpha=0.4)

                    # Cambio colore ai ticks che mi interessano
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(MultipleLocator(0.5))
                    ax.get_xticklabels()[3].set_color('red')
                    ax.get_xticklabels()[7].set_color('red')
                    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                    ax.yaxis.set_minor_locator(MultipleLocator(1))
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.grid(which='major', linestyle='-', alpha=0.4)
                    ax.grid(which='minor', linestyle='--', alpha=0.2)
                fig.tight_layout()
                fig.show()

        elif self.kind_ == 'ws' and not self.divide_by_led_:
            if self.sub_id is None:
                raise Exception('sub id must not be None when performing within sub avg')
            for ch_ in self.ch_names:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                for i, subject_average in enumerate(self.grand_average_):
                    ax.plot(time, subject_average[self.ch_names.index(ch_)] * 1e6,
                            linewidth=1, label=str(self.sub_id[i]))
                ax.set_title(ch_, fontsize=20)
                ax.legend()
                ax.set_xlim([pre_, post_])
                ax.set_ylim([fmin, fmax])
                ax.set_xlabel('time [s]', fontsize=15)
                ax.set_ylabel('µV', fontsize=15)

                ax.vlines(time[int(self.srate * np.abs(pre_))],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.vlines(time[int(self.srate * np.abs(pre_) + 2*self.srate)],
                          ymin=fmin, ymax=fmax,
                          linestyles='-', linewidth=1.8, color='k')
                ax.hlines(y=0,
                          xmin=pre_, xmax=post_,
                          linestyles='--', linewidth=2, color='gray', alpha=0.4)

                # Cambio colore ai ticks che mi interessano
                ax = plt.gca()
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.get_xticklabels()[3].set_color('red')
                ax.get_xticklabels()[7].set_color('red')
                ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid(which='major', linestyle='-', alpha=0.4)
                ax.grid(which='minor', linestyle='--', alpha=0.2)
                fig.show()


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

    # def __init__(self, epochs_, targets_, info_, sub_id_=None, ch_names_=None, ival=None, srate=None):
    #     self.epochs_ = epochs_
    #     self.targets_ = targets_
    #     self.info_ = info_
    #     self.sub_id_ = sub_id_
    #     self.ch_names_ = ch_names_
    #     self.ival = ival
    #     self.srate = srate

    def __init__(self, epochs_, targets_, dict_):
        self.epochs_ = epochs_
        self.targets_ = targets_
        self.dict_ = dict_
        self.reference = 'A2'
        for key, value in dict_.items():
            if key not in ['epochs', 'targets']:
                setattr(self, key, value)

    def apply_car(self, inplace=True):
        """
        Apply common average reference to epochs.
        :return:
        """
        print('--------------------------------------------------------------------')
        print("You are applying the Common Average Reference. Changes occur inplace")
        print('--------------------------------------------------------------------')
        epochs_object = mne.EpochsArray(self.epochs_, self.info)
        mne.set_eeg_reference(epochs_object,
                              ref_channels='average',
                              projection=True)
        epochs_object.apply_proj()
        self.reference = 'CAR'
        if inplace:
            self.epochs_ = epochs_object.get_data()
        else:
            return MyEpochs(epochs_=self.epochs_, targets_=self.targets_, dict_=self.dict_)

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
            return MyEpochs(epochs_=epochs_filt, targets_=self.targets_)

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
        return GrandAverage(grand_average_, kind_='total', divide_by_led_=divide_by_led, dict_=self.dict_)

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

        if divide_by_led:
            ws_average = np.zeros(shape=(self.epochs_length.shape[0],
                                         np.unique(self.targets_).shape[0],
                                         self.epochs_.shape[1],
                                         self.epochs_.shape[-1]))
            for k, length in enumerate(self.epochs_length):
                start = int(k*length)
                stop = int((k+1)*length)
                sub_targets = self.targets_[start:stop, ...]
                sub_epochs = self.epochs_[start:stop, ...]
                for j, u in enumerate(np.unique(sub_targets)):
                    ws_average[k, j, ...] = np.mean(sub_epochs[u == sub_targets, ...], axis=0)
        else:
            ws_average = np.zeros(shape=(self.epochs_length.shape[0],
                                         self.epochs_.shape[1],
                                         self.epochs_.shape[-1]))
            for j, length in enumerate(self.epochs_length):
                start = int(j*length)
                stop = int((j+1)*length)
                ws_average[j, ...] = np.mean(self.epochs_[start:stop, ...], axis=0)
        return GrandAverage(ws_average, kind_='ws', divide_by_led_=divide_by_led, dict_=self.dict_, sub_id=self.sub_id)

    def get_sub_id(self, paths):
        """
        Takes subject id from path. If file path is sub-002_ses-01_eeg.pkl it returns 001.
        :param paths: path of epochs.pkl file
        :return: sub_id, int.
        """
        sub_id = []
        for path in paths:
            _ = [item for item in path.split(sep='_') if 'sub' in item][0]
            _ = [f for f in _.split(sep='-') if 'sub' not in f][0]
            sub_id.append(_)
        setattr(self, 'sub_id', sub_id)


"""
SCRIPT STARTS HERE
"""
all_epochs = []
all_targets = []

root_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets'
epochs_path = glob.glob(os.path.join(root_path, 'scalp', '*.pkl'))
epochs_path = sorted(epochs_path)[11:13]
n_epochs = np.zeros(shape=len(epochs_path))

for i, epoch_path in enumerate(epochs_path):

    # Pick up pickle epochs file path:
    with open(epoch_path, 'rb') as f:
        sub_dict = pickle.load(f)

    epochs = sub_dict['epochs']
    info = sub_dict['info']
    targets = sub_dict['targets']
    srate = sub_dict['srate']
    pre, post = sub_dict['ival']
    ch_names = sub_dict['ch_names']
    n_epochs[i] = sub_dict['run_labels'].shape[0]

    all_epochs.append(epochs)
    all_targets.append(targets)

sub_dict['epochs_length'] = n_epochs.astype('int')
all_epochs = np.concatenate(all_epochs)
all_targets = np.concatenate(all_targets)

print(f"all_epochs.shape:  {all_epochs.shape}")
print(f"all_targets.shape: {all_targets.shape}")

all_epochs = MyEpochs(all_epochs, all_targets, sub_dict)

