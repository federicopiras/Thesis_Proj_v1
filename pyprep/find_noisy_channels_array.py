"""finds bad channels."""
from copy import copy
import numpy as np
from mne.utils import check_random_state
from scipy import signal
from pyprep.ransac import find_bad_by_ransac
from pyprep.removeTrend import removeTrend
from pyprep.utils import _filter_design


class NoisyChannels:

    def __init__(self, data, sample_rate, ch_names, ch_pos, do_detrend=True, do_bandpass_filter=False, random_state=None, matlab_strict=False):
        self.data = np.copy(data)
        self.sample_rate = sample_rate
        if do_detrend:
            self.data = removeTrend(
                self.data, self.sample_rate, matlab_strict=matlab_strict
            )
        self.matlab_strict = matlab_strict

        # random_state
        self.random_state = check_random_state(random_state)

        # The identified bad channels
        self.bad_by_ransac = []

        # Get original EEG channel names, channel count & samples
        ch_names = np.asarray(ch_names)
        self.ch_names_original = ch_names
        self.n_chans_original = len(ch_names)
        self.ch_pos_original = ch_pos
        self.n_samples = data.shape[1]
        self.do_bandpass_filter = do_bandpass_filter

    def _get_filtered_data(self):
        """Apply a [1 Hz - 50 Hz] bandpass filter to the EEG signal.

        Only applied if the sample rate is above 100 Hz to avoid violating the
        Nyquist theorem.

        """
        if self.sample_rate <= 100:
            return self.data.copy()

        bandpass_filter = _filter_design(
            N_order=100,
            amp=np.array([1, 1, 0, 0]),
            freq=np.array([0, 90 / self.sample_rate, 100 / self.sample_rate, 1]),
        )
        data_filt = np.zeros_like(self.data)
        for i in range(data_filt.shape[0]):
            data_filt[i, :] = signal.filtfilt(bandpass_filter, 1, self.data[i, :])

        return data_filt

    def find_bad_by_ransac(
        self,
        n_samples=50,
        sample_prop=0.25,
        corr_thresh=0.75,
        frac_bad=0.4,
        corr_window_secs=5.0,
        channel_wise=False,
        max_chunk_size=None,
    ):
        """Detect channels that are predicted poorly by other channels.

        This method uses a random sample consensus approach (RANSAC, see [1]_,
        and a short discussion in [2]_) to try and predict what the signal should
        be for each channel based on the signals and spatial locations of other
        currently-good channels. RANSAC correlations are calculated by splitting
        the recording into non-overlapping windows of time (default: 5 seconds)
        and correlating each channel's RANSAC-predicted signal with its actual
        signal within each window.

        A RANSAC window is considered "bad" for a channel if its predicted signal
        vs. actual signal correlation falls below the given correlation threshold
        (default: ``0.75``). A channel is considered "bad-by-RANSAC" if its fraction
        of bad RANSAC windows is above the given threshold (default: ``0.4``).

        Due to its random sampling component, the channels identified as
        "bad-by-RANSAC" may vary slightly between calls of this method.
        Additionally, bad channels may vary between different montages given that
        RANSAC's signal predictions are based on the spatial coordinates of each
        electrode.

        This method is a wrapper for the :func:`~ransac.find_bad_by_ransac`
        function.

        .. warning:: For optimal performance, RANSAC requires that channels bad by
                     deviation, correlation, and/or dropout have already been
                     flagged. Otherwise RANSAC will attempt to use those channels
                     when making signal predictions, decreasing accuracy and thus
                     increasing the likelihood of false positives.

        Parameters
        ----------
        n_samples : int, optional
            Number of random channel samples to use for RANSAC. Defaults
            to ``50``.
        sample_prop : float, optional
            Proportion of total channels to use for signal prediction per RANSAC
            sample. This needs to be in the range [0, 1], where 0 would mean no
            channels would be used and 1 would mean all channels would be used
            (neither of which would be useful values). Defaults to ``0.25``
            (e.g., 16 channels per sample for a 64-channel dataset).
        corr_thresh : float, optional
            The minimum predicted vs. actual signal correlation for a channel to
            be considered good within a given RANSAC window. Defaults
            to ``0.75``.
        frac_bad : float, optional
            The minimum fraction of bad (i.e., below-threshold) RANSAC windows
            for a channel to be considered bad-by-RANSAC. Defaults to ``0.4``.
        corr_window_secs : float, optional
            The duration (in seconds) of each RANSAC correlation window. Defaults
            to 5 seconds.
        channel_wise : bool, optional
            Whether RANSAC should predict signals for chunks of channels over the
            entire signal length ("channel-wise RANSAC", see `max_chunk_size`
            parameter). If ``False``, RANSAC will instead predict signals for all
            channels at once but over a number of smaller time windows instead of
            over the entire signal length ("window-wise RANSAC"). Channel-wise
            RANSAC generally has higher RAM demands than window-wise RANSAC
            (especially if `max_chunk_size` is ``None``), but can be faster on
            systems with lots of RAM to spare. Defaults to ``False``.
        max_chunk_size : {int, None}, optional
            The maximum number of channels to predict at once during
            channel-wise RANSAC. If ``None``, RANSAC will use the largest chunk
            size that will fit into the available RAM, which may slow down
            other programs on the host system. If using window-wise RANSAC
            (the default), this parameter has no effect. Defaults to ``None``.

        References
        ----------
        .. [1] Fischler, M.A., Bolles, R.C. (1981). Random sample consensus: A
            Paradigm for Model Fitting with Applications to Image Analysis and
            Automated Cartography. Communications of the ACM, 24, 381-395
        .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
            (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
            Data. NeuroImage, 159, 417-429

        """
        if self.do_bandpass_filter:
            self.data_filterd = self._get_filtered_data()
        else:
            self.data_filterd = self.data

        rng = copy(self.random_state) if self.matlab_strict else self.random_state
        bad_by_ransac, ch_correlations = find_bad_by_ransac(
            self.data_filterd,
            self.sample_rate,
            self.ch_names_original,
            self.ch_pos_original,
            [],#exlude no channels
            n_samples,
            sample_prop,
            corr_thresh,
            frac_bad,
            corr_window_secs,
            channel_wise,
            max_chunk_size,
            rng,
            self.matlab_strict,
        )
        # bad idx by ransac, channel correlations, bad window fractions
        return bad_by_ransac, ch_correlations, np.mean(ch_correlations < corr_thresh, axis=0)
