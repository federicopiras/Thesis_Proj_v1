import os
import pickle
import mne
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Read info from one subject to get the infos needed to plot the electrode on the scalp
load_path = '/Users/federico/University/Magistrale/00.TESI/data_original/experiment0/sub-002/data_preproc3/processing/' \
            'epoched_-1-4_forward_all/conc_epochs.pkl'

with open(load_path, 'rb') as f:
    m_dict = pickle.load(f)

epochs = m_dict['epochs']
info = m_dict['info']

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(info, src=src,
                       eeg=['original', 'projected'], trans=trans,
                       show_axes=True, mri_fiducials=True, dig='fiducials')

# Compute the forward model
fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)

# Save the forward model to folder path
save_path = os.path.join('/Users/federico/University/Magistrale/00.TESI/data_original/experiment0/common_data',
                         'forward_model', 'biosemi', 'experiment0-fwd.fif')
mne.write_forward_solution(save_path, fwd, overwrite=True, verbose=True)

# Try to read the forward model
load_path = os.path.join('/Users/federico/University/Magistrale/00.TESI/data_original/experiment0/common_data',
                         'forward_model', 'biosemi', 'experiment0-fwd.fif')
fwd = mne.read_forward_solution(load_path)
