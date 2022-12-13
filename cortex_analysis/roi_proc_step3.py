"""
Description

Loading the epochs on cortex.
* Plotting grand average for different cortical views (dorsal, caudal...)
* Plotting of some useful ROI as a reference
"""

import os
import numpy as np
import nibabel as nib
from surfer import Brain
import mne
import glob
import pickle
from scipy.signal import ellip, ellipord, filtfilt
import matplotlib.pyplot as plt
import cv2
from mne.source_space import label_src_vertno_sel
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# Insert the dataset path:
folder_path = '/Users/federico/University/Magistrale/00.TESI/data_original/datasets/cortex/mean_[-1,4'
# Take the .pkl with the epochs
file_paths = glob.glob(os.path.join(folder_path, '*.pkl'))
file_paths = sorted(file_paths)

# Grand averaging
all_epochs = []
all_targets = []

for file_path in file_paths[:1]:
    print(file_path)
    # Carico il dizionario con le epoche:
    with open(file_path, 'rb') as f:
        m_dict = pickle.load(f)

    # Carico i campi del dizionario in delle variabili:
    epochs = m_dict['epochs']
    targets = m_dict['targets']
    baseline_ival = m_dict['baseline_ival']
    info = m_dict['info']
    pre_time,  post_time = m_dict['ival']
    roi_info = m_dict['roi_info']
    srate = m_dict['srate']
    roi_names = roi_info['roi_names'].tolist()

    all_epochs.append(epochs)
    all_targets.append(targets)

all_epochs = np.concatenate(all_epochs)
all_targets = np.concatenate(all_targets)

print(f"all_epochs.shape:  {all_epochs.shape}")
print(f"all_targets.shape: {all_targets.shape}")

# Filter parameters
# Low pass
srate = 512
wp = 10 / (srate / 2)
ws = 15 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_lp, a_lp = ellip(ord, gpass, gstop, wn, btype='low')

# High pass
srate = 512
wp = 0.5 / (srate / 2)
ws = 0.01 / (srate / 2)
gpass = 0.1
gstop = 40
ord, wn = ellipord(wp, ws, gpass, gstop)
b_hp, a_hp = ellip(ord, gpass, gstop, wn, btype='high')

# Filtering each single epoch
n_epochs, n_channels, n_time_sample = all_epochs.shape
for j, epoch in enumerate(all_epochs):
    epoch = filtfilt(b_lp, a_lp, epoch)
    epoch = filtfilt(b_hp, a_hp, epoch)
    # baseline = np.mean(epoch[:, :512], axis=-1)
    # baseline = baseline.reshape(n_channels, 1)
    all_epochs[j, :, :] = epoch #- baseline
print(f"epoch_filt.shape: {all_epochs.shape}")

# Grand averaging
n_epochs, n_channels, n_time_sample = all_epochs.shape
n_unique_targets = np.unique(all_targets).shape[0]
grand_average = np.zeros((n_unique_targets, n_channels, n_time_sample))
for j, target in enumerate(np.unique(all_targets)):
    grand_average[j, :, :] = np.mean(all_epochs[all_targets == target, :, :], axis=0)

# Printing GA max and min value
fmin = np.min(grand_average)
fmax = np.max(grand_average)
fmin = fmin - 0.2 * fmin  # - 0.25 * fmin
fmax = fmax - 0.6 * fmax
print(f"MAX grand_average: {fmax}")
print(f"MIN grand_average: {fmin}")

# ---------------------------------------
# Plotting the signal in a virtual cortex
# ---------------------------------------

# Average subject brain and labels
subject_id = "fsaverage"
subjects_dir = mne.datasets.sample.data_path() / 'subjects'
labels_name = 'aparc'
labels = mne.read_labels_from_annot('fsaverage', labels_name)
labels = [label for label in labels if 'unknown' not in label.name]

# For each time instant, make a plot for each label
# Declaring plotting parameters
n_time_sample = grand_average.shape[-1]
print(f"n_time_sample: {n_time_sample}")
nsample = int(n_time_sample / srate * post_time + 1)
samples = np.linspace(0, n_time_sample, nsample)
samples[1:] = samples[1:] - 1
samples = samples.astype(int)
print(samples)

# Define the brain view and the time window within which you average the signal
hemi = 'both'
surf = 'inflated'
views = 'dorsal'
window = 0.25
# Define save path & save image to path
save_path = '/Users/federico/University/Magistrale/00.TESI/data_original/THESIS_IMAGES/cortex/roi_v250'
Brain = mne.viz.get_brain_class()
brain = Brain(subject_id=subject_id,
              hemi=hemi,
              surf=surf,
              subjects_dir=subjects_dir,
              background='white',
              views=views,
              offscreen=True)
print(hemi)
data_points = brain.geo['lh'].bin_curv.shape[0]

# Compute grand average inside the time window
# Plot the grand average in the corresponding ROI
# Save the image
for c, ga in enumerate(grand_average):
    for ss, s in enumerate(samples):

        # Mean of grand average around sample s
        if s == samples[0]:
            start = s
            stop = s + int(window * srate)
        elif s == samples[-1]:
            start = s - int(window * srate)
            stop = s
        else:
            start = s - int(window * 0.5 * srate)
            stop = s + int(window * 0.5 * srate)

        temp_ga = np.mean(ga[:, start:stop], axis=1)

        data_lh = np.zeros(data_points)
        data_rh = np.zeros(data_points)
        for i in range(len(labels)):
            if labels[i].hemi == 'lh':
                data_lh[labels[i].vertices] = temp_ga[i]
            if labels[i].hemi == 'rh':
                data_rh[labels[i].vertices] = temp_ga[i]

        if hemi == 'both':
            brain.add_data(data_lh,
                           fmin=fmin,
                           fmax=fmax,
                           hemi='lh',
                           alpha=1,
                           colormap='jet',
                           colorbar=False)

            brain.add_data(data_rh,
                           fmin=fmin,
                           fmax=fmax,
                           hemi='rh',
                           alpha=1,
                           colormap='jet',
                           colorbar=False)

            brain.add_annotation("aparc")

        elif hemi == 'lh':
            brain.add_data(data_lh,
                           fmin=fmin,
                           fmax=fmax,
                           hemi=hemi,
                           alpha=1,
                           colormap='jet',
                           colorbar=False)

            brain.add_annotation("aparc")

        elif hemi == 'rh':
            brain.add_data(data_rh,
                           fmin=fmin,
                           fmax=fmax,
                           hemi=hemi,
                           alpha=1,
                           colormap='jet',
                           colorbar=False)

            brain.add_annotation("aparc")

        # Define save path & save image to path
        if ss < 10:
            path = os.path.join(save_path, f'activation_led{c}_sample_00{ss}.jpg')
        elif ss >= 10:
            path = os.path.join(save_path, f'activation_led{c}_sample_0{ss}.jpg')
        brain.save_image(path)

        # brain.close()

# -----------------------------------------------------------------
# Load images path  & plot images in the same grand average figure
# -----------------------------------------------------------------
imgs_path = sorted(glob.glob(os.path.join(save_path, 'activation*')))

# Title LOGIC
if views == 'medial' and hemi == 'lh':
    title = 'Cortical Grand Average: Medial View, Left Hemisphere'
elif views == 'medial' and hemi == 'rh':
    title = 'Cortical Grand Average: Medial View, Right Hemisphere'
elif views == 'lateral' and hemi == 'lh':
    title = 'Cortical Grand Average: Lateral View, Left Hemisphere'
elif views == 'lateral' and hemi == 'rh':
    title = 'Cortical Grand Average: Lateral View, Right Hemisphere'
elif views == 'dorsal' and hemi == 'both':
    title = 'Cortical Grand Average: Dorsal View'
elif views == 'caudal' and hemi == 'both':
    title = f'Cortical Grand Average: {views.capitalize()} View'
elif views == 'rostral' and hemi == 'both':
    title = f'Cortical Grand Average: {views.capitalize()} View'

# # Initialize figure
fig, axs = plt.subplots(5, nsample, figsize=(24, 11))
fig.suptitle(title, fontsize=20, fontweight='bold')
# fig, axs = plt.subplots(1, nsample, figsize=(20, 10))
# fig.suptitle(title, fontsize=20, fontweight='bold')

# Define colorbar
colormap = plt.cm.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=colormap)
vmin = fmin
vmax = fmax
sm.set_clim(vmin=vmin, vmax=vmax)

# Create the cortical grand average figure
for img_path, ax in zip(imgs_path, axs.ravel()):
    print(img_path)
    im = plt.imread(img_path)
    os.remove(img_path)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Define x_labels and y_labels
cols = ['{}s'.format(col) for col in np.arange(-1, 4.25, 0.25)]
rows = ['Led{}'.format(row) for row in [1, 2, 3, 4, 5]]

# for ax, col in zip(axs, cols):
#     ax.set_xlabel(col, fontweight='bold', fontsize=15)
#     ax.xaxis.set_label_coords(0.55, -0.7)

# Add seconds to the bottom of the figure (x_label)
for ax, col in zip(axs[-1], cols):
    if col == '0.0s' or col == '2.0s':
        ax.set_xlabel(col, fontweight='bold', fontsize=18, color='red')
        ax.xaxis.set_label_coords(0.55, -0.4)
    else:
        ax.set_xlabel(col, fontweight='bold', fontsize=18)
        ax.xaxis.set_label_coords(0.55, -0.4)
# Add LED id to left of figure (y_labels)
for ax, row in zip(axs[:, 0], rows):
    ax.set_ylabel(row, fontweight='bold', fontsize=25, rotation=0)
    ax.yaxis.set_label_coords(-0.8, 0.35)

# Adding a big colorbar to the side of the figure
cbar_ax = fig.add_axes([0.94, 0.1, 0.01, 0.8])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.ax.tick_params(labelsize=22)
ticks = cb.get_ticks()
cb.ax.set_yticklabels(np.round(ticks*1e12, 2))
cb.ax.set_title('pA*m', fontsize=25)

# Adjusting subplot distances and location inside the figure
fig.subplots_adjust(left=0.07,
                    bottom=0.1,
                    right=0.90,
                    top=0.95,
                    wspace=0.2,
                    hspace=0.01)
fig.show()
path = os.path.join(save_path, 'CORTICAL_GA-' + views + '_' + hemi + '.pdf')
fig.savefig(path)


# ------------------------------------------------------------------------
# Plot of reference ROIs
# ------------------------------------------------------------------------
# Picking up labels
subjects_dir = mne.datasets.sample.data_path() / 'subjects'
surf = 'inflated'
subject_id = 'fsaverage'
labels_name = 'aparc'
labels = mne.read_labels_from_annot(subject_id, labels_name)
labels = [label for label in labels if 'unknown' not in label.name]
# List of ROIs of interest names
labels_of_interest = ['cuneus',
                      'lateraloccipital',
                      'paracentral',
                      'postcentral',
                      'precentral',
                      'precuneus',
                      'superiorfrontal',
                      'superiorparietal']
# List of views
views = ['dorsal', 'caudal', 'medial', 'lateral', 'rostral']

# Save_path
save_path = '/Users/federico/University/Magistrale/00.TESI/data_original/THESIS_IMAGES/cortex/reference_v1'

already_saved = False
if not already_saved:
    for label in labels_of_interest:
        for view in views:
            # Declaring Brain Class
            Brain = mne.viz.get_brain_class()
            label_to_plot = [l for l in labels if label in l.name]
            print(label_to_plot)
            if view == 'dorsal' or view == 'caudal' or view == 'rostral':
                # Selecting Label object based on labels_of_interest names
                hemi = 'both'
                brain = Brain(subject_id=subject_id,
                              hemi=hemi,
                              surf=surf,
                              subjects_dir=subjects_dir,
                              background='white',
                              views=view,
                              cortex='limegreen',
                              alpha=1)
                brain.add_label(label=label_to_plot[0], hemi='lh', color='m')
                brain.add_label(label=label_to_plot[1], hemi='rh', color='m')
                brain.add_annotation('aparc', color='m')
                path = os.path.join(save_path, label + '_' + view + '.jpg')
                brain.save_image(path)
                brain.close()
            else:
                for hemi in ['lh', 'rh']:
                    if hemi == 'lh':
                        brain = Brain(subject_id=subject_id,
                                      hemi=hemi,
                                      surf=surf,
                                      subjects_dir=subjects_dir,
                                      background='white',
                                      views=view,
                                      cortex='limegreen',
                                      alpha=1)
                        brain.add_label(label=label_to_plot[0], hemi=hemi, color='m')
                        brain.add_annotation('aparc', color='m')
                        path = os.path.join(save_path, label + '_' + view + '_' + hemi + '.jpg')
                        brain.save_image(path)
                        brain.close()
                    if hemi == 'rh':
                        brain = Brain(subject_id=subject_id,
                                      hemi=hemi,
                                      surf=surf,
                                      subjects_dir=subjects_dir,
                                      background='white',
                                      views=view,
                                      cortex='limegreen',
                                      alpha=1)
                        brain.add_label(label=label_to_plot[1], hemi=hemi, color='m')
                        brain.add_annotation('aparc', color='m')
                        path = os.path.join(save_path, label + '_' + view + '_' + hemi + '.jpg')
                        brain.save_image(path)
                        brain.close()

load_path = save_path
imgs_path = sorted(glob.glob(os.path.join(save_path, '*.jpg')))
col_names = ['Caudal', 'Dorsal', 'Lateral LH', 'Lateral RH', 'Medial LH', 'Medial RH', 'Rostral']

fig, axs = plt.subplots(nrows=len(labels_of_interest), ncols=len(col_names), figsize=(20, 10))
for img, ax in zip(imgs_path, axs.ravel()):
    im = plt.imread(img)
    os.remove(img)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
for ax, col in zip(axs[0], col_names):
    ax.set_title(col, fontweight='bold', fontsize=15, pad=20)
for ax, label in zip(axs[:, 0], sorted(labels_of_interest)):
    ax.set_ylabel(label.capitalize(), fontweight='bold', fontsize=15, rotation=0)
    ax.yaxis.set_label_coords(-1.4, 0.35)
fig.show()
path = os.path.join(save_path, 'reference_imagev1.pdf')
fig.savefig(path)