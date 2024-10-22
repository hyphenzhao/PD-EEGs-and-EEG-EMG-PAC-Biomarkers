import mne
import os
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.signal import spectrogram, resample
from scipy.signal import hilbert
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib as mpl
from scipy.interpolate import interp1d
import pickle
import mne_icalabel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tensorpac import EventRelatedPac

root_path = '/Volumes/Workspace/NextCloud/Ruijin/PD EEG Analysis'
workspace = '/Volumes/Workspace/Data/PD EEG Analysis'
index_PD_Gait = pd.read_excel(os.path.join('Record_v2.xlsx'), index_col=1)

# fpath = os.path.join(workspace, 'Results', 'erpac_epochs_lfp_random_sitting.pkl')
fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eegemg_cleaned.pkl')
with open(fpath, 'rb') as f:
    erpac_epochs_dataset = pickle.load(f)

powers_dataset = {}
fmin  = 1 
fmax  = 200
freqs = np.arange(fmin, fmax, 0.5)
sfreq = 512.0
times = np.linspace(-1.5, 1.5, 1536)
n_cycles = freqs / 2.0

for key, value in erpac_epochs_dataset.items():
    print(key, end="")
    epochs = np.array(value)
    # eshape = epochs.shape
    # epochs = epochs.reshape(eshape[0]*eshape[1], eshape[2], eshape[3])
    output = []
    for epoch in epochs:
        output.append(
            mne.time_frequency.tfr_array_morlet(
                epoch, sfreq, freqs, n_cycles=n_cycles, 
                use_fft=True, decim=1, output='avg_power_itc', 
                n_jobs=-1, verbose='INFO'))
    powers_dataset[key] = np.array(output)
    print(powers_dataset[key].shape, powers_dataset[key].dtype)

# for key, value in erpac_epochs_dataset.items():
#     print(key, end="")
#     epochs = np.array(value)
#     eshape = epochs.shape
#     epochs = epochs.reshape(eshape[0]*eshape[1], eshape[2], eshape[3])
#     output = mne.time_frequency.tfr_array_morlet(
#                 epoch, sfreq, freqs, n_cycles=n_cycles, 
#                 use_fft=True, decim=1, output='avg_power_itc', 
#                 n_jobs=-1, verbose='INFO')
#     powers_dataset[key] = np.array(output)
#     print(powers_dataset[key].shape, powers_dataset[key].dtype)
    
fpath = os.path.join(workspace, 'Results', 'erpac_powers_itcs_cleaned.pkl')
with open(fpath, 'wb') as f:
    pickle.dump(powers_dataset, f)