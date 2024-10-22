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

# fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eeglfpemg.pkl')
# fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eegemg_cleaned.pkl')
fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eegemg_cleaned.pkl')
with open(fpath, 'rb') as f:
    erpac_epochs_dataset = pickle.load(f)

fmin = 1;  fmax = 50;  low_fq_range = np.arange(fmin, fmax, 0.5)
fmin = 40; fmax = 202; high_fq_range = [40, 200]#np.arange(fmin, fmax, 1.0)
sfreq = 512.0
times = np.linspace(-1.5, 1.5, 1536)
n_surrogates = 1000
sub_path     = os.path.join(workspace, 'Results', 'erpac_dataset_eegemg_cleaned_circular')
max_workers  = 10  # Set the number of worker processes (you can adjust based on your CPU cores)

def compute_surrogate(i):
    print(f"Perform {key} surrogates no. {i}")
    pha_surr   = pha[..., np.random.permutation(pha.shape[-1])]
    p_surr     = EventRelatedPac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex='wavelet')
    erpac_surr = p_surr.fit(pha_surr, amp, method='circular', n_jobs=-1, verbose=False).squeeze()
    return erpac_surr

if not os.path.exists(sub_path):
    os.mkdir(sub_path)
for key, value in erpac_epochs_dataset.items():
    print(key)
    p     = EventRelatedPac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex='wavelet')
    pha   = np.array(value)[:, :, 0, :]
    amp   = np.array(value)[:, :, 1, :]
    pha   = pha.reshape(pha.shape[0]*pha.shape[1], pha.shape[2])
    amp   = amp.reshape(amp.shape[0]*amp.shape[1], amp.shape[2])
    kw    = dict(keepfilt=False, edges=None, n_jobs=-1)
    pha   = p.filter(sfreq, pha, ftype='phase', **kw)
    amp   = p.filter(sfreq, amp, ftype='amplitude', **kw)
    erpac = p.fit(pha, amp, method='gc', n_jobs=-1, verbose=False).squeeze()
    erpac_dataset          = {}
    erpac_dataset['erpac'] = erpac
    erpac_surrogates       = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(compute_surrogate, i) for i in range(n_surrogates)]

        # Process results as they complete
        for future in as_completed(futures):
            erpac_surr = future.result()
            erpac_surrogates.append(erpac_surr)

    # Store the results in the dataset
    erpac_dataset['surrogates'] = erpac_surrogates

    fpath = os.path.join(sub_path, f'160reso_{key}.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(erpac_dataset, f)