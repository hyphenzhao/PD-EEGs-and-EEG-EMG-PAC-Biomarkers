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
index_PD_Gait = pd.read_excel(os.path.join('Record_v1.xlsx'), index_col=1)

# Load the dataset
# fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eegeeg.pkl')
fpath = os.path.join(workspace, 'Results', 'erpac_epochs_dataset_eegemg_cleaned.pkl')
with open(fpath, 'rb') as f:
    erpac_epochs_dataset = pickle.load(f)

erpac_dataset = {}
fmin = 1;  fmax = 50;  low_fq_range  = np.arange(fmin, fmax, 0.5)
fmin = 40; fmax = 202; high_fq_range = [40, 200]
sfreq = 512.0
times = np.linspace(-1.5, 1.5, 1536)
tsels = np.where((times >= -0.5) & (times <= 0.5))[0]

num_permutations = 1000  # Set the number of permutations
num_file_segs    = 5
max_threads      = 10  # Set the maximum number of threads
max_processes    = 20
def perform_permutation(i, combined_data, len_cona, sfreq, low_fq_range, high_fq_range):
    print(f"Permutation test on {i+1} ...")
    np.random.shuffle(combined_data)
    perm_cona = combined_data[:len_cona]
    perm_conb = combined_data[len_cona:]
    
    pa = EventRelatedPac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex='wavelet', verbose=False)
    pb = EventRelatedPac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex='wavelet', verbose=False)
    
    pha_cona = perm_cona[:, :, 0, :]
    amp_cona = perm_cona[:, :, 1, :]
    pha_conb = perm_conb[:, :, 0, :]
    amp_conb = perm_conb[:, :, 1, :]
    
    pha_cona = pha_cona.reshape(pha_cona.shape[0]*pha_cona.shape[1], pha_cona.shape[2])
    amp_cona = amp_cona.reshape(amp_cona.shape[0]*amp_cona.shape[1], amp_cona.shape[2])
    pha_conb = pha_conb.reshape(pha_conb.shape[0]*pha_conb.shape[1], pha_conb.shape[2])
    amp_conb = amp_conb.reshape(amp_conb.shape[0]*amp_conb.shape[1], amp_conb.shape[2])
    
    erpac_cona = pa.filterfit(sfreq, pha_cona, amp_cona, method='circular', n_jobs=-1, verbose=True).squeeze()
    erpac_conb = pb.filterfit(sfreq, pha_conb, amp_conb, method='circular', n_jobs=-1, verbose=True).squeeze()
    
    print(f"Permutation {i+1} completed.")
    return i, erpac_cona, erpac_conb
for seg_no in range(num_file_segs):
    for foot in ['50', '60']:
        for ch in ['Cz', 'C2', 'C1']:
            cona_key = f"Med-Off Begin {foot} {ch}"
            conb_key = f"Med-On Begin {foot} {ch}"

            if cona_key not in erpac_epochs_dataset or conb_key not in erpac_epochs_dataset:
                continue

            if cona_key not in erpac_dataset:
                erpac_dataset[cona_key] = []
            if conb_key not in erpac_dataset:
                erpac_dataset[conb_key] = []

            print(f"Permutation between {cona_key} vs. {conb_key}")

            data_cona = np.array(erpac_epochs_dataset[cona_key])
            data_conb = np.array(erpac_epochs_dataset[conb_key])
            combined_data = np.concatenate((data_cona, data_conb), axis=0)
            len_cona = len(data_cona)

            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = [executor.submit(perform_permutation, i, combined_data.copy(), len_cona, sfreq, low_fq_range, high_fq_range) for i in range(num_permutations//num_file_segs)]

                for future in as_completed(futures):
                    i, erpac_cona, erpac_conb = future.result()
                    erpac_dataset[cona_key].append(erpac_cona)
                    erpac_dataset[conb_key].append(erpac_conb)

    print("All permutations completed.")

    # fpath = os.path.join(workspace, 'Results', f'erpac_dataset_20240813_160reso_eegeeg_perm1000-{seg_no}.pkl')
    fpath = os.path.join(workspace, 'Results', f'erpac_dataset_eegemg_cleaned_perm1000_circular-{seg_no}.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(erpac_dataset, f)