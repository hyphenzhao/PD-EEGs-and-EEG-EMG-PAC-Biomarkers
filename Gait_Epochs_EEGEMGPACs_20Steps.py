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

bandwidth = {
    'Delta': [1.0, 4.0],'Theta': [4.0, 8.0],'Alpha': [8.0, 12.0],
    'Low-Beta': [12.0, 20.0],'High-Beta': [20.0, 30.0],
    'Low-Gamma': [30.0, 50.0],'High-Gamma': [50.0, 80.0],
    'HFO': [80.0, 120.0],'Beta': [12.0, 30.0],'Gamma': [30.0, 80.0],
    'EMG': [100.0, 200.0],'Low-EMG': [100.0, 150.0],'High-EMG': [150.0, 200.0],
    'Petra-Beta': [12.0, 45.0],'Petra-Gamma': [45.0, 90.0],
    'Delta-Theta': [1,0, 8.0],'Delta-Alpha': [1.0, 12.0],'Theta-Alpha': [4.0, 12.0],
    'Beta-Gamma': [12.0,80.0], 'Petra-Beta-Gamma': [12.0, 90.0],
    'Beta-One': [12.0, 15.0], 'Beta-Two': [15.0, 20.0], 'Beta-Three': [18.0, 40.0]
}
selected_bands = ['HFO', 'High-Gamma', 'Low-Gamma', 'High-Beta', 'Low-Beta']

root_path = '/Volumes/Workspace/NextCloud/Ruijin/PD EEG Analysis'
workspace = '/Volumes/Workspace/Data/PD EEG Analysis'
index_PD_Gait = pd.read_excel(os.path.join('Record_v1.xlsx'), index_col=1)

from pactools import Comodulogram
import tensorpac 

import pickle
fpath = os.path.join(workspace, 'cleaned_epoched_dataset_20240605.pkl')
print(f"Loading data from file {fpath}.")
with open(fpath, 'rb') as f:
    cleaned_epochs_dataset = pickle.load(f)

for eeg_chs in ['C2C1', 'Cz']:
    import mne_connectivity
    import pickle
    import sys
    min_points = 2000000
    fmin = 1; fmax = 50; freqs = np.arange(fmin, fmax, 0.5)
    mi_all = {}
    mi_patient_list = {}
    for key, value in cleaned_epochs_dataset.items():
        patient_med, event, phase = key.split(' ')
        if 'C2' not in value['channels']: continue
        if event == '50': 
            strike_title = "Left Heel Strike"
            # emg_ch       = 'LEMG'
            emg_ch       = 'C2'
            eeg_ch       = 'C2'
        else: 
            strike_title = "Right Heel Strike"
            emg_ch       = 'REMG'
            emg_ch       = 'C1'
            eeg_ch       = 'C1'
        if eeg_chs == 'Cz': eeg_ch = 'Cz'
        patient      = patient_med.split('_')[0] + '_' + patient_med.split('_')[1]
        med_type     = patient_med.split('_')[2]
        conn_key     = f"{med_type} {phase} {strike_title}"
        ch_names     = value['channels']
        target_dura  = value['target_times'][-1] - value['target_times'][0]
        target_len   = len(value['target_times'])
        target_times = value['target_times']
        # mod_index    = []
        print(f"Processing {key} modulation index")
        epochs_pick  = None

        for cnt, single_epoch in enumerate(value['data']):
            # epochs_pick   = np.array(single_epoch)
            if epochs_pick is None: epochs_pick = np.array(single_epoch)
            else: epochs_pick = np.concatenate((epochs_pick, np.array(single_epoch)), axis=-1)
        if epochs_pick.shape[1] < min_points: min_points = epochs_pick.shape[1]
        mi_pac_model  = tensorpac.Pac(idpac=(6, 0, 0), 
                                     # f_pha=freqs, f_amp=np.arange(105.0, 145.0, 1.0), 
                                      f_pha=freqs, f_amp=np.arange(40.0, 202.0, 1.0), 
                                     dcomplex='wavelet', verbose=False)
        mi_pac_result = mi_pac_model.filterfit(value['sfreq'], 
                                               epochs_pick[ch_names.index(eeg_ch)],
                                               epochs_pick[ch_names.index(emg_ch)])
        # mod_index.append(mi_pac_result.squeeze())
        # mod_index = np.array(mod_index)
        mod_index = mi_pac_result.squeeze()
        print(mod_index.shape)
        #     shuffled_max_pac.append(mod_index)
        # shuffled_max_pac = np.array(shuffled_max_pac).mean(axis=0)
        # print()
        # print(shuffled_max_pac.shape)
        if conn_key in list(mi_all.keys()): 
            mi_all[conn_key].append(mod_index)
            mi_patient_list[conn_key].append(patient)
        else: 
            mi_all[conn_key] = [mod_index]
            mi_patient_list[conn_key] = [patient]
    for key, value in mi_all.items():
        mi_all[key] = np.array(value)

    fpath = os.path.join(workspace, 'Results', f'eegeeg_pacs_20240617_{eeg_chs}.pkl')
    with open(fpath, 'wb') as f:
        mi_object = {'mi_all': mi_all, 'mi_patient_list': mi_patient_list,
                     'low_fq_range':freqs, 'high_fq_range': np.arange(40.0, 202.0, 1.0)}
        pickle.dump(mi_object, f)
# print(min_points)