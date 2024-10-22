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
plt.close('all')
states = ['Rest Time', 'Stand Time', 'Time Slices']
medication = ['Med-On', 'Med-Off']
gc_pacs = {}; gc_pacs_patient_list = {}
target_chs = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O2', 'C1', 'C2', 'O1']
# target_chs = ['C3', 'Cz', 'C4', 'C1', 'C2']
pacs_path = os.path.join(workspace, 'Results', 'pacs')
if not os.path.exists(pacs_path):
    os.mkdir(pacs_path)
for patient, infos in index_PD_Gait.iterrows():
    if infos['Type'] in ['Med-On', 'Med-Off'] and infos['Useable'] == True:
        fpath = os.path.join(workspace, '1_Raw', f"PD_{patient}_{infos['Type']}_lfp.fif")
        if not os.path.exists(fpath): continue
        raw = mne.io.Raw(fpath, preload=True)
        
        print(f"{fpath}")
        target_channels = infos['Target'].split(',')
        if 'C2' not in raw.ch_names: continue
        # if infos['Recording'] == 'EXT': continue
        low_fq_range = np.arange(1.0,32.0,0.5)
        high_fq_range = np.arange(40.0,250.0,1.0)
        
        for i, st in enumerate(states):
            if infos[st] is np.nan: continue
            pac_key = f"{infos['Type']} {st}"
            if pac_key not in list(gc_pacs.keys()):
                gc_pacs[pac_key] = []
                gc_pacs_patient_list[pac_key] = []
            time_slices = [float(x) for x in infos[st].split('-')]
            if time_slices[1]-time_slices[0] < 60.0: continue
            raw_segment = raw.copy().pick(target_chs).crop(tmin=time_slices[0], tmax=time_slices[0]+60.0)
            
            gc_xpacs = {}
            for ch in raw_segment.ch_names:
                p = tensorpac.Pac(idpac=(6, 0, 0), f_pha=low_fq_range, f_amp=high_fq_range)
                xpac = p.filterfit(raw_segment.info['sfreq'], 
                                   # raw_segment[ch][0][0], 
                                   raw_segment[ch][0][0], n_jobs=-1)
                gc_xpacs[ch] = xpac
            
            gc_pacs[pac_key].append(gc_xpacs)
            gc_pacs_patient_list[pac_key].append(patient)
            vmax = 0.75 * np.max(gc_xpacs['C2'])
            print(f"\n{vmax}")
            plt.subplot(1,3,i+1)
            plt.imshow(gc_xpacs['C2'], interpolation='bicubic', 
                       extent=[low_fq_range[0], low_fq_range[-1], high_fq_range[-1], high_fq_range[0]],
                       vmax=vmax)
            plt.gca().invert_yaxis()
        # plt.show()
        plt.savefig(os.path.join(pacs_path, f"{patient}_C2_pacs_0624.png"))
        print()
import pickle
with open(os.path.join(pacs_path, f"lfp_cont_gcpacs.pkl"), 'wb') as f:
    gc_object = {'gc_pacs': gc_pacs, 
                 'patient_list': gc_pacs_patient_list,
                 'low_fq_range': low_fq_range,
                 'high_fq_range': high_fq_range}
    pickle.dump(gc_object, f)
        