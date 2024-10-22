import mne
import os
import math
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
from pactools import Comodulogram
import tensorpac
from statsmodels.formula.api import ols

def get_cycle_durations(epochs, target_ids):
    cycle_durations = {}
    for eid in target_ids:
        event_presents = epochs[eid].events
        if eid not in list(cycle_durations):
            cycle_durations[eid] = []
        for i in range(1, len(event_presents)):
            start_point = event_presents[i-1][0] 
            end_point = event_presents[i][0]
            start_time = start_point / epochs.info['sfreq']
            end_time = end_point / epochs.info['sfreq']
            duration = end_time - start_time
            cycle_durations[eid].append(duration)
    return cycle_durations

cycle_duras = {}
def process_value(patient, value, key):
    if key == 'mean':
        return np.mean(value)
    elif key == 'std':
        return np.std(value)
    elif key == 'range':
        return np.max(value) - np.min(value) if value else 0
    elif key == 'mad':
        return np.mean(np.abs(value - np.mean(value)))
    elif key == 'list':
        return patient
    elif key == 'num':
        return len(value)
    elif key == 'mid':
        return np.median(value)
    else:
        raise ValueError(f"Invalid key: {key}")
epochs_repo = {}
for patient, infos in index_PD_Gait.iterrows():
    if infos['Type'] in ['Med-On', 'Med-Off'] and infos['Useable'] == True:
        fpath = os.path.join(workspace, '1_Raw', f"PD_{patient}_{infos['Type']}_brain.fif")
        if not os.path.exists(fpath): continue
        raw = mne.io.Raw(fpath, preload=True)
        events, event_id = update_events(raw)
        print(f"{fpath}")
        target_channels = infos['Target'].split(',')
        if 'C2' not in raw.ch_names: continue
        # if infos['Recording'] == 'EXT': continue
        epochs = mne.Epochs(raw, events, event_id, on_missing='ignore')
        print(epochs.events.shape)
        
        if infos['Time Slices'] is not np.nan:
            crop_time        = [float(x) for x in infos['Time Slices'].split('-')]
            if raw.times[-1] - crop_time[0] < 60.0: continue
            cropped_raw      = raw.copy().crop(crop_time[0], crop_time[0]+60.0)
            events, event_id = update_events(cropped_raw)
            epochs           = mne.Epochs(cropped_raw, events, event_id, on_missing='ignore')
            print(epochs.events.shape)
            epochs_repo[f"{patient} {infos['Type']} Begin"] = epochs
        
        if infos['Time Slices'] is not np.nan:
            crop_time        = [float(x) for x in infos['Time Slices'].split('-')]
            cropped_raw      = raw.copy().crop(crop_time[1]-70, crop_time[1]-10.0)
            events, event_id = update_events(cropped_raw)
            epochs           = mne.Epochs(cropped_raw, events, event_id, on_missing='ignore')
            print(epochs.events.shape)
            epochs_repo[f"{patient} {infos['Type']} End"] = epochs
        
        if infos['Active Sound'] is not np.nan:
            crop_time        = [float(x) for x in infos['Active Sound'].split('-')]
            cropped_raw      = raw.copy().crop(crop_time[0], crop_time[0]+60.0)
            events, event_id = update_events(cropped_raw)
            epochs           = mne.Epochs(cropped_raw, events, event_id, on_missing='ignore')
            print(epochs.events.shape)
            epochs_repo[f"{patient} {infos['Type']} AS"] = epochs
        
        if infos['Passive Sound'] is not np.nan:
            crop_time        = [float(x) for x in infos['Passive Sound'].split('-')]
            cropped_raw      = raw.copy().crop(crop_time[0], crop_time[0]+60.0)
            events, event_id = update_events(cropped_raw)
            epochs           = mne.Epochs(cropped_raw, events, event_id, on_missing='ignore')
            print(epochs.events.shape)
            epochs_repo[f"{patient} {infos['Type']} PS"] = epochs
for epoch_key, epochs in epochs_repo.items():
    patient, med_type, stage = epoch_key.split(' ')
    ids_candidates = ['50', '54', '60', '64']
    target_ids = [i for i in ids_candidates if i in epochs.event_id]
    dist = get_cycle_distribution(epochs, target_ids)
    dura = get_cycle_durations(epochs, target_ids)
    
    for key, value in dura.items():
        save_key = f"{med_type} {key} {stage}"
        if save_key not in cycle_duras:
            cycle_duras[save_key] = {'mean': [], 'std': [], 'range': [], 'mad': [], 
                                     'list': [], 'num':[], 'mid':[]}
        for stats_key in cycle_duras[save_key].keys():
            cycle_duras[save_key][stats_key].append(process_value(patient, value, stats_key))
    
    heel_strike_key = f"{med_type} Heel Strike {stage}"
    if heel_strike_key not in cycle_duras:
        cycle_duras[heel_strike_key] = {'mean': [], 'std': [], 'range': [], 'mad': [], 
                                        'list': [], 'num':[], 'mid':[]}
    heel_strike_durations = dura.get('50', []) + dura.get('60', [])
    
    for stats_key in cycle_duras[save_key].keys():
        cycle_duras[heel_strike_key][stats_key].append(process_value(patient, heel_strike_durations, stats_key))
