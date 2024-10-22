import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

plt.close('all')
states = ['Rest Time', 'Stand Time', 'Time Slices']
medication = ['Med-On', 'Med-Off']
psd = {}; psd_patient_list = {}
excludes_list = []
target_chs    = ['C2','C1','C4','C3','Cz','FC3','FC4']
central_chs   = ['C2','C1','C4','C3','Cz']
eeg_chs       = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O2', 'C1', 'C2', 'O1']
emg_chs       = ['LEMG', 'REMG']
selected_chs  = eeg_chs + emg_chs
cnt = 0
for patient, infos in index_PD_Gait.iterrows():
    if infos['Type'] in ['Med-On', 'Med-Off'] and infos['Useable'] == True:
        target_channels = infos['Target'].split(',')
        if 'C2' not in target_channels: continue
        fpath = os.path.join(workspace, '1_RescaledRaw', f"PD_{patient}_{infos['Type']}_brain.fif")
        fpath = os.path.join(workspace, '1_RescaledRaw', f"PD_{patient}_{infos['Type']}_brain.fif")
        if not os.path.exists(fpath): continue
        raw = mne.io.Raw(fpath, preload=True)
        print(f"{fpath}")
        
        for st in states:
            if infos[st] is np.nan: continue
            psd_key = f"{infos['Type']} {st}"

            if psd_key not in list(psd.keys()):
                psd[psd_key] = []
                psd_patient_list[psd_key] = []
            time_slices = [float(x) for x in infos[st].split('-')]
            posture_duration = time_slices[1] - time_slices[0]
            if posture_duration < 60.0: continue
            raw_segment = raw.copy().pick(selected_chs).crop(tmin=time_slices[0], tmax=time_slices[0]+60.0)
            raw_segment = raw_segment.notch_filter([50.0, 100.0, 150.0, 200.0, 250.0], notch_widths=2.0, picks='emg')
            # raw_segment = raw_segment.notch_filter([200.0], notch_widths=2.0, picks='eeg')
            psds, freqs = mne.time_frequency.psd_array_welch(raw_segment.get_data(), 
                                                             sfreq=raw_segment.info['sfreq'], 
                                                             fmin=1.0, fmax=500.0,
                                                             n_fft=2048)
            psds_normalized = psds
            psd[psd_key].append(psds_normalized)
            psd_patient_list[psd_key].append(patient)
            cnt += 1
            
psd_info = {}
psd_info['sfreqs']   = raw_segment.info['sfreq']
psd_info['freqs']    = freqs
psd_info['ch_names'] = raw_segment.ch_names
print(f'PSD calculation and normalization finished (n={cnt})!')

bandwidth['Wide-Gamma'] = [40.0, 200.0]
band_selections = ['Alpha', 'Gamma', 'Low-Beta', 'High-Beta', 'Beta', 'Wide-Gamma'] 
psd_rms     = {}; 
channels    = psd_info['ch_names']
# central_chs = ['C2','C1','C4','C3','Cz']
central_chs = ['C2','C1','Cz']
# central_chs = ['Cz']
states      = ['Rest Time', 'Stand Time', 'Time Slices']
medication  = ['Med-On', 'Med-Off']
for bs in band_selections:
    bw          = bandwidth[bs]
    psd_rms[bs] = {}
    freqs_sels  = np.where((freqs>=bw[0]) & (freqs<=bw[1]))[0]
    ch_indices  = [channels.index(ch) for ch in central_chs]
    for st in states:
        plt.figure(figsize=(6, 3))
        for med in medication:
            key        = f"{med} {st}"
            value      = psd[key]
            psd_matrix = np.array(value)[:, ch_indices]
            psd_matrix = psd_matrix[...,freqs_sels]
            power_sum  = np.trapz(psd_matrix, freqs[freqs_sels])
            single_rms = np.sqrt(power_sum/(bw[1]-bw[0]))
            single_rms = psd_matrix.mean(axis=1)
            psd_rms[bs][key] = single_rms
            print(bs, bw, key, psd_matrix.shape, single_rms.shape)
            
            # Plot the mean PSD with confidence interval
            psd_matrix = np.log10(psd_matrix)
            mean_psd   = np.mean(psd_matrix, axis=0)
            sem_psd    = np.std(psd_matrix, axis=0) / np.sqrt(psd_matrix.shape[0])
            plt.plot(freqs[freqs_sels], mean_psd, label=med)
            plt.fill_between(freqs[freqs_sels], mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.title('Averaged PSD with Confidence Interval')
        plt.legend()
        plt.show()