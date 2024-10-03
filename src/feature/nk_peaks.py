import numpy as np
import neurokit2 as nk

def nk_peaks(clean_ecg_sig, sampling_rate=1000):
    """
    Process an ECG signal to find and clean R peaks and delineate the QRS complex using Neurokit2.

    Parameters:
    - clean_ecg_sig: ndarray. The ECG clean signal data from neurokit.
    - sampling_rate: int. The sampling rate of the ECG signal, defaults to 1000 Hz.

    Returns:
    - signals: dict. The processed ECG signal components.
    - rpeaks_cleaned: ndarray. The indices of the detected and cleaned R peaks.
    - waves: dict. The delineated ECG wave components.
    """
    # Detect R peaks
    _, rpeaks = nk.ecg_peaks(clean_ecg_sig, sampling_rate=sampling_rate)
    rpeaks_indices = rpeaks['ECG_R_Peaks']
    
    # Extract R peak heights using vectorized indexing
    r_heights = clean_ecg_sig[rpeaks_indices]
    
    # Compute median and standard deviation
    median_r = np.median(r_heights)
    std_r = np.std(r_heights)
    threshold = median_r - 1.5 * std_r
    
    # Identify and remove outliers using boolean indexing
    valid_indices = rpeaks_indices[r_heights > threshold]
    
    # Delineate ECG waves using cleaned R peaks
    waves, signals = nk.ecg_delineate(clean_ecg_sig, valid_indices, sampling_rate=sampling_rate)
    
    return signals, valid_indices, waves
