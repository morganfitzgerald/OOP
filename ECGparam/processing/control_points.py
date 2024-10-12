import numpy as np
import pandas as pd
import neurokit2 as nk


def extract_control_points(clean_ecg_sig, sampling_rate):
    """
    Process an ECG signal to find and clean R peaks and delineate the QRS complex.

    Parameters:
    - clean_ecg_sig: ndarray. The ECG clean signal data from neurokit.
    - sampling_rate: int. The sampling rate of the ECG signal, defaults to 1000 Hz.

    Returns:
    - nk_signals: dict. The processed ECG signal components.
    - waves: dict. The delineated ECG wave components.
    """

    # Find R peaks using NeuroKit's ecg_peaks function
    _, rpeaks = nk.ecg_peaks(clean_ecg_sig, sampling_rate)
    
    # Initialize an array for R peak heights
    r_heights = np.zeros(len(rpeaks['ECG_R_Peaks']))
    
    # Extract R peak heights
    for idx, r in enumerate(rpeaks['ECG_R_Peaks']):
        r_heights[idx] = clean_ecg_sig[r]
    
    # Calculate median and standard deviation of R peak heights
    median_r = np.median(r_heights)
    std_r = np.std(r_heights)
    STDS = 1.5  # Threshold of standard deviations for outlier detection
    
    # Identify outliers based on threshold
    idx_to_remove = [idx for idx, r in enumerate(r_heights) if r <= median_r - STDS*std_r]
    
    # Create DataFrame for R peaks and remove outliers
    r_df = pd.DataFrame({'x_vals': rpeaks['ECG_R_Peaks'], 'y_vals': r_heights})
    r_df_cleaned = r_df.drop(labels=idx_to_remove)
    
    # Update rpeaks with cleaned values
    rpeaks_cleaned = np.asarray(r_df_cleaned['x_vals'])
    
    # Delineate the QRS complex with the cleaned R peaks
    _, nk_signals = nk.ecg_delineate(clean_ecg_sig, rpeaks_cleaned, sampling_rate=sampling_rate)
    
    return nk_signals
