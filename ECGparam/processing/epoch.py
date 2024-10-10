import numpy as np
import pandas as pd
from scipy.signal import detrend

def epoch_cycles(signals, ecg_clean, sampling_rate):
    """
    Extract epochs from ECG data based on P onsets and T offsets.

    Parameters:
    signals : dict
        A dictionary containing signal data, including 'ECG_P_Onsets' and 'ECG_T_Offsets'.
    ecg_clean : ndarray
        The cleaned ECG signal.
    sampling_rate : int
        The sampling rate of the ECG signal.

    Returns:
    epochs_df : DataFrame
        A DataFrame containing the extracted epochs for each cycle.
    """
    # Ensure the lengths of P onsets, T offsets, and R peaks are identical.
    assert len(signals["ECG_P_Onsets"]) == len(signals["ECG_T_Offsets"])

    # Calculate the lengths of each cardiac cycle and compute the average cycle length.
    p_onsets = np.asarray(signals["ECG_P_Onsets"])
    t_offsets = np.asarray(signals["ECG_T_Offsets"])
    cycle_lens = t_offsets - p_onsets
    cycle_mean = np.nanmean(cycle_lens)

    # Filter out NaN values from P onsets and exclude the first and last cycles.
    no_nan_p_on = p_onsets[~np.isnan(p_onsets)][1:-1]

    # Precompute constants.
    pre_p = 200
    avg_cycle_len = int(cycle_mean + 200)
    pre_p_sec = pre_p / sampling_rate
    avg_cycle_len_sec = avg_cycle_len / sampling_rate

    # Prepare lists for storing the results.
    epoch_list = []

    for idx, p_on in enumerate(no_nan_p_on):
        # Define each epoch window.
        window_start = int(p_on) - pre_p
        window_end = int(p_on) + avg_cycle_len
        window = ecg_clean[window_start:window_end]

        # Apply detrending and generate time values.
        window_detrended = detrend(window)
        x_vals = np.linspace(-pre_p_sec, avg_cycle_len_sec, len(window_detrended))

        # Store data in a list.
        epoch_list.append(pd.DataFrame({
            "signal_x": x_vals,
            "signal_y": window_detrended,
            "index": np.arange(window_start, window_end),
            "cycle": idx + 1
        }))

    # Combine all epochs into a DataFrame after the loop.
    epochs_df = pd.concat(epoch_list, ignore_index=True)

    return epochs_df