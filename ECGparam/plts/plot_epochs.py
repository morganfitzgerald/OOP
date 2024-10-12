import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

def plot_epochs(signals, ecg_clean, sampling_rate, align_to='P_Onsets', pre_offset=200, post_offset=200):
    """
    Plot ECG cycles aligned to a specific feature (e.g., P onsets, R peaks).

    Parameters
    ----------
    signals : dict
        A dictionary containing the onset and offset points of ECG signals, such as P onsets, T offsets, and R peaks.
    ecg_clean : ndarray
        The cleaned ECG signal data.
    sampling_rate : int
        The sampling rate of the ECG signal (in Hz).
    align_to : str, optional
        The feature to align the cycles to, such as 'P_Onsets' or 'R_Peaks'. Default is 'P_Onsets'.
    pre_offset : int, optional
        The number of samples to include before the aligned point (e.g., before P onset). Default is 200 samples.
    post_offset : int, optional
        The number of samples to include after the aligned point. Default is 200 samples.

    Returns
    -------
    epochs_df : pd.DataFrame
        A DataFrame containing the aligned ECG cycles with time values and cycle index.
    """
    
    # Ensure the chosen alignment feature exists in the signal dictionary.
    if align_to not in signals:
        raise KeyError(f"The alignment feature '{align_to}' is not found in the signal dictionary.")
    
    # Extract the onset points for the specified alignment feature.
    align_points = np.asarray(signals[align_to])
    
    # Filter out NaN values and exclude first and last cycles if necessary.
    valid_align_points = align_points[~np.isnan(align_points)][1:-1]

    # Precompute necessary time constants.
    pre_offset_sec = pre_offset / sampling_rate
    post_offset_sec = post_offset / sampling_rate

    # Prepare lists for storing the epochs and R peak latencies.
    epoch_list = []
    r_latencies = []

    # Create a new plot for aligned ECG cycles.
    plt.figure(figsize=(10, 6))

    for idx, align_point in enumerate(valid_align_points):
        # Define the epoch window.
        window_start = int(align_point) - pre_offset
        window_end = int(align_point) + post_offset
        window = ecg_clean[window_start:window_end]

        # Detrend the signal and create time values.
        window_detrended = detrend(window)
        x_vals = np.linspace(-pre_offset_sec, post_offset_sec, len(window_detrended))

        # Append the epoch to the list.
        epoch_list.append(pd.DataFrame({
            "signal_x": x_vals,
            "signal_y": window_detrended,
            "index": np.arange(window_start, window_end),
            "cycle": idx + 1
        }))

        # Find the R peak latency within the window.
        r_peak_ind = np.argmax(window_detrended)
        r_latencies.append(x_vals[r_peak_ind])

        # Plot the detrended signal.
        plt.plot(x_vals, window_detrended)

    # Combine all epochs into a DataFrame.
    epochs_df = pd.concat(epoch_list, ignore_index=True)

    # Label the plot.
    plt.title(f"Aligned ECG Cycles to {align_to}", size=10)
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Signal (mV)')
    plt.show()

    return epochs_df, np.array(r_latencies)
