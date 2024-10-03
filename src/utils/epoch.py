import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

def epoch_cycles(signals, ecg_clean, sampling_rate, SUB_NUM, PLOT=True, SAVE=False):
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
    r_latencies = []

    if PLOT:
        plt.figure(figsize=(10, 6))

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

        # Identify the R peak latency.
        r_peak_ind = np.argmax(window_detrended)
        r_latencies.append(x_vals[r_peak_ind])

        # Plotting within loop only if PLOT is True.
        if PLOT:
            plt.plot(x_vals, window_detrended)

    # Combine all epochs into a DataFrame after the loop.
    epochs_df = pd.concat(epoch_list, ignore_index=True)
    r_latencies = np.array(r_latencies)

    # Show and/or save the plot only if requested.
    if PLOT:
        plt.title(f"Participant {SUB_NUM}: All Cycles Aligned to P Onset", size=10)
        plt.xlabel('Time (s)')
        plt.ylabel('ECG Signal (mV)')
        plt.show()

    if SAVE:
        plt.savefig(f"../docs/figures/{SUB_NUM}_allcycles_alignedtoP.png")

    return epochs_df, r_latencies
