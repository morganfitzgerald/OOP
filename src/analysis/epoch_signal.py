import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

def epoch_signals(signals, ecg_clean, sampling_rate, SUB_NUM, PLOT=True, SAVE=False):
    # Ensure the lengths of P onsets, T offsets, and R peaks are identical, implying each cycle is complete.
    assert len(signals["ECG_P_Onsets"]) == len(signals["ECG_T_Offsets"])

    # Calculate the lengths of each cardiac cycle (from P onset to T offset) and compute the average cycle length.
    cycle_lens = np.zeros(len(signals["ECG_P_Onsets"]))
    for idx, (p_on, t_off) in enumerate(zip(signals["ECG_P_Onsets"], signals["ECG_T_Offsets"])):
        cycle_lens[idx] = t_off - p_on
    cycle_mean = np.nanmean(cycle_lens)

    # Filter out NaN values from P onsets to handle missing data points.
    nan_mask = np.isnan(signals["ECG_P_Onsets"])
    no_nan_p_on = np.asarray(signals["ECG_P_Onsets"])[~nan_mask]

    # Exclude the first and last cycles to avoid incomplete data at the beginning and end of the recording.
    no_nan_p_on = no_nan_p_on[1:-1]

    # Set the pre-cycle buffer and calculate the total epoch length including a post-cycle buffer.
    pre_p = 200  # Buffer before P Onset in milliseconds/sample points
    avg_cycle_len = cycle_mean + 200  # Buffer after end of cycle in milliseconds/sample points
    r_latencies = np.zeros(len(no_nan_p_on))
    ecg_clean_idx = np.arange(0, len(ecg_clean))
    epochs_df = pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"])
    fig = plt.gcf()

    for idx, p_on in enumerate(no_nan_p_on):
        # Define each epoch using the calculated buffers.
        window_start = int(p_on) - int(pre_p)
        window_end = int(p_on) + int(avg_cycle_len)
        window = ecg_clean[window_start:window_end]

        # Apply detrending to remove linear trend from the epoch, enhancing signal analysis.
        window_detrended = detrend(window)

        # Generate time values for plotting, adjusted for the sampling frequency (sampling_rate).
        x_vals = np.linspace(-pre_p / sampling_rate, avg_cycle_len / sampling_rate, len(window_detrended))

        # Create a temporary DataFrame to store the current epoch's data.
        temp_df = pd.DataFrame({
            "signal_x": x_vals,
            "signal_y": window_detrended,
            "index": ecg_clean_idx[window_start:window_end],
            "cycle": np.repeat(idx + 1, len(x_vals))
        })

        # Append the epoch data to the main DataFrame containing all epochs, only if temp_df is not empty or all-NA.
        if not temp_df.empty and not temp_df.isna().all().all():
            epochs_df = pd.concat([epochs_df, temp_df])

        # Identify the R peak within the detrended epoch, storing its latency for further analysis.
        r_peak_ind = np.argmax(window_detrended)
        r_latencies[idx] = x_vals[r_peak_ind]

        # Plotting within loop only if PLOT is True
        if PLOT:
            # If it's the first epoch being plotted, create a new figure
            if idx == 0:
                plt.figure(figsize=(10, 6))
            plt.plot(x_vals, window_detrended)

    # Show and/or save the plot only if requested
    if PLOT:
        plt.title(f"Participant {SUB_NUM}: All Cycles Aligned to P Onset", size=10)
        plt.xlabel('Time (s)')
        plt.ylabel('ECG Signal (mV)')
        plt.show()

    if SAVE:
        # Ensure this block is executed only if plotting is done
        plt.savefig(f"../docs/figures/{SUB_NUM}_allcycles_alignedtoP.png")

    return epochs_df, r_latencies
