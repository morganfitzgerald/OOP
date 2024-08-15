
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def average_fft_of_epochs_loglog(epochs_df, sampling_rate, PLOT=True):
    """
    Calculates the FFT for each epoch in the epochs DataFrame, averages them,
    and optionally plots the magnitude spectrum on a log-log scale.

    Parameters:
    - epochs_df (pd.DataFrame): DataFrame containing the epochs with columns ['signal_x', 'signal_y', 'index', 'cycle'].
    - sampling_rate (int): The sampling rate of the ECG signal.
    - PLOT (bool, optional): Whether to plot the averaged FFT magnitude spectrum. Default is True.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - The positive frequencies of the FFT magnitude spectrum.
        - The averaged FFT magnitude spectrum.
    """

    fft_results = []
    cycles = epochs_df['cycle'].unique()

    for cycle in cycles:
        signal = epochs_df[epochs_df['cycle'] == cycle]['signal_y'].values
        fft_res = np.fft.fft(signal)
        fft_results.append(fft_res)

    # Average the FFT results
    avg_fft = np.mean(np.abs(fft_results), axis=0)

    # Frequency bins
    N = len(signal)  # Assume length of signal is consistent across epochs
    f_vals = np.fft.fftfreq(N, 1 / sampling_rate)

    # Ensure to plot only the positive frequencies and magnitudes
    positive_freqs = f_vals[:N // 2]
    positive_magnitudes = avg_fft[:N // 2]

    # Filtering out the zero frequency before plotting to avoid taking log(0)
    positive_freqs = positive_freqs[positive_freqs > 0]
    positive_magnitudes = positive_magnitudes[1:len(positive_freqs) + 1]  # Adjust indices due to removal of the zero frequency

    if PLOT:
        # Plotting the averaged FFT magnitude spectrum on a log-log scale
        plt.figure(figsize=(6, 4))
        plt.loglog(positive_freqs, positive_magnitudes, color='blue', lw=2, alpha=0.8)  # Log-log plot
        plt.title('Averaged FFT of All Epochs')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, which="both", ls="-")  # Enable grid for better readability
        plt.show()

    return positive_freqs, positive_magnitudes


#############
# def average_fft_of_epochs_loglog(epochs_df, sampling_rate):
#     """
#     Calculates the FFT for each epoch in the epochs DataFrame,

#     Parameters:
#     - epochs_df (pd.DataFrame): DataFrame containing the epochs with columns ['signal_x', 'signal_y', 'index', 'cycle'].
#     - sampling rate (int): The sampling rate of the ECG signal.

#     Returns:
#     - np.ndarray: The averaged FFT magnitude spectrum.
#     """

#     fft_results = []
#     cycles = epochs_df['cycle'].unique()

#     for cycle in cycles:
#         signal = epochs_df[epochs_df['cycle'] == cycle]['signal_y'].values
#         fft_res = np.fft.fft(signal)
#         fft_results.append(fft_res)

#     # Average the FFT results
#     avg_fft = np.mean(np.abs(fft_results), axis=0)

#     # Frequency bins
#     N = len(signal)  # Length of the signal
#     f_vals = np.fft.fftfreq(N, 1/sampling_rate)

#     # Ensure to plot only the positive frequencies and magnitudes
#     positive_freqs = f_vals[:N // 2]
#     positive_magnitudes = avg_fft[:N // 2]

#     # Filtering out the zero frequency before plotting to avoid taking log(0)
#     positive_freqs = positive_freqs[positive_freqs > 0]
#     positive_magnitudes = positive_magnitudes[1:len(positive_freqs)+1]  # Adjust indices due to removal of the zero frequency

#     return positive_freqs, positive_magnitudes