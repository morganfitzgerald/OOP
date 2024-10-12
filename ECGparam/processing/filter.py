from scipy.signal import butter, filtfilt, iirnotch

def high_pass_filter(signal, fs, cutoff_frequency=0.05, order=4):
    """
    Apply a high-pass filter to remove low-frequency noise from the signal.
    :param signal: Input ECG signal
    :param fs: Sampling frequency (Hz)
    :param cutoff_frequency: Cutoff frequency for the high-pass filter (default is 0.05 Hz)
    :param order: Filter order
    :return: Filtered signal
    """
    b, a = butter(order, cutoff_frequency, btype='high', fs=fs)
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, f0=50, quality_factor=30):
    """
    Apply a notch filter to remove power line noise (50Hz or 60Hz).
    :param signal: Input ECG signal
    :param fs: Sampling frequency (Hz)
    :param f0: Frequency to be notched (default is 50Hz)
    :param quality_factor: Q-factor for the filter (controls sharpness)
    :return: Filtered signal
    """
    b, a = iirnotch(f0, quality_factor, fs)
    return filtfilt(b, a, signal)
