import neurokit2 as nk
import numpy as np

def simulate_ecg_sig(
    duration=5, sampling_rate=1000, heart_rate=80, amplitude_factor=1.0, normalize=False
):
    """
    Simulates an ECG signal using the neurokit2 library.

    Parameters:
    - duration (float): Duration of the simulated ECG signal in seconds.
    - sampling_rate (int): Sampling rate of the ECG signal.
    - heart_rate (int): Heart rate of the simulated ECG signal in beats per minute.
    - amplitude_factor (float): Scaling factor for the amplitude of the simulated ECG signal.
    - normalize (bool): Whether to normalize the ECG signal or not.

    Returns:
    - simulated_ecg (numpy.ndarray): Simulated ECG signal with the specified parameters.
    """
    # Simulate the ECG signal with neurokit2
    ecg_signal = nk.ecg_simulate(
        duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate
    )

    # Apply amplitude scaling
    ecg_signal *= amplitude_factor

    # Apply normalization if specified
    if normalize:
        ecg_min, ecg_max = np.min(ecg_signal), np.max(ecg_signal)
        ecg_signal = (ecg_signal - ecg_min) / (ecg_max - ecg_min)

    return ecg_signal
