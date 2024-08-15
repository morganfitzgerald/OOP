import neurokit2 as nk
import numpy as np


# Function to simulate ECG signal using neurokit2
def simulate_ecg_signal(
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
    - scaled_ecg (numpy.ndarray): Simulated ECG signal with the specified parameters.
    """

    # Use neurokit2 to simulate an ECG signal
    ecg_signal = nk.ecg_simulate(
        duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate
    )

    # Scale the ECG signal by the specified amplitude factor
    simulated_ecg = ecg_signal * amplitude_factor

    # Normalize the ECG signal if specified
    if normalize:
        simulated_ecg = (simulated_ecg - np.min(simulated_ecg)) / (
            np.max(simulated_ecg) - np.min(simulated_ecg)
        )

    # Return the scaled or normalized ECG signal
    return simulated_ecg
