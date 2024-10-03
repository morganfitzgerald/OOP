from scipy.signal import correlate
import numpy as np

def match_ecg_template(template, signals, metadata):
    """
    Finds the most similar signal to a given template among a list of signals using cross-correlation.

    Parameters:
    - template (numpy.ndarray): The template signal for comparison.
    - signals (list of numpy.ndarray): List of signals to compare against the template.
    - metadata (dict): Metadata containing information about the signals.

    Returns:
    - selected_signal (numpy.ndarray): The signal that is most similar to the template.
    - selected_signal_name (str): The name of the most similar signal.
    - selected_signal_index (int): The index of the most similar signal in the list.
    """

    # Precompute the length of the template for later use
    template_length = len(template)

    # Initialize variables to store the maximum correlation and the selected signal
    max_correlation = -np.inf
    selected_signal = None
    selected_signal_name = None
    selected_signal_index = None

    # Iterate through each signal and calculate cross-correlation with the template
    for ind, signal in enumerate(signals):
        # Perform cross-correlation and normalize by the length of the template
        correlation = np.max(correlate(template, signal, mode='valid')) / template_length
        
        # Update if the current signal has a higher correlation
        if correlation > max_correlation:
            max_correlation = correlation
            selected_signal = signal
            selected_signal_name = metadata.get(f'sig{str(ind).zfill(2)}', {}).get('signal_name', f'Signal {ind}')
            selected_signal_index = ind

    # Return the most similar signal and its details
    return selected_signal, selected_signal_name, selected_signal_index
