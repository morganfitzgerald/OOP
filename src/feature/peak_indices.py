import numpy as np

def extract_peak_indices(xs, sig, peak_params):
    """
    Finds the indices of the start, peak, and end points of a signal around a peak.

    Parameters:
    - xs (np.array): The x-axis values corresponding to the signal data.
    - sig (np.array): The y-signal data as a 1D numpy array.
    - peak_params (np.array): Array containing peak parameters where:
      - peak_params[0]: x-value of the peak (center)
      - peak_params[1]: magnitude (y-value) of the peak

    Returns:
    - start_index (int or float): Index where the signal crosses half the peak magnitude before the peak.
    - peak_index (int): Index of the peak itself.
    - end_index (int or float): Index where the signal crosses half the peak magnitude after the peak.
    """
    # Compute half magnitude
    half_mag = peak_params[1] / 2

    # Get the index of the peak in the signal
    peak_index = np.argmin(np.abs(xs - peak_params[0]))

    # Handle positive and negative peaks separately
    if sig[peak_index] > 0:
        condition_start = sig[:peak_index] < half_mag
        condition_end = sig[peak_index:] < half_mag
    else:  # If the peak is negative, flip the conditions
        condition_start = sig[:peak_index] > half_mag
        condition_end = sig[peak_index:] > half_mag

    # Vectorized search for start and end indices
    start_indices = np.flatnonzero(condition_start)
    end_indices = np.flatnonzero(condition_end)

    # Determine start and end indices or assign NaN if not found
    start_index = start_indices[-1] if start_indices.size > 0 else np.nan
    end_index = peak_index + end_indices[0] if end_indices.size > 0 else np.nan

    return start_index, peak_index, end_index
