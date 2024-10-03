import numpy as np


def find_extremum(signal, xs, start_idx, end_idx, mode):
    """
    Finds the index and value of the minimum or maximum in a specified segment of a signal array,
    along with the corresponding x-axis value (center).

    Parameters:
    - signal (np.array): The signal data as a 1D numpy array.
    - xs (np.array): The x-axis values corresponding to the signal data, used to determine the center value.
    - start_idx (int): The starting index of the segment within the signal array to search for the extremum.
    - end_idx (int): The ending index of the segment within the signal array to search for the extremum.
    - mode (str, optional): Determines the type of extremum to find. 
      Use 'min' for minimum (default) or 'max' for maximum.

    Returns:
    - idx_absolute (int): The absolute index in the original signal array where the extremum is found.
    - value (float): The value of the signal at the extremum.
    - center (float): The x-axis value corresponding to the extremum, extracted from the xs array.

    Raises:
    - AssertionError: If the mode is not 'min' or 'max'.

    Example:
    >>> signal = np.array([1, 3, 2, 5, 4])
    >>> xs = np.array([10, 20, 30, 40, 50])
    >>> find_extremum(signal, xs, 1, 4, 'max')
    (3, 5, 40)  # Index 3, value 5, center value 40
    """
    assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
    
    if mode == 'min':
        idx_relative = np.argmin(signal[start_idx:end_idx])
    else:
        idx_relative = np.argmax(signal[start_idx:end_idx])
    
    idx_absolute = idx_relative + start_idx
    value = signal[idx_absolute]
    center = xs[idx_absolute]
    
    return idx_absolute, value, center
