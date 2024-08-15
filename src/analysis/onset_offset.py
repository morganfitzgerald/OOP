
# import numpy as np


import numpy as np

def find_peak_boundaries(signal, peak_index, peak_height=None, half_height=None):
    """
    Find the indices for the onset (left) and offset (right) of a peak in a signal, based on either the specified 
    half height or the peak's height from which the half height will be calculated.

    If peak_height is provided, half_height is calculated as half of peak_height. If half_height is directly provided,
    it is used as is. One of peak_height or half_height must be provided.

    Parameters:
    - signal (array-like): The signal array in which to find the onset and offset.
    - peak_index (int): The index of the peak around which to find the onset and offset.
    - peak_height (float, optional): The height of the peak, used to calculate the half height for onset and offset determination.
    - half_height (float, optional): The height at which to determine the onset and offset from the peak.

    Returns:
    - tuple: A tuple containing the indices of the onset and offset. If an onset or offset is not found within the signal bounds,
             None is returned for that side.

    Raises:
    - ValueError: If neither peak_height nor half_height is provided.
    """
    if half_height is None:
        if peak_height is None:
            raise ValueError("Either peak_height or half_height must be provided.")
        half_height = 0.5 * peak_height


    #peak_index is the cycle value, not the whole-signal value 
    le_ind = next((val for val in range(peak_index - 1, 0, -1) if np.abs(signal[val]) <= np.abs(half_height)), None)
    ri_ind = next((val for val in range(peak_index + 1, len(signal), 1) if np.abs(signal[val]) <= np.abs(half_height)), None)
    
    return le_ind, ri_ind


