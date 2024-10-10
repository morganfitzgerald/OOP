
import numpy as np
import traceback 


#adding fail safe hardcoded guesses for debugging
def extract_peak_indices(xs, sig, peak_params):
    """
    Finds the indices of the start, peak, and end points of a signal around a peak.

    Parameters:
    - xs (np.array): The x-axis values corresponding to the signal data.
    - sig (np.array): The signal data as a 1D numpy array.
    - peak_params (np.array): Array containing peak parameters where:
      - peak_params[0]: x-value of the peak (center)
      - peak_params[1]: y-value (magnitude) of the peak

    Returns:
    - left_index (int or float): Index where the signal crosses half the peak magnitude before the peak.
    - right_index (int or float): Index where the signal crosses half the peak magnitude after the peak.
    - peak_index (int): Index of the peak itself.
    """

    try:
        # Calculate half of the peak magnitude for finding half-max points
        half_mag = peak_params[1] / 2

        # Find the index of the peak by locating the x-value closest to the peak's center
        peak_index = np.argmin(np.abs(xs - peak_params[0]))

        # Check if the peak index is within bounds
        if peak_index < 0 or peak_index >= len(xs):
            raise IndexError(f"Peak index {peak_index} is out of bounds for xs array of size {len(xs)}.")

        # Handle case where peak magnitude is zero by returning NaN for the left and right indices
        if peak_params[1] == 0:
            print("Warning: Peak magnitude is zero, returning NaN for start and end indices.")
            return np.nan, np.nan, peak_index

        # Define conditions to find when the signal drops below (for positive peaks) 
        # or rises above (for negative peaks) half magnitude
        if peak_params[1] > 0:
            # Positive peak: find where the signal is less than half the peak height
            condition_start = sig[:peak_index] < half_mag
            condition_end = sig[peak_index:] < half_mag
        else:
            # Negative peak: find where the signal is greater than half the peak height
            condition_start = sig[:peak_index] > half_mag
            condition_end = sig[peak_index:] > half_mag

        # Find the indices where the conditions are satisfied (before and after the peak)
        start_indices = np.flatnonzero(condition_start)
        end_indices = np.flatnonzero(condition_end)

        # If no valid start index is found, fallback to one step before the peak
        if start_indices.size == 0:
            left_index = peak_index - 1
            if left_index < 0:
                print(f"Warning: Left index {left_index} is out of bounds, returning NaN for left index.")
                left_index = np.nan
        else:
            # Otherwise, take the last valid index before the peak
            left_index = start_indices[-1]

        # If no valid end index is found, fallback to one step after the peak
        if end_indices.size == 0:
            right_index = peak_index + 1
            if right_index >= len(xs):
                print(f"Warning: Right index {right_index} is out of bounds, returning NaN for right index.")
                right_index = np.nan
        else:
            # Otherwise, take the first valid index after the peak
            right_index = peak_index + end_indices[0]

        # Return the indices of the start (left), end (right), and peak
        return left_index, right_index, peak_index

    except Exception as e:
        print(f"Error in extract_peak_indices: {str(e)}")
        traceback.print_exc()
        return np.nan, np.nan, np.nan  # Return NaN values in case of error









###OLD
# def extract_peak_indices(xs, sig, peak_params):
#     """
#     Finds the indices of the start, peak, and end points of a signal around a peak.

#     Parameters:
#     - xs (np.array): The x-axis values corresponding to the signal data.
#     - sig (np.array): The y-signal data as a 1D numpy array.
#     - peak_params (np.array): Array containing peak parameters where:
#       - peak_params[0]: x-value of the peak (center)
#       - peak_params[1]: magnitude (y-value) of the peak

#     Returns:
#     - left_index (int or float): Index where the signal crosses half the peak magnitude before the peak.
#     - peak_index (int): Index of the peak itself.
#     - right_index (int or float): Index where the signal crosses half the peak magnitude after the peak.
#     """
#     # Compute half magnitude
#     half_mag = peak_params[1] / 2

#     # Get the index of the peak in the signal
#     peak_index = np.argmin(np.abs(xs - peak_params[0]))

#     # Handle cases where the peak magnitude is zero
#     if peak_params[1] == 0:
#         return np.nan, np.nan, peak_index

#     # Handle positive and negative peaks separately based on peak_params[1]
#     if peak_params[1] > 0:
#         condition_start = sig[:peak_index] < half_mag
#         condition_end = sig[peak_index:] < half_mag
#     else:
#         condition_start = sig[:peak_index] > half_mag
#         condition_end = sig[peak_index:] > half_mag

#     # Vectorized search for indices
#     start_indices = np.flatnonzero(condition_start)
#     end_indices = np.flatnonzero(condition_end)

#     # Determine start and end indices, or assign NaN if not found
#     left_index = start_indices[-1] if start_indices.size > 0 else np.nan
#     right_index = peak_index + end_indices[0] if end_indices.size > 0 else np.nan

#     # Debugging information
#     print(f"Peak index: {peak_index}, Left index: {left_index}, Right index: {right_index}, Half magnitude: {half_mag}")

#     return left_index, right_index, peak_index
