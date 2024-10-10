import numpy as np
from ..processing import extract_peak_indices 


# Define the sharpness calculation function using the derivative, normalized by peak amplitude
def calc_sharpness_deriv(sig, peak_index, window_size=15):
    # Extract a segment of the signal centered on the peak
    cropped_peak_sig = sig[max(0, peak_index - window_size): min(len(sig) - 1, peak_index + window_size + 1)]
    # Compute the first derivative of the segment
    derivative = np.abs(np.diff(cropped_peak_sig))
    # Calculate the mean of the absolute values of the derivative
    sharpness_deriv = np.mean(derivative)
    return sharpness_deriv


def calc_shape_params(xs, sig, peak_params):
    """
    Calculate the shape parameters (FWHM, rise-time, decay-time, 
    rise-decay symmetry, and sharpness) for each ECG peak component.
    
    Parameters:
    -----------
    xs : np.array
        The x-values of the signal (e.g., time or sample index).
    sig : np.array
        The ECG signal values corresponding to the x-values.
    peak_params : list
        A list of peak parameter guesses for different ECG components 
        (e.g., P, Q, R, S, T).

    Returns:
    --------
    shape_params : np.ndarray
        A 2D array where each row contains the computed shape parameters
        (FWHM, rise-time, decay-time, rise-decay symmetry, sharpness) for
        each peak component.
    peak_bounds : np.ndarray
        A 2D array where each row contains the start, peak, and end indices 
        for each peak component.
    """
    
    # Initialize arrays to store shape parameters and peak indices
    shape_params = np.empty((len(peak_params), 5))  
    peak_bounds = np.empty((len(peak_params), 3))

    # Iterate over each peak in the list of peak parameters
    for ii, peak in enumerate(peak_params):
        
        # Get the start, peak, and end indices of the current peak
        start_index, end_index, peak_index = extract_peak_indices(xs, sig, peak)

        # If any of the indices are NaN, skip to the next iteration
        if np.isnan(start_index) or np.isnan(end_index):
            shape_params[ii] = [np.nan] * 5  # Set all parameters to NaN
            peak_bounds[ii] = [np.nan] * 3  # Set all indices to NaN
            continue

        # Compute the full width at half maximum (FWHM)
        fwhm = xs[end_index] - xs[start_index]
        
        # Calculate rise-time
        rise_time = xs[peak_index] - xs[start_index]
        
        # Calculate decay-time
        decay_time = xs[end_index] - xs[peak_index]

        # Compute rise-decay symmetry
        rise_decay_symmetry = rise_time / fwhm if fwhm != 0 else np.nan

        # Compute sharpness using the derivative method
        sharpness_deriv = np.abs(calc_sharpness_deriv(sig, peak_index)) / sig[peak_index]

        # Store the calculated parameters
        shape_params[ii] = [fwhm, rise_time, decay_time, rise_decay_symmetry, sharpness_deriv]
        
    # Return the shape parameters and peak bounds
    return shape_params
