import numpy as np


# Define the sharpness calculation function using the derivative, normalized by peak amplitude
def calc_sharpness_deriv(sig, peak_index, window_size=15):
    # Extract a segment of the signal centered on the peak
    cropped_peak_sig = sig[max(0, peak_index - window_size): min(len(sig) - 1, peak_index + window_size + 1)]
    # Compute the first derivative of the segment
    derivative = np.abs(np.diff(cropped_peak_sig))
    # Calculate the mean of the absolute values of the derivative
    sharpness_deriv = np.mean(derivative)
    return sharpness_deriv
