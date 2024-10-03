import numpy as np


def find_peak_params(xs, sig, gaussian_params):
    """
    Creates peak parameters by finding the closest signal point to each Gaussian peak center.

    Parameters:
    - xs (np.array): The x-axis values corresponding to the signal data.
    - sig (np.array): The y-signal data as a 1D numpy array.
    - gaussian_params (np.array): Array of Gaussian parameters, where each entry is expected to contain
      at least three values: [center, amplitude, width].

    Returns:
    - peak_params (np.array): A 2D array where each row corresponds to a peak and contains:
      [closest x-value to Gaussian center, corresponding y-signal value, double the width parameter].
    """
    # Initialize the peak_params array to store the result
    peak_params = np.empty((len(gaussian_params), 3))

    # Vectorize the peak index calculation to avoid the loop
    centers = gaussian_params[:, 0]  # Extract Gaussian centers
    peak_indices = np.abs(xs[:, np.newaxis] - centers).argmin(axis=0)  # Find closest indices to Gaussian centers

    # Populate peak_params with x-value, y-signal value, and double the width parameter
    peak_params[:, 0] = xs[peak_indices]  # Closest x-values
    peak_params[:, 1] = sig[peak_indices]  # Corresponding y-signal values
    peak_params[:, 2] = gaussian_params[:, 2] * 2  # Double the width parameter

    return peak_params
