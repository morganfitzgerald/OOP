import numpy as np

#Creat peak params
def create_peak_params(xs, sig, gaussian_params):
    peak_params = np.empty((len(gaussian_params), 3))

    for ii, peak in enumerate(gaussian_params):

        # find the index of the signal the the time closest to the Gaussian center
        peak_index = np.argmin(np.abs(xs - peak[0]))

        # Collect peak parameter data
        peak_params[ii] = [xs[peak_index], sig[peak_index], peak[2] * 2]

    return peak_params

