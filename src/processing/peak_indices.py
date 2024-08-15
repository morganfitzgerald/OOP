import numpy as np

##Find peak indicies
def get_peak_indices(xs, sig, peak_params):
    # compute half magnitude
    half_mag = peak_params[1] / 2

    # get index of peak
    peak_index = np.argmin(np.abs(xs - peak_params[0]))

    # find the index closest to the peak that crosses the half magnitude
    try:
        if sig[peak_index]>0:
            start_index = np.argwhere((sig[:peak_index] - half_mag)<0)[-1][0]
            end_index = peak_index + np.argwhere(-(sig[peak_index:] - half_mag)>0)[0][0]
        else: # flip the logic if the peak is negative
            start_index = np.argwhere((sig[:peak_index] - half_mag)>0)[-1][0]
            end_index = peak_index + np.argwhere(-(sig[peak_index:] - half_mag)<0)[0][0]
    except IndexError:
        # if the half magnitude is not crossed, set the start and end indices to NaN
        start_index = np.nan
        end_index = np.nan 

    return start_index, peak_index, end_index