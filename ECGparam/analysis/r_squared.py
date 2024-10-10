import numpy as np

def calc_r_squared(sig, fit):
    """Calculate the r-squared goodness of fit of the model, compared to the original data."""

    r_val = np.corrcoef(sig, fit)
    r_squared = r_val[0][1] ** 2
    return r_squared