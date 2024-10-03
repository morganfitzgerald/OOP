# Functions for ECG Param
import numpy as np
from scipy.stats import norm

def compute_gauss_std(fwhm):
    """Compute the gaussian standard deviation, given the full-width half-max.
    Parameters
    ----------
    fwhm : float
        Full-width half-max.
    Returns
    -------
    float
        Calculated standard deviation of a gaussian.
    """

    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def gaussian_function(xs, *params):
    """Gaussian fitting function.
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.
    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """

    ys = np.zeros_like(xs)
    min_width = 1e-10  # Set a small minimum value for width to avoid division by zero

    for ii in range(0, len(params), 3):
        ctr, hgt, wid = params[ii : ii + 3]
        wid = max(wid, min_width)  # Ensure width is not zero
        ys = ys + hgt * np.exp(-((xs - ctr) ** 2) / (2 * wid**2))

    return ys

def skewed_gaussian_function(xs, *params):
    """Skewed gaussian fitting function.
    ***This function is borrowed from https://github.com/fooof-tools/fooof/commit/cfa8a2bec08dab742e9556f4aeee1698415d40ba***
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define the skewed gaussian function (center, height, width, alpha).
    Returns
    -------
    ys : 1d array
        Output values for skewed gaussian function.
    """

    ys = np.zeros_like(xs)
    for ii in range(0, len(params), 4):
        ctr, hgt, wid, alpha = params[ii : ii + 4]
        # Gaussian distribution
        ys = ys + gaussian_function(
            xs, ctr, hgt, wid
        )  # SUM of gaussians because we are fitting many at once
        # Skewed cumulative distribution function
        cdf = norm.cdf(alpha * ((xs - ctr) / wid))
        # Skew the gaussian
        ys = ys * cdf
        # Rescale height
        ys = (ys / np.max(ys)) * hgt
    return ys

def calc_r_squared(sig, fit):
    """Calculate the r-squared goodness of fit of the model, compared to the original data."""

    r_val = np.corrcoef(sig, fit)
    r_squared = r_val[0][1] ** 2
    return r_squared
