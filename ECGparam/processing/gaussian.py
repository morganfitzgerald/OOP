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
    """
    Compute the sum of multiple Gaussians given their parameters.

    Parameters:
    -----------
    xs : array-like
        The input x-values where the Gaussian will be computed.
    params : tuple
        The Gaussian parameters: (ctr1, hgt1, wid1, ctr2, hgt2, wid2, ...)

    Returns:
    --------
    ys : array-like
        The computed y-values for the sum of all Gaussians.
    """
    ys = np.zeros_like(xs)  # Initialize the output array

    num_gaussians = len(params) // 3  # Each Gaussian has 3 parameters: center, height, width

    for i in range(num_gaussians):
        ctr = params[3 * i]
        hgt = params[3 * i + 1]
        wid = np.maximum(params[3 * i + 2], 1e-6)  # Ensure width is not zero

        # Add the current Gaussian to the result
        ys += hgt * np.exp(-((xs - ctr) ** 2) / (2 * wid ** 2))

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
