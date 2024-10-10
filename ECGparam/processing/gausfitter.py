import numpy as np
import logging
from scipy.optimize import curve_fit
from ..feature import calc_bounds  # Assuming calc_bounds is already in feature
from ..processing import gaussian_function

logging.basicConfig(level=logging.DEBUG)  # Or use INFO for less verbosity
logger = logging.getLogger(__name__)


class GaussianFitter:
    """Class to handle Gaussian fitting for ECG component extraction."""

    def __init__(self, bound_factor=0.1):
        """
        Initialize the GaussianFitter with a given bound factor for fitting bounds.
        :param bound_factor: Scaling factor for the fitting bounds.
        """
        self.bound_factor = bound_factor

    def fit(self, xs, sig, guess):
        """
        Perform Gaussian fitting on the ECG signal.
        :param xs: Time indices of the signal.
        :param sig: ECG signal to fit.
        :param guess: Initial guesses for Gaussian parameters (center, height, std_dev).
        :return: Fitted Gaussian parameters as a reshaped array.
        """
        self._validate_inputs(xs, sig, guess)
        #logger.debug(f"Initial guess for Gaussian fitting: {guess}")

        bounds = self._calculate_gaussian_bounds(guess)
        #logger.debug(f"Bounds for curve fitting: {bounds}")

        try:
            # Perform the curve fitting with a maximum of 2500 iterations
            gaussian_params, _ = curve_fit(gaussian_function, xs, sig, p0=guess.flatten(),
                                           maxfev=2500, bounds=bounds)
            return gaussian_params.reshape((5, 3))  # Reshape based on the number of components
        except (ValueError, RuntimeError):
            return np.zeros((len(guess), 3))  # Return zeros if fitting fails

    def _calculate_gaussian_bounds(self, guess):
        """
        Calculate lower and upper bounds for the Gaussian parameters.
        :param guess: Initial guess for Gaussian parameters (centers, heights, std_devs).
        :return: Tuple of lower and upper bounds.
        """
        centers = np.array([g[0] for g in guess])  # Extract center positions
        heights = np.array([g[1] for g in guess])  # Extract heights
        stds = np.array([g[2] for g in guess])     # Extract standard deviations

        bounds_lo, bounds_hi = [], []
        for i, (center, height, std) in enumerate(zip(centers, heights, stds)):
            flip_height = i in [1, 3]  # Assuming Q and S waves are negative
            lo, hi = calc_bounds(center, height, std, self.bound_factor, flip_height=flip_height)
            bounds_lo.append(lo)
            bounds_hi.append(hi)

        return np.concatenate(bounds_lo), np.concatenate(bounds_hi)

    def _validate_inputs(self, xs, sig, guess):
        """
        Validate inputs for Gaussian fitting.
        :param xs: Time indices.
        :param sig: ECG signal.
        :param guess: Initial guesses for Gaussian parameters.
        :raises ValueError: If any input is invalid.
        """
        if xs.ndim != 1 or sig.ndim != 1:
            raise ValueError("xs and sig must be 1-dimensional arrays.")
        if len(xs) != len(sig):
            raise ValueError("Time indices (xs) and signal (sig) must have the same length.")
        if len(guess) == 0 or len(guess[0]) != 3:
            raise ValueError("Each guess must be a list or array of 3 values (center, height, std).")
