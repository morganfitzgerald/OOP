import numpy as np
import logging
from scipy.optimize import curve_fit

# Import your external calc_bounds function
from .bounds import calc_bounds
from .gaussian import gaussian_function

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
        Fit multiple Gaussians to the ECG signal.

        Parameters
        ----------
        xs : np.array
            X-axis values (time indices).
        sig : np.array
            The ECG signal to fit the Gaussian to.
        guess : np.array
            The initial guess for Gaussian parameters (center, height, width) for each component.
        
        Returns
        -------
        gaussian_params : np.array
            Fitted Gaussian parameters (center, height, width) for each component.
        """
        try:
            # Define bounds based on the guess values
            centers = np.array([guess[i][0] for i in range(5)])  # 5 components (P, Q, R, S, T)
            heights = np.array([guess[i][1] for i in range(5)])
            stds = np.array([guess[i][2] for i in range(5)])

            # Use external calc_bounds function for bounds calculation
            gaus_lo_bound, gaus_hi_bound = zip(
                calc_bounds(centers[0], heights[0], stds[0], self.bound_factor),
                calc_bounds(centers[1], heights[1], stds[1], self.bound_factor, flip_height=True),
                calc_bounds(centers[2], heights[2], stds[2], self.bound_factor),
                calc_bounds(centers[3], heights[3], stds[3], self.bound_factor, flip_height=True),
                calc_bounds(centers[4], heights[4], stds[4], self.bound_factor)
            )

            # Log the bounds to catch potential issues
            logger.debug(f"Gaussian Parameters Guess: {guess}")

            logger.debug(f"Lower bounds: {gaus_lo_bound}")
            logger.debug(f"Upper bounds: {gaus_hi_bound}")

            # Concatenate the bounds and reshape them
            gaus_param_bounds = (np.concatenate(gaus_lo_bound), np.concatenate(gaus_hi_bound))
            guess_flat = guess.flatten()

            # Ensure the lower bound is strictly less than the upper bound
            if np.any(np.array(gaus_param_bounds[0]) >= np.array(gaus_param_bounds[1])):
                raise ValueError("Invalid bounds: lower bound must be strictly less than upper bound.")

            # Perform Gaussian fitting with curve_fit
            gaussian_params, _ = curve_fit(gaussian_function, xs, sig, p0=guess_flat, maxfev=2500, bounds=gaus_param_bounds)

            # Reshape Gaussian parameters for 5 components (P, Q, R, S, T)
            gaussian_params_reshape = gaussian_params.reshape((5, 3))

            return gaussian_params_reshape

        except Exception as e:
            raise ValueError(f"Error in Gaussian fitting: {e}")
