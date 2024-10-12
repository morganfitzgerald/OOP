# External libraries
import numpy as np  # For array handling and mathematical operations
from scipy.signal import detrend  # For removing linear trends from signals
import logging

# Custom utility and processing functions 
from ECGparam.processing.shape import calc_shape_params
from ECGparam.processing.extremum import find_extremum
from ECGparam.processing.gaussian import gaussian_function, compute_gauss_std
from ECGparam.processing.rindex import extract_r_peak_indecies
from ECGparam.processing.gausfitter import GaussianFitter
from ECGparam.processing.fwhm import calc_fwhm
from ECGparam.processing.mappeak import map_gaussian_to_signal_peaks
from ECGparam.processing.peakindex import extract_peak_indecies

from ECGparam.analysis.r_squared import calc_r_squared


# Setup logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more detailed logging
logger = logging.getLogger(__name__)


# Instantiate the Gaussian fitter class
g_fitter = GaussianFitter()


def extract_features(epochs_df, fs):
    """
    Extract ECG features from segmented ECG data (epochs).
    
    Parameters
    ----------
    epochs_df : pd.DataFrame
        DataFrame containing segmented ECG data.
    fs : float
        Sampling frequency.
    
    Returns
    -------
    ecg_output_dict : dict
        Dictionary containing extracted ECG features.
    """
    logger.info("Starting feature extraction process.")
    
    components = ['p', 'q', 'r', 's', 't']
    metrics = ['center', 'height', 'width', 'fwhm', 'rise_time', 'decay_time', 'rise_decay_symmetry', 'sharpness']
    intervals = ['pr_interval', 'pr_segment', 'qrs_duration', 'st_segment', 'qt_interval', 'pp_interval', 'rr_interval']
    on_off_times = ['on, off']

    # Initialize the output dictionary
    ecg_output_dict = _initialize_output_dict(epochs_df, components, metrics, intervals, on_off_times)

    previous_r_center, previous_p_center = None, None

    # Loop through each ECG cycle to extract features
    for cycle_idx, (_, one_cycle) in enumerate(epochs_df.groupby('cycle')):
        try:
            if one_cycle.empty or one_cycle['signal_y'].isnull().values.any():
                logger.warning(f"Skipping cycle {cycle_idx} due to missing data.")
                continue

            xs, sig = _get_signal_cycle(one_cycle)

            # Extract indices for P, Q, R, S, T components
            component_inds = _extract_component_inds(sig, xs)
            if component_inds is None:
                logger.warning(f"Skipping cycle {cycle_idx} as no valid peaks were found.")
                continue

            # Fit Gaussian and map it to the signal
            _map_gaussian_to_signal(xs, sig, ecg_output_dict, cycle_idx, component_inds)

            # Calculate intervals (PR, QT, RR, etc.)
            _calculate_intervals(ecg_output_dict, cycle_idx, previous_r_center, previous_p_center)

            previous_r_center = component_inds['r'][2]
            previous_p_center = component_inds['p'][2]

        except Exception as e:
            logger.error(f"Error processing cycle {cycle_idx}: {e}", exc_info=True)

    logger.info("Feature extraction completed.")
    return ecg_output_dict


def _initialize_output_dict(epochs_df, components, metrics, intervals, on_off_times):
    """ Initialize output dictionary for storing feature values. """
    return {
        **{f'{comp}_{metric}': [np.nan] * len(epochs_df) for comp in components for metric in metrics},
        **{f'{comp}_{time}': [np.nan] * len(epochs_df) for comp in components for time in on_off_times},
        **{interval: [np.nan] * len(epochs_df) for interval in intervals},
        "cycle": np.arange(len(epochs_df)).tolist(),
        "r_squared": [np.nan] * len(epochs_df)
    }


def _get_signal_cycle(one_cycle):
    """ Retrieve time indices and ECG signal for one cycle. """
    try:
        # X values and Y values with offset correction
        xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1] + 1)
        sig = np.asarray(one_cycle['signal_y'])
        sig_flat = detrend(sig)
        sig = sig_flat - np.mean(sig_flat[0:25])
        return xs, sig
    except Exception as e:
        logger.error(f"Error extracting signal cycle: {e}", exc_info=True)
        raise


def _extract_component_inds(sig, xs):
    """ Extract indices for ECG components (P, Q, R, S, T). """
    try:
        # Identify the R-peak
        r_ind = np.argmax(sig)
        r_height = sig[r_ind]
        r_center = xs[r_ind]

        # Find indices for onset and offset of R peak
        le_ind_r, ri_ind_r = extract_r_peak_indecies(sig, r_ind, r_height)
        fwhm_r = calc_fwhm(le_ind_r, ri_ind_r, r_ind)
        if fwhm_r is None:
            return None
        
        # Finding P, Q, S, T components
        FWHM_Q_IND = 3
        FWHM_S_IND = 3
        q_min_ind = int(r_ind - (FWHM_Q_IND * fwhm_r))
        q_ind, q_height, q_center = find_extremum(sig, xs, q_min_ind, r_ind, mode='min')
        p_ind, p_height, p_center = find_extremum(sig, xs, 0, q_ind, mode='max')
        s_max_ind = int(r_ind + (FWHM_S_IND * fwhm_r))
        s_ind, s_height, s_center = find_extremum(sig, xs, r_ind, s_max_ind, mode='min')
        t_ind, t_height, t_center = find_extremum(sig, xs, s_ind, len(sig), mode='max')

        # Organizing component information
        return {
            'p': [p_ind, p_height, p_center],
            'q': [q_ind, q_height, q_center],
            'r': [r_ind, r_height, r_center],
            's': [s_ind, s_height, s_center],
            't': [t_ind, t_height, t_center]
        }

        # # Extract other wave characteristics (Q, P, S, T)
        # q_ind, q_height, q_center = find_extremum(xs, sig, int(r_ind - 3 * fwhm_r), r_ind, mode='min')
        # p_ind, p_height, p_center = find_extremum(xs, sig, 0, q_ind, mode='max')
        # s_ind, s_height, s_center = find_extremum(xs, sig, r_ind, int(r_ind + 3 * fwhm_r), mode='min')
        # t_ind, t_height, t_center = find_extremum(xs, sig, s_ind, len(sig), mode='max')

        # # Return dictionary of indices
        # return {
        #     'p': [p_ind, p_height, p_center, *extract_peak_indices(xs, sig, [p_center, p_height])],
        #     'q': [q_ind, q_height, q_center, *extract_peak_indices(xs, sig, [q_center, q_height])],
        #     'r': [r_ind, r_height, r_center, le_ind_r, ri_ind_r],
        #     's': [s_ind, s_height, s_center, *extract_peak_indices(xs, sig, [s_center, s_height])],
        #     't': [t_ind, t_height, t_center, *extract_peak_indices(xs, sig, [t_center, t_height])]
        # }

    except Exception as e:
        logger.error(f"Error extracting component indices: {e}", exc_info=True)
        return None


# Create guesses for Gaussian fitting based on component indices
def _create_guess(params, sig):
    """
    Create an initial guess for Gaussian fitting based on ECG component parameters.
    
    Parameters:
    -----------
    params : list
        Parameters (index, height, center) for a component.
    xs : np.array
        X-axis (time indices) for the ECG signal.
    sig : np.array
        The ECG signal.

    Returns:
    --------
    guess : list
        Initial guess for Gaussian fitting (center, height, std).
    """
    try:
        # Extract peak index and height from params
        peak_index, peak_height, center = params[0], params[1], params[2]
        
        # Find onset and offset of the peak using peak index and height
        onset, offset = extract_peak_indecies(sig, peak_index, peak_height)

        # Check for valid onset and offset, otherwise log a warning
        if onset is None or offset is None:
            logger.warning(f"No valid onset/offset found for peak at index {peak_index}. Using default guess.")
            return [center, peak_height, 1.0]  # Default guess

        # Calculate the short side of the peak to determine FWHM
        short_side = min(abs(peak_index - onset), abs(offset - peak_index))
        fwhm = short_side * 2  # Full-width half-maximum is twice the short side
        
        # Calculate the standard deviation (std_dev) based on FWHM
        std_dev = compute_gauss_std(fwhm)

        # Return the guess as a list: [center, peak height, std_dev]
        return [center, peak_height, std_dev]

    except Exception as e:
        logger.error(f"Error creating Gaussian guess: {e}", exc_info=True)
        raise



def _map_gaussian_to_signal(xs, sig, ecg_output_dict, cycle_idx, component_inds):
    """ 
    Fit Gaussian to the ECG signal and store the results in the feature dictionary.

    Parameters:
    -----------
    xs : 1D array-like
        Time indices for the signal.
    sig : 1D array-like
        The ECG signal to fit the Gaussian to.
    ecg_output_dict : dict
        Dictionary where extracted feature values are stored.
    cycle_idx : int
        Index of the current ECG cycle.
    component_inds : dict
        Dictionary of component indices (P, Q, R, S, T).
    """
    try:
        logger.info(f"Mapping Gaussian to signal for cycle {cycle_idx}.")
        
        # Create guesses for Gaussian fitting based on component indices
        guess = np.vstack([_create_guess(params, sig) for params in component_inds.values()])
        
        #logger.debug(f"Initial guess for Gaussian fitting (cycle {cycle_idx}): {guess}")

        # Perform the Gaussian fit
        gaussian_params = g_fitter.fit(xs, sig, guess)

        # Store Gaussian fitting results in the output dictionary
        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            ecg_output_dict[f'{comp}_center'][cycle_idx] = gaussian_params[i, 0]
            ecg_output_dict[f'{comp}_height'][cycle_idx] = gaussian_params[i, 1]
            ecg_output_dict[f'{comp}_width'][cycle_idx] = gaussian_params[i, 2]
            
        # Fit Gaussian model to the signal
        fit = gaussian_function(xs, *gaussian_params)
        #logger.debug(f"Gaussian fit (cycle {cycle_idx}): {fit}")

        # Calculate and store shape parameters
        peak_params = map_gaussian_to_signal_peaks(xs, sig, gaussian_params)
        shape_params = calc_shape_params(xs, sig, peak_params)

        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            # Store calculated shape parameters in the output dictionary
            ecg_output_dict[f'{comp}_fwhm'][cycle_idx] = shape_params[i, 0]
            ecg_output_dict[f'{comp}_rise_time'][cycle_idx] = shape_params[i, 1]
            ecg_output_dict[f'{comp}_decay_time'][cycle_idx] = shape_params[i, 2]
            ecg_output_dict[f'{comp}_rise_decay_symmetry'][cycle_idx] = shape_params[i, 3]
            ecg_output_dict[f'{comp}_sharpness'][cycle_idx] = shape_params[i, 4]
            

        # Calculate goodness of fit (R-squared) for the Gaussian fit
        ecg_output_dict['r_squared'][cycle_idx] = calc_r_squared(sig, fit)

    except Exception as e:
        logger.error(f"Error mapping Gaussian to signal for cycle {cycle_idx}: {e}", exc_info=True)


def _calculate_intervals(ecg_output_dict, cycle_idx, previous_r_center, previous_p_center):
    """
    Calculate PR, QT, RR, and other intervals for ECG cycles and store the results in the output dictionary.

    Parameters:
    -----------
    ecg_output_dict : dict
        Dictionary where extracted feature values are stored. It contains the times (onset and offset) for each component (P, Q, R, S, T).
    cycle_idx : int
        Index of the current ECG cycle.
    previous_r_center : float or None
        The center (time index) of the R-peak from the previous cycle. This is used to calculate the RR interval.
    previous_p_center : float or None
        The center (time index) of the P-wave from the previous cycle. This is used to calculate the PP interval.

    Returns:
    --------
    None
        Modifies `ecg_output_dict` in place by adding the calculated intervals.
    """
    try:
        # Define intervals based on the left (onset) and right (offset) indices of ECG components
        intervals = {
            'pr_interval': ('p_le_idx', 'q_le_idx'),  # Time between the onset of P-wave and the onset of Q-wave
            'pr_segment': ('p_ri_idx', 'q_le_idx'),   # Time between the offset of P-wave and the onset of Q-wave
            'qrs_duration': ('q_le_idx', 's_ri_idx'), # Duration of the QRS complex (Q-wave onset to S-wave offset)
            'st_segment': ('s_ri_idx', 't_le_idx'),   # Time between the offset of S-wave and the onset of T-wave
            'qt_interval': ('q_le_idx', 't_ri_idx')   # Time between the onset of Q-wave and the offset of T-wave
        }

        # Calculate the intervals for each cycle and store in the output dictionary
        for interval, (on_key, off_key) in intervals.items():
            # Subtract the onset time (on_key) from the offset time (off_key) to get the interval duration
            ecg_output_dict[interval][cycle_idx] = ecg_output_dict[off_key][cycle_idx] - ecg_output_dict[on_key][cycle_idx]

        # Calculate the RR interval (time between consecutive R peaks) if the previous R-peak exists
        if previous_r_center is not None:
            ecg_output_dict['rr_interval'][cycle_idx] = ecg_output_dict['r_center'][cycle_idx] - previous_r_center

        # Calculate the PP interval (time between consecutive P waves) if the previous P-wave exists
        if previous_p_center is not None:
            ecg_output_dict['pp_interval'][cycle_idx] = ecg_output_dict['p_center'][cycle_idx] - previous_p_center

    # Catch any exceptions that occur during the calculation and log the error with the cycle index
    except Exception as e:
        logger.error(f"Error calculating intervals for cycle {cycle_idx}: {e}", exc_info=True)
