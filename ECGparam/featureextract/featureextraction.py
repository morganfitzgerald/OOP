# External libraries
import numpy as np  # For array handling and mathematical operations
from scipy.signal import detrend  # For removing linear trends from signals
import logging

# Custom utility and processing functions 
from ..analysis import calc_r_squared  # Function to calculate the R-squared value
from ..feature import calc_shape_params, calc_fwhm, map_gaussian_to_signal_peaks  # Feature extraction utilities
from ..processing import find_extremum, gaussian_function, compute_gauss_std, extract_peak_indices, GaussianFitter  
# from ECGparam.utlils import save_shape_params  # Utility for saving shape parameters to a CSV file


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
    on_off_times = ['le_idx', 'ri_idx']

    # Initialize the output dictionary
    ecg_output_dict = _initialize_output_dict(epochs_df, components, metrics, intervals, on_off_times)
    shape_params_list = []
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
            _map_gaussian_to_signal(xs, sig, ecg_output_dict, cycle_idx, component_inds, shape_params_list)

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
        xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1] + 1)
        sig = detrend(one_cycle['signal_y'].values)
        sig -= np.mean(sig[:25])  # Normalize the signal
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

        # Find onset and offset of R peak
        r_peak_params = [r_center, r_height]
        le_ind_r, ri_ind_r, _ = extract_peak_indices(xs, sig, r_peak_params)
        fwhm_r = calc_fwhm(le_ind_r, ri_ind_r, r_ind)
        if fwhm_r is None:
            return None

        # Extract other wave characteristics (Q, P, S, T)
        q_ind, q_height, q_center = find_extremum(xs, sig, int(r_ind - 3 * fwhm_r), r_ind, mode='min')
        p_ind, p_height, p_center = find_extremum(xs, sig, 0, q_ind, mode='max')
        s_ind, s_height, s_center = find_extremum(xs, sig, r_ind, int(r_ind + 3 * fwhm_r), mode='min')
        t_ind, t_height, t_center = find_extremum(xs, sig, s_ind, len(sig), mode='max')

        # Return dictionary of indices
        return {
            'p': [p_ind, p_height, p_center, *extract_peak_indices(xs, sig, [p_center, p_height])],
            'q': [q_ind, q_height, q_center, *extract_peak_indices(xs, sig, [q_center, q_height])],
            'r': [r_ind, r_height, r_center, le_ind_r, ri_ind_r],
            's': [s_ind, s_height, s_center, *extract_peak_indices(xs, sig, [s_center, s_height])],
            't': [t_ind, t_height, t_center, *extract_peak_indices(xs, sig, [t_center, t_height])]
        }

    except Exception as e:
        logger.error(f"Error extracting component indices: {e}", exc_info=True)
        return None


# Create guesses for Gaussian fitting based on component indices
def _create_guess(params):
    """
    Create an initial guess for Gaussian fitting based on ECG component parameters.
    
    Parameters:
    -----------
    params : list
        Parameters (index, height, center) for a component.

    Returns:
    --------
    guess : list
        Initial guess for Gaussian fitting (center, height, std).
    """
    try:
        le_ind, ri_ind, center = params[3], params[4], params[2]
        fwhm = calc_fwhm(le_ind, ri_ind, center)
        if fwhm is None:
            logger.warning(f"No valid FWHM found for center: {center}. Using default guess.")
            return [center, params[1], 1.0]  # Default guess
        std_dev = compute_gauss_std(fwhm)
        return [center, params[1], std_dev]
    except Exception as e:
        logger.error(f"Error creating Gaussian guess: {e}", exc_info=True)
        raise


def _map_gaussian_to_signal(xs, sig, ecg_output_dict, cycle_idx, component_inds, shape_params_list):
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
    shape_params_list : list
        List to store shape parameters for each cycle.
    """
    try:
        logger.info(f"Mapping Gaussian to signal for cycle {cycle_idx}.")
        
        # Create guesses for Gaussian fitting based on component indices
        guess = np.vstack([_create_guess(params) for params in component_inds.values()])
        #logger.debug(f"Initial guess for Gaussian fitting (cycle {cycle_idx}): {guess}")

        # Perform the Gaussian fit
        gaussian_params = g_fitter.fit(xs, sig, guess)

        # Store Gaussian fitting results in the output dictionary
        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            ecg_output_dict[f'{comp}_center'][cycle_idx] = gaussian_params[i, 0]
            ecg_output_dict[f'{comp}_height'][cycle_idx] = gaussian_params[i, 1]
            ecg_output_dict[f'{comp}_width'][cycle_idx] = gaussian_params[i, 2]
            ecg_output_dict[f'{comp}_le_idx'][cycle_idx] = component_inds[comp][3]
            ecg_output_dict[f'{comp}_ri_idx'][cycle_idx] = component_inds[comp][4]

        # Fit Gaussian model to the signal
        fit = gaussian_function(xs, *gaussian_params)
        #logger.debug(f"Gaussian fit (cycle {cycle_idx}): {fit}")

        # Calculate and store shape parameters
        peak_params = map_gaussian_to_signal_peaks(xs, sig, gaussian_params)
        shape_params = calc_shape_params(xs, sig, peak_params)

        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            shape_params_list.append({
                'cycle': cycle_idx,
                'component': comp,
                'fwhm': shape_params[i, 0],
                'rise_time': shape_params[i, 1],
                'decay_time': shape_params[i, 2],
                'rise_decay_symmetry': shape_params[i, 3],
                'sharpness': shape_params[i, 4]
            })

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
