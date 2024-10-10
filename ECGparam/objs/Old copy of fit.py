####Old copy of fit.py 

This is running, the gaussian fitting looks good, but the shape params calc is all off. 
Struggling to troubleshoot this, so going to work on simplifying the structure more. 
Wanted to save this as a copy of what would run. 

###

import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, iirnotch, filtfilt
from scipy.optimize import curve_fit
import neurokit2 as nk
import os  # For handling file paths

# Import your custom functions
from ..analysis import calc_r_squared
from ..feature import calc_shape_params, calc_bounds, calc_fwhm, map_gaussian_to_signal_peaks
from ..processing import nk_peaks, epoch_cycles, find_extremum, compute_gauss_std, gaussian_function, extract_peak_indices, high_pass_filter, notch_filter
from ..utils import save_shape_params


###################################################################################################

class GaussianFitter:
    """Class to handle Gaussian fitting for ECG component extraction."""

    def __init__(self, bound_factor=0.1):
        """
        Initialize the GaussianFitter with a given bound factor for fitting bounds.
        :param bound_factor: Scaling factor for the fitting bounds
        """
        self.bound_factor = bound_factor

    def fit(self, xs, sig, guess):
        """
        Perform Gaussian fitting on the ECG signal.
        :param xs: Time indices of the signal
        :param sig: ECG signal to fit
        :param guess: Initial guesses for Gaussian parameters
        :return: Fitted Gaussian parameters
        """
        bounds = self._calculate_gaussian_bounds(guess)
        try:
            # Use curve fitting to find Gaussian parameters, limited to a max of 2500 iterations
            gaussian_params, _ = curve_fit(gaussian_function, xs, sig, p0=guess.flatten(), maxfev=2500, bounds=bounds)
            return gaussian_params.reshape((5, 3))  # Reshape to match the 5 ECG components (P, Q, R, S, T)
        except (ValueError, RuntimeError):
            return np.zeros((5, 3))  # Handle fitting failures by returning zeros

    def _calculate_gaussian_bounds(self, guess):
        """
        Calculate lower and upper bounds for the Gaussian parameters.
        :param guess: Initial guess for Gaussian parameters (centers, heights, widths)
        :return: Lower and upper bounds for the fitting process
        """
        centers = np.array([g[0] for g in guess])  # Extract center positions
        heights = np.array([g[1] for g in guess])  # Extract heights
        stds = np.array([g[2] for g in guess])     # Extract standard deviations

        # Compute bounds for each Gaussian component (P, Q, R, S, T)
        gaus_lo_bound, gaus_hi_bound = zip(
            calc_bounds(centers[0], heights[0], stds[0], self.bound_factor),
            calc_bounds(centers[1], heights[1], stds[1], self.bound_factor, flip_height=True),  # Handle negative peaks
            calc_bounds(centers[2], heights[2], stds[2], self.bound_factor),
            calc_bounds(centers[3], heights[3], stds[3], self.bound_factor, flip_height=True),  # Handle negative peaks
            calc_bounds(centers[4], heights[4], stds[4], self.bound_factor)
        )
        
        return np.concatenate(gaus_lo_bound), np.concatenate(gaus_hi_bound)

###################################################################################################
# ECG Processing Classes
###################################################################################################

class ECGparamBase:
    """Base class for handling common ECG processing functionalities."""

    def __init__(self, fs):
        """
        Initialize ECGParamBase with sampling frequency.
        :param fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.signal = None
        self.peaks = None

    def load_signal(self, signal):
        """
        Load and validate the ECG signal (must be 1-dimensional).
        :param signal: Input ECG signal
        """
        if signal.ndim != 1:
            raise ValueError("ECG signal must be 1-dimensional.")
        self.signal = signal

###################################################################################################

class ECGFeatureExtractor:
    """Class for extracting features from ECG signals using Gaussian fitting."""

    def __init__(self):
        """Initialize the feature extractor with a Gaussian fitter."""
        self.g_fitter = GaussianFitter()

    def extract_features(self, epochs_df):
        """
        Extract ECG component features from segmented ECG data (epochs).
        :param epochs_df: DataFrame containing segmented ECG data
        :return: Dictionary of extracted ECG features
        """
        components = ['p', 'q', 'r', 's', 't']
        metrics = ['center', 'height', 'width', 'fwhm', 'rise_time', 'decay_time', 'rise_decay_symmetry', 'sharpness']
        intervals = ['pr_interval', 'pr_segment', 'qrs_duration', 'st_segment', 'qt_interval', 'pp_interval', 'rr_interval']
        on_off_times = ['le_idx', 'ri_idx']  # Updated to use onset (le_idx) and offset (ri_idx)

        # Initialize an output dictionary to store extracted feature values
        ecg_output_dict = self._initialize_output_dict(epochs_df, components, metrics, intervals, on_off_times)
        previous_r_center, previous_p_center = None, None

        # List to collect shape parameters for all cycles
        shape_params_list = []  # Initialize shape_params_list here

        # Iterate through each ECG cycle (epoch) to extract features
        for cycle_idx, (_, one_cycle) in enumerate(epochs_df.groupby('cycle')):
            if one_cycle.empty or one_cycle['signal_y'].isnull().values.any():
                continue

            # Process the ECG cycle by extracting relevant signal portions
            xs, sig = self._get_signal_cycle(one_cycle)

            # Extract indices and characteristics for the P, Q, R, S, T components
            component_inds = self._extract_component_inds(sig, xs)
            
            if component_inds is None:
                continue  # Skip if no valid peaks are found

            # Fit Gaussian to the extracted components and map to the ECG signal
            self._map_gaussian_to_signal(xs, sig, ecg_output_dict, cycle_idx, component_inds, shape_params_list)

            # Calculate intervals (PR, QT, RR, etc.)
            self._calculate_intervals(ecg_output_dict, cycle_idx, previous_r_center, previous_p_center)

            # Update R and P centers for RR/PP interval calculations
            previous_r_center = component_inds['r'][2]  # R center
            previous_p_center = component_inds['p'][2]  # P center

        # After all cycles are processed, save the shape parameters
        save_shape_params(shape_params_list, '/Users/morganfitzgerald/Projects/ECG_tool_val/shape_params.csv')

        return ecg_output_dict

    

    def _initialize_output_dict(self, epochs_df, components, metrics, intervals, on_off_times):
        """
        Initialize the output dictionary for storing feature values.
        :param epochs_df: DataFrame containing segmented ECG data
        :param components: ECG components (P, Q, R, S, T)
        :param metrics: List of metrics to extract (center, height, etc.)
        :param intervals: List of interval metrics (PR interval, etc.)
        :param on_off_times: Onset and offset times for ECG components
        :return: Dictionary initialized with NaN values for each metric
        """
        ecg_output_dict = {f'{comp}_{metric}': [np.nan] * len(epochs_df) for comp in components for metric in metrics}
        ecg_output_dict.update({f'{comp}_{time}': [np.nan] * len(epochs_df) for comp in components for time in on_off_times})
        ecg_output_dict.update({interval: [np.nan] * len(epochs_df) for interval in intervals})
        ecg_output_dict["cycle"] = np.arange(len(epochs_df)).tolist()
        ecg_output_dict["r_squared"] = [np.nan] * len(epochs_df)
        return ecg_output_dict

    def _get_signal_cycle(self, one_cycle):
        """
        Retrieve the time indices and ECG signal for one cycle.
        :param one_cycle: DataFrame containing one ECG cycle
        :return: Time indices (xs) and signal (sig)
        """
        xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1] + 1)
        sig = detrend(one_cycle['signal_y'].values)  # Remove linear trends from the signal
        sig -= np.mean(sig[:25])  # Normalize the signal by subtracting the mean of the first 25 samples
        return xs, sig

    def _extract_component_inds(self, sig, xs):
        """
        Extract indices for ECG components (P, Q, R, S, T).
        :param sig: ECG signal
        :param xs: Time indices
        :return: Dictionary of indices and characteristics for P, Q, R, S, T components
        """
        # Identify the R-peak (highest point in the ECG cycle)
        r_ind = np.argmax(sig)
        r_height = sig[r_ind]
        r_center = xs[r_ind]

        # Find indices for onset and offset of R peak
        r_peak_params = [r_center, r_height]
        le_ind_r, ri_ind_r, _ = extract_peak_indices(xs, sig, r_peak_params)
        
        fwhm_r = calc_fwhm(le_ind_r, ri_ind_r, r_ind)
        if fwhm_r is None:
            return None  # Handle the case where no valid FWHM is found

        # Extract P, Q, S, T wave characteristics using find_extremum
        FWHM_Q_IND = 3
        FWHM_S_IND = 3

        q_min_ind = int(r_ind - (FWHM_Q_IND * fwhm_r))
        q_ind, q_height, q_center = find_extremum(xs, sig, q_min_ind, r_ind, mode='min')
        p_ind, p_height, p_center = find_extremum(xs, sig, 0, q_ind, mode='max')
        s_max_ind = int(r_ind + (FWHM_S_IND * fwhm_r))
        s_ind, s_height, s_center = find_extremum(xs, sig, r_ind, s_max_ind, mode='min')
        t_ind, t_height, t_center = find_extremum(xs, sig, s_ind, len(sig), mode='max')

        # Now use extract_peak_indices for P, Q, S, T
        p_peak_params = [p_center, p_height]
        le_ind_p, ri_ind_p, _ = extract_peak_indices(xs, sig, p_peak_params)

        q_peak_params = [q_center, q_height]
        le_ind_q, ri_ind_q, _ = extract_peak_indices(xs, sig, q_peak_params)

        s_peak_params = [s_center, s_height]
        le_ind_s, ri_ind_s, _ = extract_peak_indices(xs, sig, s_peak_params)

        t_peak_params = [t_center, t_height]
        le_ind_t, ri_ind_t, _ = extract_peak_indices(xs, sig, t_peak_params)

        # Return dictionary with indices and characteristics for all components
        return {
            'p': [p_ind, p_height, p_center, le_ind_p, ri_ind_p],
            'q': [q_ind, q_height, q_center, le_ind_q, ri_ind_q],
            'r': [r_ind, r_height, r_center, le_ind_r, ri_ind_r],
            's': [s_ind, s_height, s_center, le_ind_s, ri_ind_s],
            't': [t_ind, t_height, t_center, le_ind_t, ri_ind_t]
        }

   
    def _map_gaussian_to_signal(self, xs, sig, ecg_output_dict, cycle_idx, component_inds, shape_params_list):
        """
        Map Gaussian fit to the original ECG signal, fit the components, and compute shape parameters.
        
        :param xs: Time indices
        :param sig: ECG signal
        :param ecg_output_dict: Dictionary for storing features
        :param cycle_idx: Index of the current ECG cycle
        :param component_inds: Indices of ECG components (P, Q, R, S, T)
        :param shape_params_list: List to collect shape parameters for all cycles
        """
        
        # Create guesses for Gaussian fitting based on component indices
        def _create_guess(params):
            """
            Create an initial guess for Gaussian fitting based on ECG component parameters.
            Parameters:
            - params: Parameters (index, height, center) for a component.
            Returns: 
            - Initial guess for Gaussian fitting (center, height, standard deviation).
            """
            le_ind, ri_ind, center_index = params[0], params[1], params[2]
            
            # Calculate Full-Width Half-Maximum (FWHM) using the calc_fwhm function
            fwhm = calc_fwhm(le_ind, ri_ind, center_index)
            
            if fwhm is None:
                return [center_index, params[1], 1.0]  # Default values if FWHM cannot be computed
            
            # Compute standard deviation (sigma) for the Gaussian fit
            std_dev = compute_gauss_std(fwhm)
            
            return [params[2], params[1], std_dev]

        # Create initial guesses for each component (P, Q, R, S, T)
        guess = np.vstack([_create_guess(params) for params in component_inds.values()])

        # Perform the Gaussian fit
        gaussian_params = self.g_fitter.fit(xs, sig, guess)

        #debugging
        # Print gaussian_params for debugging
        print(f"Cycle {cycle_idx} Gaussian Params: {gaussian_params}")  

        # Store left and right indices (onset and offset)
        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            ecg_output_dict[f'{comp}_center'][cycle_idx] = gaussian_params[i, 0]
            ecg_output_dict[f'{comp}_height'][cycle_idx] = gaussian_params[i, 1]
            ecg_output_dict[f'{comp}_width'][cycle_idx] = gaussian_params[i, 2]

            # Use the new le_idx and ri_idx values from component_inds
            ecg_output_dict[f'{comp}_le_idx'][cycle_idx] = xs[component_inds[comp][3]] if not np.isnan(component_inds[comp][3]) else np.nan
            ecg_output_dict[f'{comp}_ri_idx'][cycle_idx] = xs[component_inds[comp][4]] if not np.isnan(component_inds[comp][4]) else np.nan

        # Print P and T peak indices for debugging
        print(f"Cycle {cycle_idx}:")
        print(f"  P peak indices: {component_inds['p']}")
        print(f"  T peak indices: {component_inds['t']}")
    
        # Map Gaussian fit to the real signal and calculate shape-related metrics
        fit = gaussian_function(xs, *gaussian_params.flatten())
        
        peak_params = map_gaussian_to_signal_peaks(xs, sig, gaussian_params)  # Map peaks to the signal
        shape_params = calc_shape_params(xs, sig, peak_params)  # Calculate shape-related parameters
   
        # Collect shape parameters for this cycle
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

        for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
            ecg_output_dict[f'{comp}_fwhm'][cycle_idx] = shape_params[i, 0]
            ecg_output_dict[f'{comp}_rise_time'][cycle_idx] = shape_params[i, 1]
            ecg_output_dict[f'{comp}_decay_time'][cycle_idx] = shape_params[i, 2]
            ecg_output_dict[f'{comp}_rise_decay_symmetry'][cycle_idx] = shape_params[i, 3]
            ecg_output_dict[f'{comp}_sharpness'][cycle_idx] = shape_params[i, 4]

        # Calculate goodness of fit (R-squared)
        ecg_output_dict['r_squared'][cycle_idx] = calc_r_squared(sig, fit)




    def _calculate_intervals(self, ecg_output_dict, cycle_idx, previous_r_center, previous_p_center):
        """
        Calculate intervals such as PR, QT, and RR intervals for ECG cycles.
        :param ecg_output_dict: Dictionary storing extracted features
        :param cycle_idx: Index of the current ECG cycle
        :param previous_r_center: Center of the R wave from the previous cycle
        :param previous_p_center: Center of the P wave from the previous cycle
        """
        # Define the intervals in terms of the left (onset) and right (offset) indices of the ECG components
        intervals = {
            'pr_interval': ('p_le_idx', 'q_le_idx'),       # Time between the start of P wave and the beginning of QRS wave
            'pr_segment': ('p_ri_idx', 'q_le_idx'),        # PR segment (similar to PR interval)
            'qrs_duration': ('q_le_idx', 's_ri_idx'),      # QRS complex duration
            'st_segment': ('s_ri_idx', 't_le_idx'),        # ST segment (end of S wave to beginning of T wave)
            'qt_interval': ('q_le_idx', 't_ri_idx')        # QT interval (start of Q wave to the end of T wave)
        }

        # Calculate each interval based on component onsets and offsets
        for interval, (on_key, off_key) in intervals.items():
            # Calculate the difference between the onset (le_idx) and offset (ri_idx) for each interval
            ecg_output_dict[interval][cycle_idx] = ecg_output_dict[off_key][cycle_idx] - ecg_output_dict[on_key][cycle_idx]

        # Calculate RR and PP intervals between consecutive cycles, based on the R and P wave centers
        if previous_r_center is not None and previous_p_center is not None:
            ecg_output_dict['rr_interval'][cycle_idx] = ecg_output_dict['r_center'][cycle_idx] - previous_r_center
            ecg_output_dict['pp_interval'][cycle_idx] = ecg_output_dict['p_center'][cycle_idx] - previous_p_center

###################################################################################################

import matplotlib.pyplot as plt

class ECGparam:
    """Main class to handle ECG parameter extraction, extending ECGParamBase."""
    
    def __init__(self, fs=1000):
        """
        Initialize the ECGparam class with default sampling frequency (fs).
        :param fs: Sampling frequency in Hz (default is 1000 Hz)
        """
        super().__init__(fs)  # Call the base class constructor
        self.feature_extractor = ECGFeatureExtractor()  # Initialize the feature extractor class
        self.features = None  # Placeholder for extracted features
        self.signal = None  # Placeholder for the input ECG signal
        # self.filtered_signal = None  # Placeholder for the high-pass filtered signal
        # self.ecg_notch = None  # Placeholder for the notch-filtered signal
        # self.ecg_clean = None  # Placeholder for the cleaned signal (NeuroKit2)
        # self.peaks = None  # Placeholder for detected R-peaks

    def process(self, signal):
        """
        Process the input ECG signal through various stages of filtering, peak detection, and feature extraction.
        :param signal: Input ECG signal to be processed
        :return: Extracted features dictionary
        """
        self.load_signal(signal)  # Load and validate the ECG signal
        
        # Apply high-pass filter using the function from the signal_processing module
        self.filtered_signal = high_pass_filter(self.signal, fs=self.fs)
        
        # Apply notch filter (default value of f0=50) using the function from the signal_processing module
        self.ecg_notch = notch_filter(self.filtered_signal, fs=self.fs)

        # Clean the signal using NeuroKit2's built-in cleaning method
        self.ecg_clean = nk.ecg_clean(self.ecg_notch, sampling_rate=self.fs)

        # Detect R-peaks using NeuroKit2
        p_peaks_nk, _, _ = nk_peaks(self.ecg_clean, sampling_rate=self.fs)
        self.peaks = p_peaks_nk  # Store detected R-peaks

        # Segment the ECG signal into cycles (epochs)
        epochs_df = epoch_cycles(self.peaks, self.ecg_notch, self.fs)

        # Extract features from the segmented ECG signal
        self.features = self.feature_extractor.extract_features(epochs_df)
        
        return self.features  # Return the dictionary of extracted features


    
    def save_features(self, file_path):
        """
        Save the extracted features (ecg_output DataFrame) to a CSV file.
        :param file_path: Path to save the CSV file.
        """
        if self.features is None:
            raise ValueError("No features extracted. Run the 'process' method first.")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert the feature dictionary to a DataFrame and save as CSV
        features_df = pd.DataFrame(self.features)
        features_df.to_csv(file_path, index=False)
        print(f"ECG features saved to {file_path}")





