#!/usr/bin/env python
# coding: utf-8

# # ECG Parameterization Toolbox

# ### Imports
import sys
print(sys.executable)
sys.path.append('..')

import pandas as pd
import neurokit2 as nk
import numpy as np
import os
import traceback  # For detailed error reporting

from scipy.signal import detrend, butter, iirnotch, filtfilt
from scipy.optimize import curve_fit

# Custom imports
from src.utils import create_subject_file_mapping, extract_data, extract_metadata, simulate_ecg_sig, epoch_cycles

from src.template import match_ecg_template

from src.feature import nk_peaks, find_peak_params, calc_shape_params, calc_bounds, extract_peak_indices

from src.analysis import find_extremum, calc_fwhm, find_peak_boundaries, calc_r_squared

from src.gaussian import gaussian_function, compute_gauss_std


# ### Global Attributes
fs = 1000
sampling_rate = fs
CROP_MIN = 1000
CROP_MAX = 3000
WINDOW_LENGTH = 5000

FWHM_Q_IND = 3
FWHM_S_IND = 3

# Function to process a single subject
def process_subject(SUB_NUM, dat_path, hea_path, results_dir):
    print(f'Processing subject {SUB_NUM}')
    try:
        sigs, metadata = extract_data(
            dat_path, hea_path, raw_dtype='int16'
        )

        # Iterate through signals
        for ind in range(metadata['n_sigs']):
            signal_name = metadata[f'sig{str(ind).zfill(2)}']['signal_name']

        # Select the template ECG signal
        template_ecg = simulate_ecg_sig(duration=5, sampling_rate = 1000, heart_rate=80, amplitude_factor=7, normalize=False)

        # List to store normalized signals
        normalized_signals = []

        # Normalize each signal
        for ind in range(metadata['n_sigs']):
            signal_name = metadata[f'sig{str(ind).zfill(2)}']['signal_name']
            if signal_name == 'NIBP':  # Skip non-ECG signals
                continue

            # Crop and normalize the signal
            cropped_signal = sigs[ind][CROP_MIN:CROP_MAX]
            normalized_signal = (cropped_signal - np.mean(cropped_signal)) / np.std(cropped_signal)
            normalized_signals.append(normalized_signal)

        # Find the most similar signal to the template using cross-correlation
        selected_signal, selected_signal_name, selected_signal_index = match_ecg_template(template_ecg, normalized_signals, metadata)

        # Now 'ecg' contains the selected signal
        ecg = sigs[selected_signal_index]
        fs = metadata['fs']  # Update sampling rate

        # Filtering
        cutoff_frequency = 0.05
        order = 4
        b, a = butter(order, cutoff_frequency, btype='high', analog=False, fs=fs)
        ecg_hp = filtfilt(b, a, ecg)

        # Remove line noise with a notch filter
        f0 = 50  # Line noise frequency (50 Hz)
        quality_factor = 30
        b, a = iirnotch(f0, quality_factor, fs)
        ecg_notch = filtfilt(b, a, ecg_hp)

        # Clean ECG using NeuroKit
        ecg_clean_nk = nk.ecg_clean(ecg, sampling_rate=fs)
        p_peaks_nk, _, _ = nk_peaks(ecg_clean_nk, sampling_rate)

        # Epoch cycles
        epochs_df, result_r_latencies = epoch_cycles(p_peaks_nk, ecg_notch, fs, SUB_NUM, PLOT=False, SAVE=False)

        # PARAMETERIZATION: Process the data (as per your existing logic)
        print(f"Starting parameterization for {SUB_NUM}")

        previous_r_center = None
        previous_p_center = None

        num_cycles = len(epochs_df['cycle'].unique())

        # Define the components and metrics
        components = ['p', 'q', 'r', 's', 't']
        metrics = ['center', 'height', 'width', 'fwhm', 'rise_time', 'decay_time', 'rise_decay_symmetry', 'sharpness']
        intervals = ['pr_interval', 'pr_segment', 'qrs_duration', 'st_segment', 'qt_interval', 'p_duration', 'pp_interval', 'rr_interval']
        on_off_times = ['on', 'off']

        # Initialize the output dictionary
        ecg_output_dict = {
            "cycle": np.arange(0, num_cycles).tolist(),
            "r_squared": [np.nan] * num_cycles
        }

        # Add the component-specific metrics
        for comp in components:
            for metric in metrics:
                ecg_output_dict[f'{comp}_{metric}'] = [np.nan] * num_cycles
            # Adding 'on' and 'off' times for each component
            for time in on_off_times:
                ecg_output_dict[f'{comp}_{time}'] = [np.nan] * num_cycles

        # Add interval-specific keys to the dictionary
        for interval in intervals:
            ecg_output_dict[interval] = [np.nan] * num_cycles

        # Map cycle values to their corresponding indices
        cycle_to_index = {cycle: idx for idx, cycle in enumerate(epochs_df['cycle'].unique())}

        # Loop through the cycles
        for cycle in epochs_df['cycle'].unique():  # np.arange(0, 20):  #for all limited cycles
            cycle_idx = cycle_to_index[cycle]  # Get the index for the current cycle

            one_cycle = epochs_df.loc[epochs_df['cycle'] == cycle]
            
            if one_cycle.empty or one_cycle['signal_y'].isnull().values.any():
                continue

            # X values and Y values with offset correction
            xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1] + 1)
            sig = np.asarray(one_cycle['signal_y'])
            sig_flat = detrend(sig)
            sig = sig_flat - np.mean(sig_flat[0:25])

            ### GUESSES START #####
            # Defining R guesses first
            r_ind = np.argmax(sig)
            r_height = sig[r_ind]
            r_center = xs[r_ind]

            # Find indices for onset and offset of R peak
            le_ind_r, ri_ind_r = find_peak_boundaries(sig, r_ind, peak_height=r_height)
            fwhm_r = calc_fwhm(le_ind_r, ri_ind_r, r_ind)

            if fwhm_r is None:
                continue 

            # Finding P, Q, S, T components
            q_min_ind = int(r_ind - (FWHM_Q_IND * fwhm_r))
            q_ind, q_height, q_center = find_extremum(sig, xs, q_min_ind, r_ind, mode='min')
            p_ind, p_height, p_center = find_extremum(sig, xs, 0, q_ind, mode='max')
            s_max_ind = int(r_ind + (FWHM_S_IND * fwhm_r))
            s_ind, s_height, s_center = find_extremum(sig, xs, r_ind, s_max_ind, mode='min')
            t_ind, t_height, t_center = find_extremum(sig, xs, s_ind, len(sig), mode='max')

            # Organizing component information
            component_inds = {
                'p': [p_ind, p_height, p_center],
                'q': [q_ind, q_height, q_center],
                'r': [r_ind, r_height, r_center],
                's': [s_ind, s_height, s_center],
                't': [t_ind, t_height, t_center]
            }

            ### FITTING START #####
            guess = np.empty([0, 3])

            # Skip if any expected positive components are negative
            if component_inds['p'][1] < 0 or component_inds['r'][1] < 0 or component_inds['t'][1] < 0:
                continue

            # Fit Gaussian for each component
            # Use cycle_idx instead of cycle for assignments
            for comp, params in component_inds.items():
                onset, offset = find_peak_boundaries(sig, peak_index=params[0], peak_height=params[1])

                ecg_output_dict[f'{comp}_on'][cycle_idx] = xs[onset] if onset is not None else np.nan
                ecg_output_dict[f'{comp}_off'][cycle_idx] = xs[offset] if offset is not None else np.nan

                # Guess bandwidth and fit Gaussian
                short_side = min(abs(params[0] - onset), abs(offset - params[0])) if onset is not None and offset is not None else 0
                fwhm = short_side * 2
                guess_std = compute_gauss_std(fwhm)

                guess = np.vstack((guess, (params[2], params[1], guess_std)))

            # Extract components and calculate bounds
            centers = np.array([guess[i][0] for i in range(5)])
            heights = np.array([guess[i][1] for i in range(5)])
            stds = np.array([guess[i][2] for i in range(5)])
            bound_factor = 0.1
            gaus_lo_bound, gaus_hi_bound = zip(
                calc_bounds(centers[0], heights[0], stds[0], bound_factor),
                calc_bounds(centers[1], heights[1], stds[1], bound_factor, flip_height=True),
                calc_bounds(centers[2], heights[2], stds[2], bound_factor),
                calc_bounds(centers[3], heights[3], stds[3], bound_factor, flip_height=True),
                calc_bounds(centers[4], heights[4], stds[4], bound_factor)
            )

            gaus_param_bounds = (np.concatenate(gaus_lo_bound), np.concatenate(gaus_hi_bound))
            guess_flat = guess.flatten()


            # Perform Gaussian fitting with curve_fit
            try:
                gaussian_params, _ = curve_fit(gaussian_function, xs, sig, p0=guess_flat, maxfev=2500, bounds=gaus_param_bounds)
            except (ValueError, RuntimeError):
                continue

            # Reshape and store Gaussian parameters
            gaussian_params_reshape = gaussian_params.reshape((5, 3))
            for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
                ecg_output_dict[f'{comp}_center'][cycle_idx] = gaussian_params_reshape[i, 0]
                ecg_output_dict[f'{comp}_height'][cycle_idx] = gaussian_params_reshape[i, 1]
                ecg_output_dict[f'{comp}_width'][cycle_idx] = gaussian_params_reshape[i, 2]

            fit = gaussian_function(xs, *gaussian_params)

            peak_params = find_peak_params(xs, sig, gaussian_params_reshape)
            shape_params, peak_indices = calc_shape_params(xs, sig, peak_params)

            # Store shape parameters for each ECG component (P, Q, R, S, T)
            for i, comp in enumerate(['p', 'q', 'r', 's', 't']):
                ecg_output_dict[f'{comp}_fwhm'][cycle_idx] = shape_params[i, 0]
                ecg_output_dict[f'{comp}_rise_time'][cycle_idx] = shape_params[i, 1]
                ecg_output_dict[f'{comp}_decay_time'][cycle_idx] = shape_params[i, 2]
                ecg_output_dict[f'{comp}_rise_decay_symmetry'][cycle_idx] = shape_params[i, 3]
                ecg_output_dict[f'{comp}_sharpness'][cycle_idx] = shape_params[i, 4]

            # Store other features 
            intervals_to_calculate = {
                'p_duration': ('p_off', 'p_on'),
                'pr_interval': ('q_on', 'p_on'),
                'pr_segment': ('q_on', 'p_off'),
                'qrs_duration': ('s_off', 'q_on'),
                'st_segment': ('t_off', 's_off'),
                'qt_interval': ('t_off', 'q_on')
            }

            # Calculate the intervals
            for interval, (off_key, on_key) in intervals_to_calculate.items():
                ecg_output_dict[interval][cycle_idx] = ecg_output_dict[off_key][cycle_idx] - ecg_output_dict[on_key][cycle_idx]

            # Only calculate R-R and P-P intervals if there is a valid previous cycle
            if previous_r_center is None or previous_p_center is None:
                # This ensures that the first valid cycle will have NaN intervals
                ecg_output_dict['rr_interval'][cycle_idx] = np.nan
                ecg_output_dict['pp_interval'][cycle_idx] = np.nan
            else:
                # Calculate R-R interval
                ecg_output_dict['rr_interval'][cycle_idx] = r_center - previous_r_center
                # Calculate P-P interval
                ecg_output_dict['pp_interval'][cycle_idx] = p_center - previous_p_center

            # After calculation, update the previous peaks' locations for the next cycle
            previous_r_center = r_center
            previous_p_center = p_center


            # Now pass the fitted curve to calc_r_squared instead of gaussian_params
            r_squared = calc_r_squared(sig, fit)  # Pass the fitted curve instead of gaussian_params
            ecg_output_dict['r_squared'][cycle_idx] = r_squared


        # Calculate RR intervals and metrics based on all cycles
        rr_intervals = np.array(ecg_output_dict['rr_interval'])
        rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
        heart_rate = 60 / (rr_intervals / 1000)
        average_heart_rate = np.nanmean(heart_rate) if len(heart_rate) > 0 else np.nan

        # HRV metrics
        if len(rr_intervals) > 1:
            sdnn = np.std(rr_intervals, ddof=1)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        else:
            sdnn, rmssd, nn50 = np.nan, np.nan, np.nan

        # Save results
        summary_metrics = {'Average_Heart_Rate': [average_heart_rate], 'SDNN': [sdnn], 'RMSSD': [rmssd], 'NN50': [nn50]}
        summary_df = pd.DataFrame(summary_metrics)
        summary_df.to_csv(f'../docs/saved_files/timedomain_results/{SUB_NUM}_ecg_summary_data.csv', index=False)

        ecg_output = pd.DataFrame(ecg_output_dict)
        ecg_output.to_csv(f'../docs/saved_files/timedomain_results/{SUB_NUM}_ecg_output_data.csv', index=False)

    except Exception as e:
        print(f"Error processing subject {SUB_NUM}: {str(e)}")
        traceback.print_exc()  # Print detailed error traceback



# Main execution block
if __name__ == "__main__":
    dir_path = '/Users/morganfitzgerald/Projects/ECG_tool_val/data/AutoAge_data/raw'
    files_dat_dict, files_hea_dict = create_subject_file_mapping(dir_path)
    results_dir = '/Users/morganfitzgerald/Projects/ECG_tool_val/saved_files/timedomain_results'

    subject_args = [
        (SUB_NUM, dat_path, files_hea_dict[SUB_NUM], results_dir)
        for SUB_NUM, dat_path in files_dat_dict.items()
        if SUB_NUM in files_hea_dict
    ]

    # Process each subject
    for SUB_NUM, dat_path, hea_path, results_dir in subject_args:
        process_subject(SUB_NUM, dat_path, hea_path, results_dir)