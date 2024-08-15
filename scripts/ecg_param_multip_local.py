#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import iirnotch, filtfilt
import pandas as pd
import neurokit2 as nk
import numpy as np
import os
from src.gaussian_funcs import gaussian_function, compute_gauss_std, calc_r_squared
from src.utils import (
    create_subject_file_mapping,
    extract_data,
    extract_metadata,
    compute_knee_frequency,
    compute_time_constant,
)
from src.processing import (
    simulate_ecg_signal,
    average_fft_of_epochs_loglog,
    extract_control_points,
    find_most_similar_signal,
    create_peak_params,
    get_peak_indices,
)
from src.analysis import (
    epoch_signals,
    find_extremum,
    estimate_fwhm,
    find_peak_boundaries,
    generate_histograms,
    calculate_sharpness_deriv,
)
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.optimize import curve_fit
from src.utils import normalize
import multiprocessing as mp
import time

# FS = sampling rate; The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
FS = 1000
sampling_rate = 1000
CROP_MIN = 1000
CROP_MAX = 3000
WINDOW_LENGTH = 5000
FWHM_Q_IND = 5
FWHM_S_IND = 5
STD_R_HEIGHT = 10
PLOT = False

# Directories
dir_path = "/Users/morganfitzgerald/Projects/ecg_param/data/raw"
files_dat_dict, files_hea_dict = create_subject_file_mapping(dir_path)
results_dir = "../docs/saved_files/timedomain_results/"


def process_subject(SUB_NUM, dat_path, hea_path, results_dir):
    start_time = time.time()
    print(f"Processing subject {SUB_NUM}")
    try:
        # Load the current subject's data using the .dat and .hea file paths
        sigs, metadata = extract_data(
            dat_path,  # Path to the .dat file for the current subject
            hea_path,  # Path to the .hea file for the current subject
            raw_dtype="int16",
        )

        # SELECT THE SIGNAL
        template_ecg = simulate_ecg_signal(
            duration=5,
            sampling_rate=1000,
            heart_rate=80,
            amplitude_factor=7,
            normalize=False,
        )
        normalized_signals = []

        for ind in range(metadata["n_sigs"]):
            signal_name = metadata[f"sig{str(ind).zfill(2)}"]["signal_name"]
            if signal_name == "NIBP":
                continue
            cropped_signal = sigs[ind][CROP_MIN:CROP_MAX]
            normalized_signal = (cropped_signal - np.mean(cropped_signal)) / np.std(
                cropped_signal
            )
            normalized_signals.append(normalized_signal)

        selected_signal, selected_signal_name, selected_signal_index = (
            find_most_similar_signal(template_ecg, normalized_signals, metadata)
        )
        ecg = sigs[selected_signal_index]

        # High-pass filter
        cutoff_frequency = 0.05  # Set your desired cutoff frequency in Hz
        order = 4  # Set the filter order
        b, a = butter(
            order, cutoff_frequency, btype="high", analog=False, fs=metadata["fs"]
        )
        ecg_hp = filtfilt(b, a, ecg)

        # Notch filtering
        fs = metadata["fs"]  # Sampling rate
        f0 = 50  # Line noise frequency (50 Hz)
        quality_factor = 30  # Quality factor, determines bandwidth around f0
        b, a = iirnotch(f0, quality_factor, fs)
        ecg_notch = filtfilt(b, a, ecg_hp)

        # Neurokit Cleaning
        ecg_clean_nk = nk.ecg_clean(ecg, sampling_rate=1000)
        p_peaks_nk, rpeaks_nk, waves_nk = extract_control_points(
            ecg_clean_nk, sampling_rate
        )
        epochs_nk_df, _ = epoch_signals(
            p_peaks_nk, ecg_clean_nk, FS, SUB_NUM, PLOT=False, SAVE=False
        )

        # Epoch and Detrend Sig
        epochs_df, result_r_latencies = epoch_signals(
            p_peaks_nk, ecg_notch, FS, SUB_NUM, PLOT=False, SAVE=False
        )

        # PARAMETERIZATION LOOP: Fit Gaussians
        num_cycles = len(epochs_df["cycle"].unique())

        # Create the dictionary
        ecg_output_dict = {
            "cycle": np.arange(0, num_cycles).tolist(),
            "p_center": [np.nan] * num_cycles,
            "p_height": [np.nan] * num_cycles,
            "p_width": [np.nan] * num_cycles,
            "q_center": [np.nan] * num_cycles,
            "q_height": [np.nan] * num_cycles,
            "q_width": [np.nan] * num_cycles,
            "r_center": [np.nan] * num_cycles,
            "r_height": [np.nan] * num_cycles,
            "r_width": [np.nan] * num_cycles,
            "s_center": [np.nan] * num_cycles,
            "s_height": [np.nan] * num_cycles,
            "s_width": [np.nan] * num_cycles,
            "t_center": [np.nan] * num_cycles,
            "t_height": [np.nan] * num_cycles,
            "t_width": [np.nan] * num_cycles,
            "r_squared": [np.nan] * num_cycles,
            "pr_interval": [np.nan] * num_cycles,
            "pr_segment": [np.nan] * num_cycles,
            "qrs_duration": [np.nan] * num_cycles,
            "st_segment": [np.nan] * num_cycles,
            "qt_interval": [np.nan] * num_cycles,
            "p_duration": [np.nan] * num_cycles,
            "pp_interval": [np.nan] * num_cycles,
            "rr_interval": [np.nan] * num_cycles,
            "fwhm_p": [np.nan] * num_cycles,
            "rise_time_p": [np.nan] * num_cycles,
            "decay_time_p": [np.nan] * num_cycles,
            "rise_decay_symmetry_p": [np.nan] * num_cycles,
            "sharpness_deriv_p": [np.nan] * num_cycles,
            "sharpness_diff_p": [np.nan] * num_cycles,
            "fwhm_q": [np.nan] * num_cycles,
            "rise_time_q": [np.nan] * num_cycles,
            "decay_time_q": [np.nan] * num_cycles,
            "rise_decay_symmetry_q": [np.nan] * num_cycles,
            "sharpness_deriv_q": [np.nan] * num_cycles,
            "sharpness_diff_q": [np.nan] * num_cycles,
            "fwhm_r": [np.nan] * num_cycles,
            "rise_time_r": [np.nan] * num_cycles,
            "decay_time_r": [np.nan] * num_cycles,
            "rise_decay_symmetry_r": [np.nan] * num_cycles,
            "sharpness_deriv_r": [np.nan] * num_cycles,
            "sharpness_diff_r": [np.nan] * num_cycles,
            "fwhm_s": [np.nan] * num_cycles,
            "rise_time_s": [np.nan] * num_cycles,
            "decay_time_s": [np.nan] * num_cycles,
            "rise_decay_symmetry_s": [np.nan] * num_cycles,
            "sharpness_deriv_s": [np.nan] * num_cycles,
            "sharpness_diff_s": [np.nan] * num_cycles,
            "fwhm_t": [np.nan] * num_cycles,
            "rise_time_t": [np.nan] * num_cycles,
            "decay_time_t": [np.nan] * num_cycles,
            "rise_decay_symmetry_t": [np.nan] * num_cycles,
            "sharpness_deriv_t": [np.nan] * num_cycles,
            "sharpness_diff_t": [np.nan] * num_cycles,
            "Average_Heart_Rate": [np.nan] * num_cycles,
            "SDNN": [np.nan] * num_cycles,
            "RMSSD": [np.nan] * num_cycles,
            "NN50": [np.nan] * num_cycles,
        }

        # Initialize variables to hold the previous peaks' locations
        previous_r_center = None
        previous_p_center = None

        # Ensure on and off keys are initialized
        for comp in ["p", "q", "r", "s", "t"]:
            ecg_output_dict[f"{comp}_on"] = [np.nan] * num_cycles
            ecg_output_dict[f"{comp}_off"] = [np.nan] * num_cycles

        for cycle in np.arange(0, num_cycles):

            # print(f"Parameterizing cycle #{cycle}.")
            one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle]

            if one_cycle.empty:
                # print(f'cycle #{cycle} is empty')
                continue

            if one_cycle["signal_y"].isnull().values.any():
                # print(f'cycle #{cycle} has NaNs')
                continue

            # X values and Y values with offset correction
            xs = np.arange(one_cycle["index"].iloc[0], one_cycle["index"].iloc[-1] + 1)
            sig = np.asarray(one_cycle["signal_y"])
            sig_flat = detrend(sig)
            sig = sig_flat - np.mean(sig_flat[0:25])

            ##### Defining R guesses first #####
            r_ind = np.argmax(sig)
            r_height = sig[r_ind]
            r_center = xs[r_ind]

            half_r_height = 0.5 * r_height
            le_ind_r, ri_ind_r = find_peak_boundaries(sig, r_ind, peak_height=r_height)

            # Use estimate_fwhm to calculate the FWHM based on the left and right indices
            fwhm_r = estimate_fwhm(le_ind_r, ri_ind_r, r_ind)

            # Check if FWHM calculation was successful
            if fwhm_r is None:
                # print(f"Cycle #{cycle} could not estimate FWHM.")
                continue

            # #### Now define rest of component guesses ####
            # Finding P, Q, S, T components
            q_min_ind = int(r_ind - (FWHM_Q_IND * fwhm_r))
            q_ind, q_height, q_center = find_extremum(
                sig, xs, q_min_ind, r_ind, mode="min"
            )
            p_ind, p_height, p_center = find_extremum(sig, xs, 0, q_ind, mode="max")
            s_max_ind = int(r_ind + (FWHM_S_IND * fwhm_r))
            s_ind, s_height, s_center = find_extremum(
                sig, xs, r_ind, s_max_ind, mode="min"
            )
            t_ind, t_height, t_center = find_extremum(
                sig, xs, s_ind, len(sig), mode="max"
            )

            # Organizing component information
            component_inds = {
                "p": [p_ind, p_height, p_center],
                "q": [q_ind, q_height, q_center],
                "r": [r_ind, r_height, r_center],
                "s": [s_ind, s_height, s_center],
                "t": [t_ind, t_height, t_center],
            }

            # Initialize matrix of guess parameters for gaussian fitting
            guess = np.empty([0, 3])

            # Skip cycle if any of the expected positive components are negative
            if component_inds["p"][1] < 0:
                # print(f"cycle #{cycle}'s p component is negative")
                continue
            if component_inds["r"][1] < 0:
                # print(f"cycle #{cycle}'s r component is negative")
                continue
            if component_inds["t"][1] < 0:
                # print(f"cycle #{cycle}'s t component is negative")
                continue

            for comp, params in component_inds.items():
                # Directly use the find_peak_boundaries function with peak_height parameter
                onset, offset = find_peak_boundaries(
                    sig, peak_index=params[0], peak_height=params[1]
                )

                # Store the onset and offset values in the dictionary
                ecg_output_dict[f"{comp}_on"][cycle] = (
                    xs[onset] if onset is not None else np.nan
                )
                ecg_output_dict[f"{comp}_off"][cycle] = (
                    xs[offset] if offset is not None else np.nan
                )

                # Guess bandwidth procedure: estimate the width of the peak
                if onset is not None and offset is not None:
                    short_side = min(abs(params[0] - onset), abs(offset - params[0]))
                else:
                    short_side = 0

                fwhm = short_side * 2
                guess_std = compute_gauss_std(fwhm)

                # Collect guess parameters and subtract this guess gaussian from the data
                guess = np.vstack((guess, (params[2], params[1], guess_std)))
                peak_gauss = gaussian_function(xs, params[2], params[1], guess_std)

            # center, height, width
            lo_bound = [
                [
                    guess[0][0] - 0.5 * guess[0][2],
                    -np.inf,
                    guess[0][2] - 2 * guess[0][2],
                ],
                [
                    guess[1][0] - 0.5 * guess[1][2],
                    -np.inf,
                    guess[1][2] - 2 * guess[1][2],
                ],
                [
                    guess[2][0] - 0.5 * guess[2][2],
                    -np.inf,
                    guess[2][2] - 2 * guess[2][2],
                ],
                [
                    guess[3][0] - 0.5 * guess[3][2],
                    -np.inf,
                    guess[3][2] - 2 * guess[3][2],
                ],
                [
                    guess[4][0] - 0.5 * guess[4][2],
                    -np.inf,
                    guess[4][2] - 2 * guess[4][2],
                ],
            ]

            hi_bound = [
                [
                    guess[0][0] + 0.5 * guess[0][2],
                    np.inf,
                    guess[0][2] + 2 * guess[0][2],
                ],
                [
                    guess[1][0] + 0.5 * guess[1][2],
                    np.inf,
                    guess[1][2] + 2 * guess[1][2],
                ],
                [
                    guess[2][0] + 0.5 * guess[2][2],
                    np.inf,
                    guess[2][2] + 2 * guess[2][2],
                ],
                [
                    guess[3][0] + 0.5 * guess[3][2],
                    np.inf,
                    guess[3][2] + 2 * guess[3][2],
                ],
                [
                    guess[4][0] + 0.5 * guess[4][2],
                    np.inf,
                    guess[4][2] + 2 * guess[4][2],
                ],
            ]

            gaus_param_bounds = (
                tuple([item for sublist in lo_bound for item in sublist]),
                tuple([item for sublist in hi_bound for item in sublist]),
            )

            # Flatten guess, for use with curve fit
            guess_flat = np.ndarray.flatten(guess)

            maxfev = 5000

            # Check if any lower bound is not strictly less than its corresponding upper bound
            skip_cycle = False
            for lb, ub in zip(lo_bound, hi_bound):
                if not all(l < u for l, u in zip(lb, ub)):
                    # print(f"Skipping cycle #{cycle} due to invalid bounds.")
                    skip_cycle = True
                    break

            if skip_cycle:
                continue

            # If the check passes, proceed with fitting
            try:
                gaussian_params, _ = curve_fit(
                    gaussian_function,
                    xs,
                    sig,
                    p0=guess_flat,
                    maxfev=maxfev,
                    bounds=gaus_param_bounds,
                )
            except RuntimeError as e:
                # print(f"Could not fit cycle #{cycle}: {e}")
                continue

            # Reshape gaussian_params from 1,15 to 3, 5 to feed into create peak params
            gaussian_params_reshape = gaussian_params.reshape((5, 3))

            # Store the center, height, and width for each peak in the dictionary
            for i, comp in enumerate(["p", "q", "r", "s", "t"]):
                ecg_output_dict[f"{comp}_center"][cycle] = gaussian_params_reshape[i, 0]
                ecg_output_dict[f"{comp}_height"][cycle] = gaussian_params_reshape[i, 1]
                ecg_output_dict[f"{comp}_width"][cycle] = gaussian_params_reshape[i, 2]

            # Bycycle fit
            peak_params = create_peak_params(xs, sig, gaussian_params_reshape)

            # Initialize list of shape parameters
            shape_params = np.empty((len(peak_params), 6))
            peak_indices = np.empty((len(peak_params), 3))

            for ii, peak in enumerate(peak_params):
                # Get peak indices
                start_index, peak_index, end_index = get_peak_indices(xs, sig, peak)

                # If the peak indices could not be determined, set all shape params to NaN
                if np.isnan(start_index) or np.isnan(end_index):
                    shape_params[ii] = [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                    peak_indices[ii] = [np.nan, np.nan, np.nan]
                    continue

                # Compute fwhm, rise-, and decay-time
                fwhm = xs[end_index] - xs[start_index]
                rise_time = xs[peak_index] - xs[start_index]
                decay_time = xs[end_index] - xs[peak_index]

                # Compute rise-decay symmetry
                rise_decay_symmetry = rise_time / fwhm

                # Compute sharpness using diff
                half_mag = int(np.abs(peak[1] / 2))
                half_mag_cropped_xval = np.argmin(np.abs(sig - half_mag))
                left_index = peak_index - 15
                right_index = peak_index + 15

                sharpness_diff = np.mean(
                    (
                        sig[peak_index] - sig[left_index],
                        sig[peak_index] + sig[right_index],
                    )
                )
                sharpness_diff = np.abs(sharpness_diff)

                # Compute sharpness using the derivative method
                sharpness_deriv = calculate_sharpness_deriv(sig, peak_index) / (
                    sig[peak_index]
                )
                sharpness_deriv = np.abs(sharpness_deriv)

                # Collect results
                shape_params[ii] = [
                    fwhm,
                    rise_time,
                    decay_time,
                    rise_decay_symmetry,
                    sharpness_diff,
                    sharpness_deriv,
                ]
                peak_indices[ii] = [start_index, peak_index, end_index]

            fit = gaussian_function(xs, *gaussian_params)

            # Calculate durations and intervals
            ecg_output_dict["p_duration"][cycle] = (
                ecg_output_dict["p_off"][cycle] - ecg_output_dict["p_on"][cycle]
            )
            ecg_output_dict["pr_interval"][cycle] = (
                ecg_output_dict["q_on"][cycle] - ecg_output_dict["p_on"][cycle]
            )
            ecg_output_dict["pr_segment"][cycle] = (
                ecg_output_dict["q_on"][cycle] - ecg_output_dict["p_off"][cycle]
            )
            ecg_output_dict["qrs_duration"][cycle] = (
                ecg_output_dict["s_off"][cycle] - ecg_output_dict["q_on"][cycle]
            )
            ecg_output_dict["st_segment"][cycle] = (
                ecg_output_dict["t_off"][cycle] - ecg_output_dict["s_off"][cycle]
            )
            ecg_output_dict["qt_interval"][cycle] = (
                ecg_output_dict["t_off"][cycle] - ecg_output_dict["q_on"][cycle]
            )

            # Calculate R-R interval if there's a previous R peak
            if previous_r_center is not None:
                r_r_interval = r_center - previous_r_center
                ecg_output_dict["rr_interval"][cycle] = r_r_interval
            else:
                ecg_output_dict["rr_interval"][cycle] = np.nan

            # Calculate P-P interval if there's a previous P peak
            if previous_p_center is not None:
                p_p_interval = p_center - previous_p_center
                ecg_output_dict["pp_interval"][cycle] = p_p_interval
            else:
                ecg_output_dict["pp_interval"][cycle] = np.nan

            # Update the previous peaks' locations
            previous_r_center = r_center
            previous_p_center = p_center

            r_squared = calc_r_squared(sig, fit)

            # Add features to dictionary
            ecg_output_dict["r_squared"][cycle] = r_squared

            shape_params_flat = np.ndarray.flatten(shape_params)
            (
                ecg_output_dict["fwhm_p"][cycle],
                ecg_output_dict["rise_time_p"][cycle],
                ecg_output_dict["decay_time_p"][cycle],
                ecg_output_dict["rise_decay_symmetry_p"][cycle],
                ecg_output_dict["sharpness_diff_p"][cycle],
                ecg_output_dict["sharpness_deriv_p"][cycle],
            ) = shape_params_flat[:6]
            (
                ecg_output_dict["fwhm_q"][cycle],
                ecg_output_dict["rise_time_q"][cycle],
                ecg_output_dict["decay_time_q"][cycle],
                ecg_output_dict["rise_decay_symmetry_q"][cycle],
                ecg_output_dict["sharpness_diff_q"][cycle],
                ecg_output_dict["sharpness_deriv_q"][cycle],
            ) = shape_params_flat[6:12]
            (
                ecg_output_dict["fwhm_r"][cycle],
                ecg_output_dict["rise_time_r"][cycle],
                ecg_output_dict["decay_time_r"][cycle],
                ecg_output_dict["rise_decay_symmetry_r"][cycle],
                ecg_output_dict["sharpness_diff_r"][cycle],
                ecg_output_dict["sharpness_deriv_r"][cycle],
            ) = shape_params_flat[12:18]
            (
                ecg_output_dict["fwhm_s"][cycle],
                ecg_output_dict["rise_time_s"][cycle],
                ecg_output_dict["decay_time_s"][cycle],
                ecg_output_dict["rise_decay_symmetry_s"][cycle],
                ecg_output_dict["sharpness_diff_s"][cycle],
                ecg_output_dict["sharpness_deriv_s"][cycle],
            ) = shape_params_flat[18:24]
            (
                ecg_output_dict["fwhm_t"][cycle],
                ecg_output_dict["rise_time_t"][cycle],
                ecg_output_dict["decay_time_t"][cycle],
                ecg_output_dict["rise_decay_symmetry_t"][cycle],
                ecg_output_dict["sharpness_diff_t"][cycle],
                ecg_output_dict["sharpness_deriv_t"][cycle],
            ) = shape_params_flat[24:30]

        # Process the raw ECG signal (not the cleaned signal)
        processed_data, _ = nk.ecg_process(ecg_notch, sampling_rate=1000)

        # Access heart rate from the processed data
        heart_rate = processed_data["ECG_Rate"]

        # Calculate average heart rate
        average_heart_rate = heart_rate.mean()

        rr_intervals = np.array(ecg_output_dict["rr_interval"])
        rr_intervals = rr_intervals[~np.isnan(rr_intervals)]  # Ensure no NaN values

        # Calculate HRV metrics
        if len(rr_intervals) > 1:  # Need at least two intervals for RMSSD
            sdnn = np.std(rr_intervals, ddof=1)
            diff_nn_intervals = np.diff(rr_intervals)
            squared_diff_nn_intervals = diff_nn_intervals**2
            rmssd = np.sqrt(np.mean(squared_diff_nn_intervals))
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        else:
            sdnn, rmssd, nn50 = np.nan, np.nan, np.nan

        # Add the calculated values to the dictionary for the first cycle
        ecg_output_dict["Average_Heart_Rate"][0] = average_heart_rate
        ecg_output_dict["SDNN"][0] = sdnn
        ecg_output_dict["RMSSD"][0] = rmssd
        ecg_output_dict["NN50"][0] = nn50

        # Convert dictionary to DataFrame
        ecg_output = pd.DataFrame(ecg_output_dict)

        # Save output in new file
        ecg_output.to_csv(os.path.join(results_dir, f"{SUB_NUM}_ecg_output.csv"))

    except Exception as e:
        print(f"Error processing subject {SUB_NUM}: {e}")

    elapsed_time = time.time() - start_time
    print(f"Subject {SUB_NUM} processed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    # Path to the directory containing the raw data files
    dir_path = "/Users/morganfitzgerald/Projects/ecg_param/data/raw"
    files_dat_dict, files_hea_dict = create_subject_file_mapping(dir_path)
    results_dir = "../docs/saved_files/timedomain_results/"

    # Create a list of arguments for each subject
    subject_args = [
        (SUB_NUM, dat_path, files_hea_dict[SUB_NUM], results_dir)
        for SUB_NUM, dat_path in files_dat_dict.items()
        if SUB_NUM in files_hea_dict
    ]

    # Create a pool of workers and process each subject in parallel using 4 cores
    with mp.Pool(1) as pool:  # Use 4 cores
        pool.starmap(process_subject, subject_args)

    print("Processing complete.")
