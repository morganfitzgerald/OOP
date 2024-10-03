#!/usr/bin/env python
# coding: utf-8

# # Using R-peak as main fitting parameter
# ### We will now be:
# 1. Finding R peak (highest)
# 2. Fitting R peak
# 3. Finding the minimum left of R (aka Q), and fitting
# 4. Finding the maximum left of Q (aka P), and fitting
# 5. Finding minimum right of R (aka S), and fitting
# 6. Finding maxmimum right of S (aka T), and fitting

# ### Imports

# In[1]:


import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import pandas as pd
import neurokit2 as nk
import numpy as np
import os

from src.gaussian_funcs import gaussian_function, compute_gauss_std, calc_r_squared
from src.utils import create_subject_file_mapping, extract_data, extract_metadata, compute_knee_frequency, compute_time_constant
from src.processing import simulate_ecg_signal, average_fft_of_epochs_loglog, extract_control_points, find_most_similar_signal, create_peak_params, get_peak_indices
from src.analysis import epoch_signals, find_extremum, estimate_fwhm, find_peak_boundaries, generate_histograms

from scipy.signal import detrend, butter, filtfilt, welch
from scipy.optimize import curve_fit

from fooof import FOOOF

from src.acfs.conversions import convert_knee, psd_to_acf, acf_to_psd
from src.utils import normalize
from src.acfs.fit import ACF
from src.acfs.conversions import convert_knee
from src.autoreg import compute_ar_spectrum

from timescales.plts import set_default_rc
set_default_rc()

print('finished package imports')

# ### Attributes

# In[2]:

plt.rcParams["figure.figsize"] = (10, 5)

#FS = sampling rate; The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
FS = 1000
sampling_rate = 1000

CROP_MIN = 1000
CROP_MAX = 3000
WINDOW_LENGTH = 5000

FWHM_Q_IND = 5
FWHM_S_IND = 5
STD_R_HEIGHT = 10

PLOT = False


# ## Load Data & Select Sig

# In[3]:


# Manual input of subject number as a string to match dictionary keys
SUB_NUM = '20'  # Ensure this matches the format of your subject numbers in the dictionary

# Code to select the ECG signal
dir_path = '/Users/morganfitzgerald/Projects/ecg_param/data/raw'
files_dat_dict, files_hea_dict = create_subject_file_mapping(dir_path)

# Extract single subject
if SUB_NUM in files_dat_dict and SUB_NUM in files_hea_dict:
    sigs, metadata = extract_data(
        files_dat_dict[SUB_NUM],
        files_hea_dict[SUB_NUM],
        raw_dtype='int16'
    )
else:
    print(f"Subject number {SUB_NUM} not found.")
    
# Iterate through signals and plot only the original signals
for ind in range(metadata['n_sigs']):

    plt.figure(figsize=(12, 2))

    # Original signal
    signal_name = metadata[f'sig{str(ind).zfill(2)}']['signal_name']
    original_signal = sigs[ind][CROP_MIN:CROP_MAX]
    
    # plt.plot(original_signal)
    # plt.title(f'Original {signal_name}', size=10)

# plt.show()


# In[4]:


#SELECT THE SIGNAL
# Template ECG signal
template_ecg = simulate_ecg_signal(duration=5, sampling_rate=1000, heart_rate=80, amplitude_factor=7, normalize=False)

# List to store normalized signals
#Must normalize or the diff amplitudes mess up signal selection
normalized_signals = []

# Iterate through signals and normalize each
for ind in range(metadata['n_sigs']):
    signal_name = metadata[f'sig{str(ind).zfill(2)}']['signal_name']

    if signal_name == 'NIBP': 
        continue
    
    # Cut the signal shorter before normalization
    cropped_signal = sigs[ind][CROP_MIN:CROP_MAX]

    # Normalize the signal
    normalized_signal = (cropped_signal - np.mean(cropped_signal)) / np.std(cropped_signal)

    # Add normalized signal to the list
    normalized_signals.append(normalized_signal)

   #plt.plot(normalized_signal)

# Find the most similar signal to the template using cross-correlation
selected_signal, selected_signal_name, selected_signal_index = find_most_similar_signal(template_ecg, normalized_signals, metadata)

# # Plot the most similar signal with the original signal name in the title
# plt.figure(figsize=(8, 1))
# plt.plot(selected_signal)
# # plt.plot(template_ecg)
# plt.title(f'Selected ECG: {selected_signal_name}', size=10)
# plt.show()

# Now 'ecg' contains the selected signal
ecg = sigs[selected_signal_index]


# In[5]:


# Define the high-pass filter parameters
cutoff_frequency = 0.05  # Set your desired cutoff frequency in Hz
order = 4  # Set the filter order

# Design the high-pass filter
b, a = butter(order, cutoff_frequency, btype='high', analog=False, fs=metadata['fs'])

# Apply the high-pass filter to the ECG signal
ecg_hp = filtfilt(b, a, ecg)

# # Plot the original and filtered signals
# plt.figure(figsize=(10, 2))
# plt.plot(ecg[CROP_MIN:CROP_MAX], label='Original ECG')
# plt.plot(ecg_hp[CROP_MIN:CROP_MAX], label=f'High-pass Filtered ECG (Cutoff: {cutoff_frequency} Hz)')
# plt.title('High-pass Filtering of ECG Signal')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# # plt.legend()
# plt.show()


# #### Notch filtering

# In[6]:


from scipy.signal import iirnotch, filtfilt
# import matplotlib.pyplot as plt
# import numpy as np

# Assuming 'ecg' is your ECG signal, 'fs' is the sampling rate
fs = metadata['fs']  # Sampling rate
f0 = 50  # Line noise frequency (50 Hz)
quality_factor = 30  # Quality factor, determines bandwidth around f0

# Design the bandstop (notch) filter
b, a = iirnotch(f0, quality_factor, fs)

# Apply the filter
ecg_notch = filtfilt(b, a, ecg_hp)

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(ecg[CROP_MIN:CROP_MAX], label='Original ECG')
# plt.plot(ecg_notch[CROP_MIN:CROP_MAX], label=f'High-pass Filtered ECG (Cutoff: {cutoff_frequency} Hz)')
# plt.title('ECG Signal Before and After 50 Hz Line Noise Removal')
# plt.xlabel('Sample Number')
# plt.ylabel('Amplitude')
# # plt.legend()
# plt.show()


# ## Neurokit Cleaning
# 
# - Here we are using neurokit's cleaning methods to extract a P onset value in the time domain. 
# - Subsequently, we take this onset value and epoch our 'more raw' signal to preserve shape. 
# 

# In[7]:

print("Starting the neurokit clean...")
ecg_clean_nk = nk.ecg_clean(ecg, sampling_rate=1000)
p_peaks_nk, rpeaks_nk, waves_nk = extract_control_points(ecg_clean_nk, sampling_rate)
epochs_nk_df, _ = epoch_signals(p_peaks_nk, ecg_clean_nk, FS, SUB_NUM, PLOT = False, SAVE = False)
print("Finished the neurokit epoching...")

# ## Epoch and Detrend Sig
#
# In[8]:
# Call the epoch functoin
epochs_df, result_r_latencies = epoch_signals(p_peaks_nk, ecg_notch,FS, SUB_NUM,  PLOT = False, SAVE = False)

# ## PSD and Fit from SpecParam

# In[9]:
##FFT Signal
avg_freq, avg_magnitude = average_fft_of_epochs_loglog(epochs_df, sampling_rate = 1000, PLOT=False)
# In[36]:


# Set the frequency range to fit the model
freq_range = [2, 200]

# Define columns for the DataFrame based on your results dictionary
columns = ['SUB_NUM', 'Offset', 'Exponent', 'Error', 'R^2']
specparam_results = pd.DataFrame(columns=columns)

# Initialize a FOOOF object
fm = FOOOF(aperiodic_mode='fixed', verbose=False, max_n_peaks=10)

# Fit the model to the data
fm.fit(avg_freq, avg_magnitude, freq_range)

# # Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.report(avg_freq, avg_magnitude, freq_range)

offset = fm.aperiodic_params_[0]  # Assuming first element is offset
exponent = fm.aperiodic_params_[1]  # Assuming third element is exponent
error = fm.error_
r_squared = fm.r_squared_

# Create a dictionary with this data
specparam_data = {
    'SUB_NUM': SUB_NUM,
    'Offset_sp': [offset],
    'Exponent_sp': [exponent],
    'Error_sp': [error],
    'R^2_sp': [r_squared]
}

specparam_results = pd.DataFrame(specparam_data)

#Save output in new file 
specparam_results.to_csv(f'../docs/saved_files/spectral_results/{SUB_NUM}_specparam_results.csv')


# # PARAMETERIZATION: Fit Gaussians

# In[11]:


#Build empty dataframe
ecg_output = pd.DataFrame({
    "cycle": np.arange(0, len(epochs_df['cycle'].unique())),
    "p_center": np.zeros(len(epochs_df['cycle'].unique())),
    "p_height": np.zeros(len(epochs_df['cycle'].unique())),
    "p_width": np.zeros(len(epochs_df['cycle'].unique())),
    "q_center": np.zeros(len(epochs_df['cycle'].unique())),
    "q_height": np.zeros(len(epochs_df['cycle'].unique())),
    "q_width": np.zeros(len(epochs_df['cycle'].unique())),
    "r_center": np.zeros(len(epochs_df['cycle'].unique())),
    "r_height": np.zeros(len(epochs_df['cycle'].unique())),
    "r_width": np.zeros(len(epochs_df['cycle'].unique())),
    "s_center": np.zeros(len(epochs_df['cycle'].unique())),
    "s_height": np.zeros(len(epochs_df['cycle'].unique())),
    "s_width": np.zeros(len(epochs_df['cycle'].unique())),
    "t_center": np.zeros(len(epochs_df['cycle'].unique())),
    "t_height": np.zeros(len(epochs_df['cycle'].unique())),
    "t_width": np.zeros(len(epochs_df['cycle'].unique())),
    "r_squared": np.zeros(len(epochs_df['cycle'].unique())),
    "pr_interval": np.zeros(len(epochs_df['cycle'].unique())),
    "pr_segment": np.zeros(len(epochs_df['cycle'].unique())),
    "qrs_duration": np.zeros(len(epochs_df['cycle'].unique())),
    "st_segment": np.zeros(len(epochs_df['cycle'].unique())),
    "qt_interval": np.zeros(len(epochs_df['cycle'].unique())),
    "p_duration": np.zeros(len(epochs_df['cycle'].unique())), 
    "pp_interval": np.zeros(len(epochs_df['cycle'].unique())),  
    "rr_interval": np.zeros(len(epochs_df['cycle'].unique())),

    "fwhm_p": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_time_p": np.zeros(len(epochs_df['cycle'].unique())),
    "decay_time_p": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_decay_symmetry_p": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_p": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_rise_p": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_decay_p": np.zeros(len(epochs_df['cycle'].unique())), 


    "fwhm_q": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_time_q": np.zeros(len(epochs_df['cycle'].unique())),
    "decay_time_q": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_decay_symmetry_q": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_q": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_rise_q": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_decay_q": np.zeros(len(epochs_df['cycle'].unique())), 

    "fwhm_r": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_time_r": np.zeros(len(epochs_df['cycle'].unique())),
    "decay_time_r": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_decay_symmetry_r": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_r": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_rise_r": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_decay_r": np.zeros(len(epochs_df['cycle'].unique())),

    "fwhm_s": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_time_s": np.zeros(len(epochs_df['cycle'].unique())),
    "decay_time_s": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_decay_symmetry_s": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_s": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_rise_s": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_decay_s": np.zeros(len(epochs_df['cycle'].unique())),


    "fwhm_t": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_time_t": np.zeros(len(epochs_df['cycle'].unique())),
    "decay_time_t": np.zeros(len(epochs_df['cycle'].unique())),
    "rise_decay_symmetry_t": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_t": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_rise_t": np.zeros(len(epochs_df['cycle'].unique())),
    "sharpness_decay_t": np.zeros(len(epochs_df['cycle'].unique())),

    "Average_Heart_Rate": np.zeros(len(epochs_df['cycle'].unique())),
    "SDNN": np.zeros(len(epochs_df['cycle'].unique())),
    "RMSSD": np.zeros(len(epochs_df['cycle'].unique())),

})        
      
ecg_output


# In[30]:


# Initialize variables to hold the previous peaks' locations
previous_r_center = None
previous_p_center = None

#Isolate one subject at a time and build a for loop for fitting gaussians
#for cycle in np.arange(0, 40):

for cycle in np.arange(0, len((epochs_df['cycle'].unique()))): 
    print(f"Parameterizing cycle #{cycle}.")
    one_cycle = epochs_df.loc[epochs_df['cycle'] == cycle]

    if one_cycle.empty:
        print(f'cycle #{cycle} is empty')
        continue
    
    if one_cycle['signal_y'].isnull().values.any():
        print(f'cycle #{cycle} has NaNs')
        continue

    # X values and Y values with offset correction
    xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1]+1)
    sig = np.asarray(one_cycle['signal_y'])
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
        print(f"Cycle #{cycle} could not estimate FWHM.")
        continue

    # #### Now define rest of component guesses ####     
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

    # Initialize matrix of guess parameters for gaussian fitting
    guess = np.empty([0, 3])

    # Skip cycle if any of the expected positive components are negative
    if component_inds['p'][1] < 0:
        print(f"cycle #{cycle}'s p component is negative")
        continue
    if component_inds['r'][1] < 0:
        print(f"cycle #{cycle}'s r component is negative")
        continue
    if component_inds['t'][1] < 0:
        print(f"cycle #{cycle}'s t component is negative")
        continue


     # Initialize an empty list for storing guess parameters
    guess_params = []
    
    for comp, params in component_inds.items():
        
        # Directly use the find_peak_boundaries function with peak_height parameter
        onset, offset = find_peak_boundaries(sig, peak_index=params[0], peak_height=params[1])
        
        # Store the onset and offset values in the dataframe
        ecg_output.loc[cycle, f'{comp}_on'] = xs[onset] if onset is not None else np.nan
        ecg_output.loc[cycle, f'{comp}_off'] = xs[offset] if offset is not None else np.nan

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
    lo_bound = [[guess[0][0]- 0.5*guess[0][2], -np.inf, guess[0][2] - 2*guess[0][2]],
            [guess[1][0]- 0.5*guess[1][2], -np.inf, guess[1][2] - 2*guess[1][2]], 
            [guess[2][0]- 0.5*guess[2][2], -np.inf, guess[2][2] - 2*guess[2][2]],
            [guess[3][0]- 0.5*guess[3][2], -np.inf, guess[3][2] - 2*guess[3][2]],
            [guess[4][0]- 0.5*guess[4][2], -np.inf, guess[4][2] - 2*guess[4][2]]]

    hi_bound = [[guess[0][0]+ 0.5*guess[0][2], np.inf, guess[0][2] + 2*guess[0][2]],
                [guess[1][0]+ 0.5*guess[1][2], np.inf, guess[1][2] + 2*guess[1][2]], 
                [guess[2][0]+ 0.5*guess[2][2], np.inf, guess[2][2] + 2*guess[2][2]],
                [guess[3][0]+ 0.5*guess[3][2], np.inf, guess[3][2] + 2*guess[3][2]],
                [guess[4][0]+ 0.5*guess[4][2], np.inf, guess[4][2] + 2*guess[4][2]]]
    
    # Unpacks the embedded lists into flat tuples
    #   This is what the fit function requires as input
    gaus_param_bounds = (tuple([item for sublist in lo_bound for item in sublist]),
                        tuple([item for sublist in hi_bound for item in sublist]))
    
    # Flatten guess, for use with curve fit
    guess_flat = np.ndarray.flatten(guess)

    maxfev = 5000


    # # Fit the peaks
     # Check if any lower bound is not strictly less than its corresponding upper bound
    skip_cycle = False
    for lb, ub in zip(lo_bound, hi_bound):
        if not all(l < u for l, u in zip(lb, ub)):
            print(f"Skipping cycle #{cycle} due to invalid bounds.")
            skip_cycle = True
            break

    if skip_cycle:
        continue

    # If the check passes, proceed with fitting
    try:
        gaussian_params, _ = curve_fit(gaussian_function, xs, sig,
                                       p0=guess_flat, maxfev=maxfev, bounds=gaus_param_bounds)
        # Assuming the rest of your code for fitting and plotting remains unchanged
    except RuntimeError as e:
        print(f"Could not fit cycle #{cycle}: {e}")


    # Reshape gaussian_params from 1,15 to 3, 5 to feed into create peak params
    gaussian_params_reshape = gaussian_params.reshape((5,3))

    ## Bycycle fit
    peak_params = create_peak_params(xs, sig, gaussian_params_reshape)
    
    # initialize list of shape parameters
    shape_params = np.empty((len(peak_params), 7))
    peak_indices = np.empty((len(peak_params), 3))

    for ii, peak in enumerate(peak_params):

        # get peak indices
        start_index, peak_index, end_index = get_peak_indices(xs, sig, peak)

        # if the peak indices could not be determined, set all shape params to NaN
        if np.isnan(start_index) or np.isnan(end_index):
            shape_params[ii] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            peak_indices[ii] = [np.nan, np.nan, np.nan]
            continue

        # compute fwhm, rise-, and decay-time
        fwhm = xs[end_index] - xs[start_index]
        rise_time = xs[peak_index] - xs[start_index]
        decay_time = xs[end_index] - xs[peak_index]

        # compute rise-decay symmetry
        rise_decay_symmetry = rise_time / fwhm

        # compute sharpness
        half_mag = np.abs(peak[1] / 2)
        sharpness_rise = np.arctan(half_mag / rise_time) * (180 / np.pi) / 90
        sharpness_decay = np.arctan(half_mag / decay_time) * (180 / np.pi) / 90
        sharpness = 1 - ((180 - ((np.arctan(half_mag / rise_time) * (180 / np.pi)) + (np.arctan(half_mag / decay_time)) * (180 / np.pi))) / 180)

        # collect results
        shape_params[ii] = [fwhm, rise_time, decay_time, rise_decay_symmetry,
                            sharpness, sharpness_rise, sharpness_decay]
        # peak_indices[ii] = [start_index, peak_index, end_index]


    fit = gaussian_function(xs, *gaussian_params)

    # Calculate durations and intervals
    ecg_output.loc[cycle, 'p_duration'] = ecg_output.loc[cycle, 'p_off'] - ecg_output.loc[cycle, 'p_on'] 
    ecg_output.loc[cycle, 'pr_interval'] = ecg_output.loc[cycle, 'q_on'] - ecg_output.loc[cycle, 'p_on'] 
    ecg_output.loc[cycle, 'pr_segment'] = ecg_output.loc[cycle, 'q_on'] - ecg_output.loc[cycle, 'p_off'] 
    ecg_output.loc[cycle, 'qrs_duration'] = ecg_output.loc[cycle, 's_off'] - ecg_output.loc[cycle, 'q_on'] 
    ecg_output.loc[cycle, 'st_segment'] = ecg_output.loc[cycle, 't_off'] - ecg_output.loc[cycle, 's_off'] 
    ecg_output.loc[cycle, 'qt_interval'] = ecg_output.loc[cycle, 't_off'] - ecg_output.loc[cycle, 'q_on'] 


   # Calculate R-R interval if there's a previous R peak
    if previous_r_center is not None:
        r_r_interval = r_center - previous_r_center
        # Assign R-R interval to the DataFrame
        ecg_output.loc[cycle, 'rr_interval'] = r_r_interval
    else:
        # Assign NaN for the first cycle or if previous R peak is missing
        ecg_output.loc[cycle, 'rr_interval'] = np.nan

    # Calculate P-P interval if there's a previous P peak
    if previous_p_center is not None:
        p_p_interval = p_center - previous_p_center
        # Assign P-P interval to the DataFrame
        ecg_output.loc[cycle, 'pp_interval'] = p_p_interval
    else:
        # Assign NaN for the first cycle or if previous P peak is missing
        ecg_output.loc[cycle, 'pp_interval'] = np.nan

    # Update the previous peaks' locations
    previous_r_center = r_center
    previous_p_center = p_center

    r_squared = calc_r_squared(sig, fit)
    
    # if r_squared < 0.8:
    #     plt.plot(xs, sig, label='raw')
    #     plt.plot(xs, fit, label='fit')
    #     plt.show()     

             
    # if PLOT:
    #     plt.plot(xs, sig, label='raw')
    #     plt.plot(xs, fit, label='fit')

    #     # Plot the on/off points for each component as points
    #     for comp in ['p', 'q', 'r', 's', 't']:
    #         on_point = ecg_output.loc[cycle, f'{comp}_on']
    #         off_point = ecg_output.loc[cycle, f'{comp}_off']

    #         if not pd.isnull(on_point):
    #             on_y = sig[np.argmin(np.abs(xs - on_point))]
    #             plt.scatter(on_point, on_y, c='g', marker='o', label=f'{comp}_on')

    #         if not pd.isnull(off_point):
    #             off_y = sig[np.argmin(np.abs(xs - off_point))]
    #             plt.scatter(off_point, off_y, c='r', marker='o', label=f'{comp}_off')

    #     plt.legend()
    #     plt.title(f'Cycle #{cycle} - Gaussian Fit')
    #     plt.show()
    
    # Add features to dataframe
    ecg_output.iloc[cycle, 1:16] = gaussian_params
    ecg_output.loc[cycle, "r_squared"] = r_squared
    
    shape_params_flat = np.ndarray.flatten(shape_params)  
 
    ecg_output.loc[cycle, 25:60] =  shape_params_flat





# # Visualize Waveform Shape Features

# In[33]:


## Clean first so there are no Nans
cleaned_ecg_output = ecg_output[:].dropna()
ecg_feature_hist = generate_histograms(cleaned_ecg_output, cycle_column='cycle')

# Define the path where the file will be saved
hist_path = f'../docs/figures/{SUB_NUM}_timedomain_feat_hist.png'

# Save the figure using the figure object returned by the function
ecg_feature_hist.savefig(hist_path)


# ## Autocorrelations 
# 
# Plot the autocorrelations of each feature

# In[ ]:


## Clean first so there are no Nans
cleaned_ecg_output = ecg_output[:].dropna()


# #### ACF plots

# In[ ]:


# # Autocorrelation
# from statsmodels.tsa.stattools import acf as acf
# NLAGS = 100

# #### Height 
# fig = plt.gcf()

# for col in ecg_output.columns:
#     if 'height' in col:
#         plt.plot(np.arange(0,(NLAGS+1)), acf(ecg_output[col], nlags=NLAGS), 
#                  label=col)

# plt.legend()
# plt.title(f'Participant {SUB_NUM}: Component Height ACF')
# plt.show()

# ## Width 
# fig = plt.gcf()
# for col in ecg_output.columns:
#     if 'width' in col:
#         plt.plot(np.arange(0,(NLAGS+1)), acf(ecg_output[col], nlags=NLAGS), 
#                  label=col)
# plt.legend()
# plt.title(f'Participant {SUB_NUM}: Component Width ACF')
# plt.show()

# #### Duration
# fig = plt.gcf()

# for col in ecg_output.columns:
#     if 'duration' in col:
#         plt.plot(np.arange(0,(NLAGS+1)), acf(ecg_output[col], nlags=NLAGS), 
#                  label=col)

# plt.legend()
# plt.title(f'Participant {SUB_NUM}: Component Duration ACF')
# plt.show()

# ### Intervals
# #cleaned_RR_interval, pr_invterval, qt_interval, pp_interval
# fig = plt.gcf()
# for col in ecg_output.columns:
#     if 'interval' in col:
#         plt.plot(np.arange(0,(NLAGS+1)), acf(ecg_output[col], nlags=NLAGS), 
#                  label=col)
# plt.legend()
# plt.title(f'Participant {SUB_NUM}: Component Interval ACF')
# plt.show()

# ### Segment
# fig = plt.gcf()
# for col in ecg_output.columns:
#     if 'segment' in col:
#         plt.plot(np.arange(0,(NLAGS+1)), acf(ecg_output[col], nlags=NLAGS), 
#                  label=col)
# plt.legend()
# plt.title(f'Participant {SUB_NUM}: Component Segment ACF')
# plt.xlabel('Lags')
# plt.ylabel('..')
# plt.show()

# # fig.savefig(f'../figures/{SUB_NUM}_acf_width.png', transparent=False)


# ### NEW ACF Fitting Method (from Ryan)
# 
# - When you run an Autocorrelation Function (ACF) analysis with 200 lags on a signal that is 5 minutes long, what you're analyzing is how the signal correlates with itself up to a 0.2-second offset. 
# 
# - The analysis is focused on identifying patterns of self-similarity within a window that extends 0.2 seconds into the past (relative to any given point in time).

# In[31]:


# Instantiate the ACF class
acf = ACF()

# Select all columns except the first one
features_to_analyze = cleaned_ecg_output.columns[1:60]

# Initialize an empty DataFrame to store the results
acf_results_df = pd.DataFrame(columns=['Feature', 'tau', 'height', 'offset'])

# Iterate over each feature and compute its ACF and fit parameters
for feature in features_to_analyze:
    try:
        #print(f"Processing feature: {feature}")  # Print the feature being processed
        

        # Assuming acf is correctly set up for each loop iteration
        acf.compute_acf(cleaned_ecg_output[feature], fs, nlags=200)
        #print(f"ACF computation done for {feature}")  # Indicate ACF computation is done

        acf.fit()
       # print(f"Fitting done for {feature}")  # Indicate fitting is done
    
        # Extract the fitted parameters
        tau, height, offset = acf.params[:3]
       # print(f"Parameters extracted for {feature}")  # Indicate parameters extraction is done
    
        # Create a temporary DataFrame for the current feature's results
        temp_df = pd.DataFrame({
            'Feature': [feature], 
            'tau': [tau], 
            'height': [height], 
            'offset': [offset]
        })
    
        # Use pd.concat to append the results to acf_results_df DataFrame
        acf_results_df = pd.concat([acf_results_df, temp_df], ignore_index=True)
        #print(f"Results appended for {feature}")  # Indicate successful appending of results

    except Exception as e:
        #print(f"Error processing feature {feature}: {e}")  # Print error message
        break  # Optionally break the loop or continue to the next iteration

# Now, compute ACF for the entire signal, 'sig'
acf.compute_acf(sig, fs, nlags=10)  # Assuming 'fs' is your sampling frequency
acf.fit()

# Extract the fitted parameters for the whole signal
tau_sig, height_sig, offset_sig = acf.params[:3]

# Add the whole signal's ACF analysis results to the DataFrame
temp_df_sig = pd.DataFrame({
    'Feature': ['Whole Signal'],  # Indicating this row is for the whole signal
    'tau': [tau_sig],
    'height': [height_sig],
    'offset': [offset_sig]
})

# Append the results to the acf_results_df DataFrame
acf_results = pd.concat([acf_results_df, temp_df_sig], ignore_index=True)

# #Save output in new file 
acf_results.to_csv(f'../docs/saved_files/spectral_results/{SUB_NUM}_acf_results.csv')


# In[26]:


# Fit whole sig
acf = ACF()
acf.compute_acf(sig, fs, nlags=800)
acf.fit(with_cos=False, n_jobs=-1)
acf.plot()


# ## Additional ECG Features (time domain)

# ## Heart Rate and HRV 

# In[34]:


####### #### ####  Heart Rate and Avg HR  #### #### #### #### 

# Process the raw ECG signal (not the cleaned signal)
processed_data, _ = nk.ecg_process(ecg_notch, sampling_rate=1000)

# Access heart rate from the processed data
heart_rate = processed_data['ECG_Rate']

# Calculate average heart rate
average_heart_rate = heart_rate.mean()

print("Average Heart Rate:", average_heart_rate)


#########################

#########################

rr_intervals = (ecg_output['rr_interval'])
rr_intervals = rr_intervals[~np.isnan(rr_intervals)]  # Ensure no NaN values

if len(rr_intervals) > 1:  # Need at least two intervals for RMSSD
    # Calculate SDNN (using ddof=1 for sample standard deviation)
    sdnn = np.std(rr_intervals, ddof=1)
    
    # Calculate RMSSD
    diff_nn_intervals = np.diff(rr_intervals)
    squared_diff_nn_intervals = diff_nn_intervals ** 2
    rmssd = np.sqrt(np.mean(squared_diff_nn_intervals))
    
    print(f"SDNN: {sdnn:.2f} ms")
    print(f"RMSSD: {rmssd:.2f} ms")
else:
    print("Not enough data for HRV calculations.")


# Add the calculated values to the first row
ecg_output.loc[0, 'Average_Heart_Rate'] = average_heart_rate
ecg_output.loc[0, 'SDNN'] = sdnn if len(rr_intervals) > 1 else np.nan
ecg_output.loc[0, 'RMSSD'] = rmssd if len(rr_intervals) > 1 else np.nan

# Ensure the rest of the rows for these columns are filled with NaN
# This step is crucial if your DataFrame already has more than one row when you add these new columns.
if len(ecg_output) > 1:
    ecg_output.loc[1:, ['Average_Heart_Rate', 'SDNN', 'RMSSD']] = np.nan

print(ecg_output.head())  # Just printing the first few rows for verification


# In[35]:


#Save output in new file 
ecg_output.to_csv(f'../docs/saved_files/timedomain_results/{SUB_NUM}_ecg_output.csv')

3