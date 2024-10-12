# Import the necessary modules and classes from the ECGparam package
from ECGparam.objs.fit import ECGparam  # Import your ECGparam class
import os  # For handling file paths
import numpy as np  # For numerical operations (loading ECG signal)
import pandas as pd  # For handling DataFrames
from ECGparam.plts import plot_epochs, plot_rpeaks  # Import plotting functions
import neurokit2 as nk  # For processing the ECG signal

# Define the path to the ECG data file (assuming the file is a NumPy array)
ecg_data_path = '/Users/morganfitzgerald/Projects/ECG_tool_val/ECGparam/test_data/clean_ecgsim.npy'

# Load the ECG data from the .npy file
ecg_signal = np.load(ecg_data_path)

# Ensure the signal is in the correct shape (should be 1D)
print(f"ECG Signal Shape: {ecg_signal.shape}")
if ecg_signal.ndim != 1:
    raise ValueError("ECG signal must be one-dimensional.")

# Define the sampling frequency for the ECG signal (update this if your data has a different sampling rate)
sampling_rate = 1000  # Hz

# Initialize the ECG parameterization class
ecg_param = ECGparam()

# Try fitting the signal and handling any potential errors
try:
    # Run the ECG processing pipeline to extract features
    features = ecg_param.fit(ecg_signal, sampling_rate)
    
    # Get the cleaned ECG signal and the detected peaks (P onsets, R peaks, etc.)
    ecg_clean = ecg_param.ecg_clean
    signals = ecg_param.nk_signals  # Assuming this contains the relevant signals

    # Call the plot_aligned_ecg_cycles function to plot the ECG cycles aligned to P onsets
    plot_epochs(signals, ecg_clean, sampling_rate, align_to='ECG_P_Onsets', pre_offset=200, post_offset=200)

except Exception as e:
    # If there's an error, print the error message for debugging
    print(f"An error occurred during ECG processing: {e}")

# Define the output file path for saving the extracted features
output_file = os.path.join('/Users/morganfitzgerald/Projects/ECG_tool_val', 'ecg_features.csv')

# Save the extracted features to a CSV file
if features is not None:
    # Convert the features dictionary to a DataFrame for easy saving
    features_df = pd.DataFrame(features)

    # Save the DataFrame to a CSV file
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

else:
    print("No features were extracted. Please check your input signal and processing steps.")