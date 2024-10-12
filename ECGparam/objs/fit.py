import neurokit2 as nk

# Import your custom functions
from ..processing import (high_pass_filter, notch_filter, extract_control_points, epoch_cycles, extract_features)


class ECGBase:
    """Base class for shared functionalities and attributes.
    
    This class provides core attributes and methods that can be reused in derived classes,
    such as ECG signal processing classes. It handles common setup tasks like loading signals,
    setting filter parameters, and managing feature extraction settings.
    """

    def __init__(self, filter_params=None, feature_params=None):
        """
        Initialize common attributes for ECG processing.
        
        Parameters
        ----------
        filter_params : dict, optional
            Dictionary containing parameters for filtering the ECG signal, such as the cutoff frequency for high-pass
            filtering and the frequency of the notch filter to remove power line noise. If not provided, default values are used.
        feature_params : dict, optional
            Dictionary containing parameters for feature extraction, such as settings for peak detection. 
            If not provided, default values are used.
        """
        # Set default filter parameters if none are provided. These control the filters applied to the ECG signal.
        self.filter_params = filter_params if filter_params else {
            'high_pass_cutoff': 0.05,  # High-pass filter cutoff frequency to remove low-frequency noise
            'notch_frequency': 50,     # Notch filter frequency to remove power line interference (50Hz)
            'quality_factor': 30       # Quality factor for the notch filter, determining its bandwidth
        }

        # Set default parameters for feature extraction, specifically for detecting peaks in the ECG signal.
        self.feature_params = feature_params if feature_params else {
            'find_peaks_kwargs': {'prominence': 0.2}  # Peak detection settings, like required prominence of peaks
        }

        # Initialize attributes to hold the ECG signal, its sampling frequency, filtered signal, and extracted features.
        self.sig = None               # Placeholder for the original ECG signal
        self.fs = None                # Placeholder for the sampling frequency of the ECG signal
        self.filtered_signal = None   # Placeholder for the filtered version of the ECG signal
        self.features = None          # Placeholder for storing extracted features after processing

    def load_signal(self, signal, fs):
        """
        Load and validate the input ECG signal.
        
        Parameters
        ----------
        signal : array-like
            The ECG signal to be processed. It must be a one-dimensional array representing the voltage over time.
        fs : int or float
            The sampling frequency of the signal, in Hz.
        
        Raises
        ------
        ValueError
            If the input signal is not one-dimensional.
        """
        # Check if the input signal is one-dimensional (which it must be for ECG processing).
        if signal.ndim != 1:
            raise ValueError('Signal must be 1-dimensional.')  # Raise an error if the signal has more than one dimension
        
        # Store the validated ECG signal and sampling frequency as attributes.
        self.sig = signal  # Save the ECG signal
        self.fs = fs       # Save the sampling frequency


###################################################################################################
###################################################################################################


class ECGparam(ECGBase):
    """Main class for ECG parameter extraction.
    
    This class handles the full ECG processing pipeline, including filtering, R-peak detection,
    segmentation into cycles, and feature extraction. It inherits shared functionality and attributes
    from the `ECGBase` class, allowing the use of filter parameters, and feature extraction settings.
    """

    def __init__(self, filter_params=None, feature_params=None):
        """
        Initialize the ECGparam class with  filter parameters, and feature extraction settings.
        
        Parameters
        ----------
        filter_params : dict, optional
            Dictionary of filter parameters (e.g., high-pass and notch filter settings). If None, default values are used.
        feature_params : dict, optional
            Dictionary of parameters for feature extraction (e.g., settings for peak detection). If None, default values are used.
        """
        # Call the constructor of the parent class (ECGBase) to initialize common attributes
        super().__init__(filter_params, feature_params)

    def _apply_filters(self, signal):
        """
        Apply high-pass and notch filters to the input ECG signal to remove low-frequency drifts and power line noise.
        
        Parameters
        ----------
        signal : array-like
            The raw ECG signal to be filtered.
        
        Returns
        -------
        filtered_signal : array-like
            The ECG signal after applying high-pass and notch filters.
        """
        # Apply high-pass filter to remove low-frequency drifts
        filtered_signal = high_pass_filter(signal, self.fs, cutoff_frequency=self.filter_params['high_pass_cutoff'])
        # Apply notch filter to remove power line noise (e.g., 50/60 Hz)
        filtered_signal = notch_filter(filtered_signal, self.fs, f0=self.filter_params['notch_frequency'], 
                                       quality_factor=self.filter_params['quality_factor'])
        return filtered_signal

    def fit(self, signal, fs):
        """
        Run the ECG processing pipeline on the input signal, including filtering, R-peak detection,
        segmentation into cycles, and feature extraction.
        
        Parameters
        ----------
        signal : array-like
            The raw ECG signal to be processed.
        fs : int or float
            The sampling frequency of the ECG signal.
        
        Returns
        -------
        features : dict
            A dictionary of extracted features from the segmented ECG cycles.
        """
        # Load and validate the input signal, storing it and the sampling frequency as attributes
        self.load_signal(signal, fs)

        # Apply high-pass and notch filters to the signal to remove noise and drifts
        self.filtered_signal = self._apply_filters(self.sig)

        # Use NeuroKit2's method to clean the filtered signal, ensuring a smooth baseline
        self.ecg_clean = nk.ecg_clean(self.filtered_signal, sampling_rate=self.fs)

        # Detect signals using NeuroKit2's peak detection method
        _, self.nk_signals = extract_control_points(self.ecg_clean, sampling_rate=self.fs)

        # Segment the ECG signal into individual cycles using the detected p-peaks
        # and filtered signal, and return the segmented cycles as a DataFrame
        epochs_df = epoch_cycles(self.nk_signals, self.filtered_signal, self.fs)  # Call `epoch_cycles` directly

        # Extract features from the segmented ECG cycles (P, Q, R, S, T components, intervals, etc.)
        self.features = extract_features(epochs_df, self.fs)

        # Return the dictionary of extracted features
        return self.features
