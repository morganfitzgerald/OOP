#Visualizing individal cycle fits 

import numpy as np
import matplotlib.pyplot as plt

def plt_ecg_fit_with_components(cycle, epochs_df, ecg_output, fit, components=['p', 'q', 'r', 's', 't']):
    """
    Plot the original ECG signal, its Gaussian fit, and mark the on/off points for each ECG component.

    Parameters:
    - cycle: The cycle number to plot.
    - epochs_df: DataFrame containing the segmented ECG data (including the original signal).
    - ecg_output: DataFrame containing the feature extraction results (including on/off points for each component).
    - fit: The fitted Gaussian curve corresponding to the ECG signal for this cycle.
    - components: List of ECG components to mark on the plot (default is ['p', 'q', 'r', 's', 't']).
    """
    # Extract the cycle's data from epochs_df
    one_cycle = epochs_df.loc[epochs_df['cycle'] == cycle]
    xs = np.arange(one_cycle['index'].iloc[0], one_cycle['index'].iloc[-1] + 1)
    sig = np.asarray(one_cycle['signal_y'])

    # Plot the original signal and the Gaussian fit
    plt.figure(figsize=(10, 6))
    plt.plot(xs, sig, label='Original Signal', color='blue')
    plt.plot(xs, fit, label='Gaussian Fit', color='red', linestyle='--')

    # Retrieve R-squared value for the current cycle
    r_squared_value = ecg_output.loc[cycle, 'r_squared']
    print(f'R-squared for cycle #{cycle}: {r_squared_value}')

    # Mark the on/off points for each ECG component
    for comp in components:
        on_key = f'{comp}_on'
        off_key = f'{comp}_off'

        if on_key in ecg_output.columns and off_key in ecg_output.columns:
            on_point = ecg_output.loc[cycle, on_key]
            off_point = ecg_output.loc[cycle, off_key]

            # Mark on/off points on the plot
            if not np.isnan(on_point) and not np.isnan(off_point):
                plt.scatter([on_point, off_point], 
                            [sig[np.where(xs == on_point)[0][0]], sig[np.where(xs == off_point)[0][0]]],
                            marker='o', label=f'{comp.upper()} on/off')

    # Final plot customization
    plt.title(f'Cycle #{cycle}: Original Signal and Gaussian Fit')
    plt.xlabel('Sample Index')
    plt.ylabel()
