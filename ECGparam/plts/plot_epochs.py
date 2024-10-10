import matplotlib.pyplot as plt

def plt_epochs(epochs_df, title="ECG Epochs", SAVE=False, file_path=None):
    """
    Plot the extracted ECG epochs.

    Parameters:
    epochs_df : DataFrame
        The DataFrame containing the extracted ECG epochs.
    title : str, optional
        The title for the plot (default is "ECG Epochs").
    SAVE : bool, optional
        Whether to save the plot to a file (default is False).
    file_path : str, optional
        File path to save the plot if SAVE is True. (default is None)
    """
    plt.figure(figsize=(10, 6))

    # Loop through the cycles and plot each one
    for cycle in epochs_df['cycle'].unique():
        cycle_data = epochs_df[epochs_df['cycle'] == cycle]
        plt.plot(cycle_data['signal_x'], cycle_data['signal_y'], alpha=0.5)

    # Title and labels
    plt.title(title, size=10)
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Signal (mV)')
    
    # Show plot
    plt.show()

    # Optionally save the plot
    if SAVE and file_path:
        plt.savefig(file_path)
