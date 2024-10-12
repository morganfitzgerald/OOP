# plts/plot_rpeaks.py
import matplotlib.pyplot as plt

def plot_rpeaks(clean_ecg_sig, rpeaks):
    """
    Plot the ECG signal with R peaks marked.

    Parameters:
    - clean_ecg_sig: ndarray. The cleaned ECG signal.
    - rpeaks: ndarray. The positions of R peaks.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(clean_ecg_sig, label='ECG Signal', color='blue')
    
    # Mark the R peaks
    plt.scatter(rpeaks, clean_ecg_sig[rpeaks], color='red', label='R Peaks', zorder=5)
    
    # Add labels and legend
    plt.title('ECG Signal with R Peaks')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
