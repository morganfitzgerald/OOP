import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def threshold_cycles_rpeak(qrs_epochs_df, SUB_NUM, plot=True, save=False):
    """
    Analyzes QRS cycles from ECG data to identify and exclude outliers based on linear regression.

    This function performs linear regression on each cycle's y-values against the mean of all cycles' y-values.
    Cycles whose correlation coefficients are in the lowest 1% are considered outliers and are excluded.
    Optionally, this function can plot the remaining cycles and the mean of all cycles for visual inspection and save the plot.

    Parameters:
    - qrs_epochs_df (pd.DataFrame): DataFrame containing the QRS epoch data with at least 'signal_y' and 'cycle' columns.
    - SUB_NUM (int or str): Identifier for the subject/participant whose data is being analyzed.
    - plot (bool): If True, the function will generate and show a plot of the processed cycles. Default is True.
    - save (bool): If True, and if `plot` is also True, the plot will be saved to a file. Default is False.

    Returns:
    - tuple: A tuple containing the following elements:
        - idx_to_exclude (list of int): Indices of cycles considered as outliers and recommended for exclusion.
        - fig (matplotlib.figure.Figure or None): The figure object of the plot if `plot` is True; otherwise, None.

    The function saves a figure to '../figures/{SUB_NUM}_allcycles_cleaned.png' if `plot` and `save` are both True.

    Example usage:
    idx_to_exclude, fig = analyze_qrs_cycles(qrs_epochs_df, SUB_NUM=123, plot=True, save=True)
    """
    # Only include y values
    qrs_epochs_yvals = qrs_epochs_df[['signal_y', 'cycle']]

    # Take mean of all cycles
    qrs_epochs_yvals_piv = qrs_epochs_yvals.pivot(columns='cycle')
    all_cycles_mean = qrs_epochs_yvals_piv.mean(axis=1)

    # Perform linear regression on each cycle against all cycles mean
    rvals = np.zeros(len(qrs_epochs_yvals_piv.columns))
    for idx, col in enumerate(qrs_epochs_yvals_piv.columns):
        cycle = qrs_epochs_yvals_piv[col]
        reg_output = linregress(cycle, all_cycles_mean)
        rvals[idx] = reg_output.rvalue

    # Cycles to exclude based on quantile threshold
    exclude_threshold = np.quantile(rvals, 0.01)
    idx_to_exclude = [idx+1 for idx, r in enumerate(rvals) if r < exclude_threshold]

    # Plotting section
    fig = None
    if plot:
        fig, ax = plt.subplots()
        for idx, col in enumerate(qrs_epochs_yvals_piv.columns):
            if (idx+1) not in idx_to_exclude:
                ax.plot(qrs_epochs_yvals_piv[col], alpha=0.3)
        # ax.legend(['Top 99% Cleaned Cycles'], loc='best')
        ax.set_title(f"Participant {SUB_NUM}: Top 99% 'Cleaned' Cycles Aligned to P Onset", size=20)
        plt.tight_layout()
        if save:
            fig.savefig(f'../figures/{SUB_NUM}_allcycles_cleaned.png')
        if not save:  # Show the plot only if not saving, or always show depending on implementation choice
            plt.show()

    return idx_to_exclude
