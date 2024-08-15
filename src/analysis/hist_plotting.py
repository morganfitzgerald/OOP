import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def generate_histograms(df, cycle_column='cycle'):
    # Define the order of columns based on the organization
    height_columns = ['p_height', 'q_height', 'r_height', 's_height', 't_height']
    width_columns = ['p_width', 'q_width', 'r_width', 's_width', 't_width']
    segment_columns = ['pr_segment', 'st_segment']
    interval_columns = ['pr_interval', 'qt_interval', 'pp_interval', 'rr_interval']
    duration_columns = ['qrs_duration', 'p_duration']

    # Combine all column lists into one ordered list
    column_order = height_columns + width_columns + segment_columns + interval_columns + duration_columns

    # Efficient filtering: Replace zeros with NaN and drop them
    filtered_df = df.replace(0, np.nan).dropna(subset=column_order)

    # Get unique cycles after filtering
    unique_cycles = filtered_df[cycle_column].unique()

    # Setup for subplots
    ncols = 5  # Adjust columns as per preference
    nrows = len(column_order) // ncols + (len(column_order) % ncols > 0)  # Calculate rows needed
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))  # Adjust figsize as needed
    axs = axs.flatten() if nrows * ncols > 1 else [axs]  # Flatten or wrap axs for iteration

    # Generate and plot histograms for each feature
    for i, column in enumerate(column_order):
        if i >= len(axs):  # Handle more subplots than columns
            break
        ax = axs[i]
        # Concatenate values for plotting, ensuring cycles are in the filtered set
        cycle_values = np.concatenate([filtered_df[filtered_df[cycle_column] == cycle][column].values for cycle in unique_cycles])
        ax.hist(cycle_values, bins=30, color='blue', edgecolor='black')
        ax.set_title(column.replace('_', ' ').title())  # Improve title readability
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
   
    return plt.gcf()
    # plt.show()
