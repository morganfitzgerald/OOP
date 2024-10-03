import pandas as pd
import os

# Directory containing the result files
folder_path = "../docs/saved_files/spectral_results/"

# List all files in the folder
all_files = os.listdir(folder_path)

# Filter out the specparam_results files
specparam_files = [f for f in all_files if "specparam_results.csv" in f]

# Initialize an empty DataFrame to hold all the combined data
sp_results_combined_df = pd.DataFrame()

# Loop through each file, read it, and append it to the combined_df
for file in specparam_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a DataFrame
    # Assuming the first unnamed column can be skipped (usecols to select columns starting from the second one)
    df = pd.read_csv(
        file_path,
        usecols=[1, 2, 3, 4, 5],
        header=None,
        skiprows=[0],
        names=["SUM_NUM", "Offset_sp", "Exponent_sp", "Error_sp", "R^2_sp"],
    )

    # Append the dataframe to the combined_df
    sp_results_combined_df = pd.concat([sp_results_combined_df, df], ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_csv_path = os.path.join(folder_path, "combined_specparam_results.csv")
sp_results_combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined file created at: {combined_csv_path}")
