import pandas as pd
import matplotlib.pyplot as plt
import os
from rich import print


# Function to plot histograms
def plot_histogram(data, title, xlabel, ylabel, rotation=0, colors="blue"):
    plt.figure(figsize=(6, 4))
    data.plot(kind="bar", color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


# Setup directories and file paths
result_dir = "../docs/saved_files/timedomain_results/"
subject_info_file = "../docs/saved_files/subject-info.csv"


# Read and prepare sub∏ect information
subject_info_df = pd.read_csv(subject_info_file)
subject_info_df["ID"] = subject_info_df["ID"].astype(str).str.zfill(4)

# ##############################################
# Define mappings
# ##############################################

# Get the list of subject IDs from the result files
result_files = [f for f in os.listdir(result_dir) if f.endswith("_ecg_output.csv")]
subject_ids = [f.split("_")[0].zfill(4) for f in result_files]

# Filter subject_info_df to include only subjects that are in the result_files
filtered_subject_info_df = subject_info_df[subject_info_df["ID"].isin(subject_ids)]

age_ranges = {
    1: "18-19 years",
    2: "20-24 years",
    3: "25-29 years",
    4: "30-34 years",
    5: "35-39 years",
    6: "40-44 years",
    7: "45-49 years",
    8: "50-54 years",
    9: "55-59 years",
    10: "60-64 years",
    11: "65-69 years",
    12: "70-74 years",
    13: "75-79 years",
    14: "80-84 years",
    15: "85-92 years",
}
sex_values = {0: "Male", 1: "Female"}

# Map age ranges and sex classes
filtered_subject_info_df["Age_range"] = (
    filtered_subject_info_df["Age_group"].map(age_ranges).fillna("Unknown")
)
filtered_subject_info_df["Sex_class"] = (
    filtered_subject_info_df["Sex"].map(sex_values).fillna("Unknown")
)

# ##############################################
# Age and sex counts for run subs
# ##############################################

# Print age and sex counts for the filtered subjects
print("Age Range Counts for filtered subjects:")
print(filtered_subject_info_df["Age_range"].value_counts().sort_index())
print("\nSex Counts for filtered subjects:")
print(filtered_subject_info_df["Sex_class"].value_counts().sort_index())

# Plot histograms for Age_range and Sex_class
plot_histogram(
    filtered_subject_info_df["Age_range"].value_counts().sort_index(),
    "Histogram of Age Range Counts",
    "Age Range",
    "Counts",
    45,
)

plot_histogram(
    filtered_subject_info_df["Sex_class"].value_counts().sort_index(),
    "Histogram of Sex Counts",
    "Sex",
    "Counts",
)
