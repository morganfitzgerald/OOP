import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

##########################################
# Define constants
##########################################
RESULT_DIR = "../docs/saved_files/timedomain_results/"
SUBJECT_INFO_FILE = "../docs/saved_files/subject-info.csv"
SUBJECT_DETAILS_FILE = "../docs/saved_files/spectral_results/subject_details.csv"
SP_RESULTS_FILE = "../docs/saved_files/spectral_results/sp_results.csv"
PEAKS = ["p", "q", "r", "s", "t"]
AGE_RANGES = {
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
SEX_VALUES = {0: "Male", 1: "Female"}


##########################################
# Define functions
##########################################
def plot_bar_chart(data, title, xlabel, ylabel, rotation=0, colors="blue"):
    plt.figure(figsize=(6, 4))
    data.plot(kind="bar", color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_histogram(data, title, xlabel, ylabel, bins=20, color="skyblue"):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_box_plot(data, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, patch_artist=True, showfliers=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


##########################################
#  Read and prepare subject information
##########################################

subject_info_df = pd.read_csv(SUBJECT_INFO_FILE)

# Ensure the subject numbers are extracted and matched correctly
result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith("_ecg_output.csv")]
subject_nums = [f.split("_")[0].zfill(4) for f in result_files]

# Filter the DataFrame based on the extracted subject numbers
filtered_subject_info_df = subject_info_df[
    subject_info_df["ID"].astype(str).str.zfill(4).isin(subject_nums)
]
filtered_subject_info_df["Age_range"] = (
    filtered_subject_info_df["Age_group"].map(AGE_RANGES).fillna("Unknown")
)
filtered_subject_info_df["Sex_class"] = (
    filtered_subject_info_df["Sex"].map(SEX_VALUES).fillna("Unknown")
)

# # Print and plot age and sex counts
# print("Age Range Counts for filtered subjects:")
# print(filtered_subject_info_df['Age_range'].value_counts().sort_index())
# print("\nSex Counts for filtered subjects:")
# print(filtered_subject_info_df['Sex_class'].value_counts().sort_index())
# plot_bar_chart(filtered_subject_info_df['Age_range'].value_counts().sort_index(),
#                'Histogram of Age Range Counts for Filtered Sub', 'Age Range', 'Counts', 45)
# plot_bar_chart(filtered_subject_info_df['Sex_class'].value_counts().sort_index(),
#                'Histogram of Sex Counts for Filtered Sub', 'Sex', 'Counts', colors=['blue', 'red'])


##########################################
#  Pull out variables of interest
##########################################

# Extract and plot sharpness deriv values
sharpness_deriv_values = {peak: [] for peak in PEAKS}
r_squared_values = {}

for file_name in result_files:
    df = pd.read_csv(os.path.join(RESULT_DIR, file_name))
    subject_num = file_name.split("_")[0].zfill(4)

    r_squared_non_zero = df["r_squared"].dropna()[df["r_squared"] > 0]
    r_squared_values[subject_num] = r_squared_non_zero.mean()

    for peak in PEAKS:
        col = f"sharpness_deriv_{peak}"
        sharpness_deriv_values[peak].extend(df[col].dropna()[df[col] != 0])


# # Plot histograms and box plots for sharpness deriv values
# for peak in PEAKS:
#     plot_histogram(sharpness_deriv_values[peak], f'Histogram of Sharpness deriv ({peak.upper()})',
#                    f'Sharpness deriv ({peak.upper()})', 'Frequency')

# plot_box_plot([sharpness_deriv_values[peak] for peak in PEAKS],
#               'Box Plot of Sharpness Deriv for Each Peak', 'Peaks', 'Sharpness Deriv')
# plt.xticks(ticks=range(1, len(PEAKS) + 1), labels=[peak.upper() for peak in PEAKS], rotation=45)

# # Plot distribution of average R-squared values
# average_r_squared_values = list(r_squared_values.values())
# plot_histogram(average_r_squared_values, 'Distribution of Average $R^2$ Values',
#                'Average $R^2$ Value', 'Frequency')

# Print subjects with average R-squared < 0.8
for subject_num, average_r_squared in r_squared_values.items():
    if average_r_squared < 0.8:
        print(f"{subject_num} has averaged R-squared less than 0.8")


# Collect R-squared values for each participant
r_squared_values = {}
all_r_squared_values = []

for file_name in result_files:
    subject_num = file_name.split("_")[0].zfill(4)
    df = pd.read_csv(os.path.join(RESULT_DIR, file_name))
    r_squared_non_zero = df["r_squared"].dropna()[df["r_squared"] > 0]
    r_squared_values[subject_num] = r_squared_non_zero.tolist()
    all_r_squared_values.extend(r_squared_non_zero)

# Plot histogram for the distribution of R-squared values for each subject
plt.figure(figsize=(12, 6))
for subject_num, values in r_squared_values.items():
    plt.hist(values, bins=20, alpha=0.5, label=f"Subject {subject_num}")

plt.title("Distribution of $R^2$ Values for Each Subject")
plt.xlabel("$R^2$ Value")
plt.ylabel("Frequency")
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()


##########################################
#  # #Onset and offset
##########################################

# # Load and merge datasets for average exponent and offset values
# subject_details = pd.read_csv(SUBJECT_DETAILS_FILE)
# sp_results = pd.read_csv(SP_RESULTS_FILE)
# merged_data = pd.merge(subject_details, sp_results, on='Subject')
# merged_data = merged_data[merged_data['Age_range'] != 'Unknown']

# # Calculate and plot weighted averages
# mean_values = merged_data.groupby('Age_range')[['Offset_sp', 'Exponent_sp']].mean().reset_index()
# count_data = merged_data.groupby('Age_range')['Subject'].count().reset_index().rename(columns={'Subject': 'Count'})
# mean_values = mean_values.merge(count_data, on='Age_range')
# mean_values['Weighted_Offset_sp'] = mean_values['Offset_sp'] * mean_values['Count'] / mean_values['Count'].sum()
# mean_values['Weighted_Exponent_sp'] = mean_values['Exponent_sp'] * mean_values['Count'] / mean_values['Count'].sum()

# # Perform t-tests for significance testing
# age_20_to_30_ranges = ['20-24 years', '25-29 years']
# age_over_50_ranges = ['50-54 years', '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85-92 years']
# age_20_to_30 = merged_data[merged_data['Age_range'].isin(age_20_to_30_ranges)]
# age_over_50 = merged_data[merged_data['Age_range'].isin(age_over_50_ranges)]

# t_stat_exp, p_value_exp = ttest_ind(age_20_to_30['Exponent_sp'], age_over_50['Exponent_sp'], equal_var=False)
# t_stat_offset, p_value_offset = ttest_ind(age_20_to_30['Offset_sp'], age_over_50['Offset_sp'], equal_var=False)

# # Output t-test results
# print(f'Exponent_sp t-test (20-30 vs. >50): t={t_stat_exp}, p={p_value_exp}')
# print(f'Offset_sp t-test (20-30 vs. >50): t={t_stat_offset}, p={p_value_offset}')
