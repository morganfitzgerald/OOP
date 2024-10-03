# Import the os module for interacting with the operating system
import os

def subject_file_mapping(dir_path):
    """
    Creates mappings of subject numbers to their respective .dat and .hea file paths.

    This function scans a specified directory for files with '.dat' and '.hea' extensions,
    extracts the subject number from each file name, and maps these numbers to their
    corresponding file paths. This facilitates direct access to the file paths based on
    subject numbers.

    Parameters:
    - dir_path (str): The path to the directory containing the .dat and .hea files.

    Returns:
    - tuple: A tuple containing two dictionaries:
        - The first dictionary maps subject numbers to their .dat file paths.
        - The second dictionary maps subject numbers to their .hea file paths.
    """
    # Initialize dictionaries to store mappings
    files_dat_dict = {}
    files_hea_dict = {}

    for file in os.listdir(dir_path):
        if file.endswith('.dat'):
            # Extract and process subject number
            sub_num = str(int(os.path.splitext(file)[0]))
            files_dat_dict[sub_num] = os.path.join(dir_path, file)
        elif file.endswith('.hea'):
            # Extract and process subject number
            sub_num = str(int(os.path.splitext(file)[0]))
            files_hea_dict[sub_num] = os.path.join(dir_path, file)

    return files_dat_dict, files_hea_dict
