import os

def save_shape_params(shape_params_list, file_name='shape_params.csv', folder_name='output'):
    """
    Save shape parameters to a CSV file in a specified folder.

    Parameters
    ----------
    shape_params_list : list
        A list of dictionaries containing shape parameters for each ECG cycle.
    file_name : str, optional
        The name of the file to save the data (default is 'shape_params.csv').
    folder_name : str, optional
        The folder where the file will be saved (default is 'output').
    """
    # Dynamically create the path to save the file
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists
    
    # Full path for the file
    file_path = os.path.join(folder_path, file_name)
    
    # Save the file as a CSV
    pd.DataFrame(shape_params_list).to_csv(file_path, index=False)
    print(f"Shape parameters saved to {file_path}")
