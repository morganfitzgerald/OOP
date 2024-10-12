def calc_bounds(center, height, std, bound_factor, flip_height=False):
    """
    Calculate lower and upper bounds for the parameters of a Gaussian component.

    Parameters:
    - center (float): The central value of the Gaussian component (mean of the distribution).
    - height (float): The height of the Gaussian peak (amplitude).
    - std (float): The standard deviation of the Gaussian component (width of the peak).
    - bound_factor (float): A factor to scale the bounds for height and std.
    - flip_height (bool, optional): If True, flips the bound factor for height (default is False).

    Returns:
    - lower_bound (list): A list containing the lower bounds for center, height, and std.
    - upper_bound (list): A list containing the upper bounds for center, height, and std.
    """
    
    # Center is bounded by ± bound_factor * std
    center_lower_bound = center - bound_factor * std
    center_upper_bound = center + bound_factor * std
    
    # Flip the height bounds if flip_height is True
    if flip_height:
        lower_height = height * (1 + bound_factor)
        upper_height = height * (1 - bound_factor)
    else:
        lower_height = height * (1 - bound_factor)
        upper_height = height * (1 + bound_factor)

    # Standard deviation bounds should not go negative, use bound_factor to define limits
    lower_std = max(std * (1 - bound_factor), 1e-6)  # Use a small positive value for minimum std
    upper_std = std * (1 + bound_factor)
    
    # Return lower and upper bounds as lists
    lower_bound = [center_lower_bound, lower_height, lower_std]
    upper_bound = [center_upper_bound, upper_height, upper_std]
    
    return lower_bound, upper_bound