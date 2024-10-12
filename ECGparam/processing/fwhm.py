
def calc_fwhm(le_ind, ri_ind, center_index):
    """
    Estimate the Full-Width Half-Maximum (FWHM) of a peak based on its left and right indices at half maximum.
    
    Parameters:
    - le_ind: Left index at half maximum.
    - ri_ind: Right index at half maximum.
    - center_index: Index of the peak.
    
    Returns:
    - fwhm: Estimated Full-Width Half-Maximum.
    """
    if le_ind is None or ri_ind is None:
        return None  # or handle as needed
    short_side = min(abs(center_index - le_ind), abs(ri_ind - center_index))
    return short_side * 2