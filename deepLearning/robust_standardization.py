import torch
import numpy as np

def robust_standardization(time_series):
    """
    Robust standardization of a time series.

    Parameters:
    - time_series: numpy array, the input time series.

    Returns:
    - standardized_series: numpy array, 
        the robustly standardized time series.
    """

    # Calculate the median and interquartile range (IQR)
    median_value = np.median(time_series)
    lower_quantile = np.percentile(time_series, 25)
    upper_quantile = np.percentile(time_series, 75)
    iqr = upper_quantile - lower_quantile

    # Robust standardization equation
    standardized_series = (time_series - median_value) / iqr

    return standardized_series




