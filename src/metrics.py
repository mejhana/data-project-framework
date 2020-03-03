# -*- coding: utf-8 -*-
"""
Task : To describe the metrics to be used to judge model performance
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def mean_square_error(actual, predicted):
    """
    Mean Square Error
    """
    return (mean_squared_error(actual, predicted))

def root_mean_square_error(actual, predicted):
    """
    Root Mean Squared Error
    """
    return (np.sqrt(mean_squared_error(actual, predicted)))

def mean_abs_error(actual, predicted):
    """
    Mean Absolute Error
    """
    return(mean_absolute_error(actual, predicted))

