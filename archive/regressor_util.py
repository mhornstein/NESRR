'''
This utility class provides common functions that are shared among all regressors.
'''

import numpy as np

def transform_mi(series, transformation_type):
    '''
    returns a scaled version of a series containing mi scores.
    :param series: the mi scores
    :param transformation_type: string representing the required transformaion. can be either None, minmax, ln, or sqrt
    :return:
    '''
    if transformation_type == 'None':
        scaled_series = series
    elif transformation_type == 'sqrt':
        scaled_series = np.sqrt(series)
    elif transformation_type == 'ln':
        scaled_series = -1 / np.log(series)
    elif transformation_type == 'log10':
        scaled_series = -1 / np.log10(series)
    else:
        raise ValueError(f'Unknown MI transformation: {transformation_type}')
    return scaled_series
