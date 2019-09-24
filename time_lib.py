import numpy as np
import math
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
from statsmodels.tsa.stattools import pacf


def zero_crossings(signal):
    zero_crossings_count = (signal[:-1] * signal[1:] < 0).sum()
    return zero_crossings_count

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def mean(signal):
    return np.mean(signal)

def abs_mean(signal):
    return np.abs(np.mean(signal))

def std(signal):
    return np.std(signal)

def skewness(signal):
    return skew(signal)

def peak_to_peak(signal):
    peak_to_peak = np.amax(signal) - np.amin(signal)
    return peak_to_peak

def kurtosis(signal):
    return kurt(signal)

def decorrelation_time(signal):
    """ Returns decorrelation time (first zero-crossing of partial autocorrelation function).
        The time is returned in terms of the sample number """
    """ Find a way to make it faster. It's taking too long to compute! """
    autocorrelation_function = pacf(signal)
    zero_crossings_idxs = np.where(autocorrelation_function[:-1] * autocorrelation_function[1:] < 0)[0]
    decorrelation_time = (zero_crossings_idxs[0] + 1)   #sample right after first zero crossing *  ts
    return decorrelation_time