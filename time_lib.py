import numpy as np
import math
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
import scipy.signal
import bottleneck as bn
import visualization

""" Possible implementations : 
    - Count number of spikes in the signal. Spike formulation can be thought as a deviation of 3*sigma from the mean, for example
    - Total intensity of spikes : something like the sum of the amplitude of all detected spikes.
"""

SPIKES_FACTOR = 0.08

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

def mobility(signal):
    signal_derivative = np.diff(signal)
    return np.sqrt(np.var(signal_derivative) / np.var(signal))

def complexity(signal):
    signal_derivative = np.diff(signal)
    return mobility(signal_derivative) / mobility(signal)

def skewness(signal):   
    return skew(signal)

def peak_to_peak(signal):
    peak_to_peak = np.amax(signal) - np.amin(signal)
    return peak_to_peak

def kurtosis(signal):
    return kurt(signal)



""" Fix spike features. Do some reserach on how to calculate these spikes. Right now, these features won't help at all"""
def spikes_intensity(signal):
    b, a = scipy.signal.butter(N=1, Wn=0.8, btype='high')
    y = scipy.signal.filtfilt(b, a, signal)
    signal_peak_to_peak = peak_to_peak(signal)    
    spikes_array = y[abs(y) >= signal_peak_to_peak*SPIKES_FACTOR]
    return sum( abs(spikes_array) )

def nr_spikes(signal):
    b, a = scipy.signal.butter(N=1, Wn=0.8, btype='high')
    y = scipy.signal.filtfilt(b, a, signal)
    signal_peak_to_peak = peak_to_peak(signal)    
    spikes_array = y[abs(y) >= signal_peak_to_peak*SPIKES_FACTOR]
    return len(spikes_array)



# def decorrelation_time(signal):
#     """ Returns decorrelation time (first zero-crossing of partial autocorrelation function).
#         The time is returned in terms of the sample number """
#     """ Find a way to make it faster. It's taking too long to compute! """
#     autocorrelation_function = pacf(signal)
#     zero_crossings_idxs = np.where(autocorrelation_function[:-1] * autocorrelation_function[1:] < 0)[0]
#     decorrelation_time = (zero_crossings_idxs[0] + 1)   #sample right after first zero crossing *  ts
#     return decorrelation_time