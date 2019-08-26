import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import pacf


class TimeFeatures():
    def __init__(self, signal, fs, features_list=None):
        self.features_list = features_list
        self.signal = signal
        self.fs = fs
        self.ts = 1/fs
        self.Np = len(signal)
        self._compute_base_variables()

    def _compute_base_variables(self):
        self.duration = self.ts * self.Np; 

    def zero_crossings(self):
        zero_crossings_count = (self.signal[:-1] * self.signal[1:] < 0).sum()
        return {'zero_crossings': zero_crossings_count}

    def rms(self):
        return {'rms': np.sqrt(np.mean(self.signal**2))}
    
    def mean(self):
        return {'mean': np.mean(self.signal)}

    def abs_mean(self):
        return {'abs_mean': np.abs(np.mean(self.signal))}

    def std(self):
        return {'std': np.std(self.signal)}

    def skewness(self):
        return {'skewness': skew(self.signal)}

    def peak_to_peak(self):
        peak_to_peak = np.amax(self.signal) - np.amin(self.signal)
        return {'peak_to_peak': peak_to_peak}

    def kurtosis(self):
        return {'kurtosis': kurtosis(self.signal)}

    def decorrelation_time(self):
        """ Returns decorrelation time (first zero-crossing of partial autocorrelation function).
            The time is returned in terms of the sample number """
        autocorrelation_function = pacf(self.signal)
        zero_crossings_idxs = np.where(autocorrelation_function[:-1] * autocorrelation_function[1:] < 0)[0]
        decorrelation_time = (zero_crossings_idxs[0] + 1) * self.ts # sample right after first zero crossing *  ts
        return {'decorrelation_time': decorrelation_time}

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in TimeFeatures!")
        return features_dict

# # test eeg segment
# fs = 400
# ts = 1/fs
# with open('eeg_segment.p', 'rb') as pkl_file:
#     signal = pickle.load(pkl_file)

# t = np.arange(0, len(signal)) * ts
# time_features = TimeFeatures(signal, fs, ['decorrelation_time', 'mean','skewness', 'peak_to_peak',
#                                'kurtosis', 'abs_mean', 'std', 'rms', 'zero_crossings'])
# computed_features = time_features.extract_features()
# print(computed_features)