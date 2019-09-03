import numpy as np
import pickle
from matplotlib import pyplot as plt
import time_lib


""" Implement calculation methods outside of class in order to make
    it possible to generate features from first and second derivative."""

class TimeFeatures():
    def __init__(self, signal, fs, features_list=None):
        self.features_list = features_list
        self.signal = signal
        self.fs = fs
        self.ts = 1/fs
        self.Np = len(signal)
        self._compute_base_variables()

    def _compute_base_variables(self):
        self.signal_first_derivative = np.diff(self.signal)
        self.signal_second_derivative = np.diff(self.signal_first_derivative)

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(time_lib, feature_name) # feature to be calculated
                # Signal feature
                output_params = method_to_call(self.signal)
                features_dict.update({feature_name: output_params})
                
                # First derivative feature
                output_params = method_to_call(self.signal_first_derivative)
                features_dict.update({'delta_' + feature_name: output_params})

                # Second derivative feature
                output_params = method_to_call(self.signal_second_derivative)
                features_dict.update({'delta2_' + feature_name: output_params})

            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in TimeFeatures!")
        return features_dict

# #test eeg segment
# fs = 400
# ts = 1/fs
# with open('eeg_segment.p', 'rb') as pkl_file:
#     signal = pickle.load(pkl_file)

# t = np.arange(0, len(signal)) * ts
# time_features = TimeFeatures(signal, fs, ['decorrelation_time', 'mean','skewness', 'peak_to_peak',
#                                'kurtosis', 'abs_mean', 'std', 'rms', 'zero_crossings'])
# computed_features = time_features.extract_features()