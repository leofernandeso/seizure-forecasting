import numpy as np
import math
import pickle
from matplotlib import pyplot as plt


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
        return dict(zero_crossings=zero_crossings_count)

    def rms(self):
        return dict(rms=np.sqrt(np.mean(self.signal**2)))
    
    def mean(self):
        return dict(mean=np.mean(self.signal))

    def compute_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented!")
        return features_dict

# test eeg segment
fs = 400
ts = 1/fs
with open('eeg_segment.p', 'rb') as pkl_file:
    signal = pickle.load(pkl_file)

t = np.arange(0, len(signal)) * ts
ff = TimeFeatures(signal, fs, ['mean', 'rms', 'zero_crossings'])
computed_features = ff.compute_features()
print(computed_features)