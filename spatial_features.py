import numpy as np
import math
import pandas as pd
import pickle
from matplotlib import pyplot as plt

# Implement weighted graph. Use correlation between channels to assign edge weight values.
# Apply the following mathematical transformation in case one obtains negative correlation
# w = math.sqrt(1 + (calculated_correlation))
# Max weight = sqrt(2) (correlation = 1), min weight = 0 (correlation = -1)

# Consider distance between electrodes ?

class CorrelationFeatures():
    def __init__(self, channels_signals, channels_spectra, features_list=None):
        self.features_list = features_list
        self.n_channels = len(channels_signals)
        self.channels_signals = np.array(channels_signals)
        self.channels_spectra = np.array(channels_spectra)

    def _compute_correlation(self, metric_array, corr_type):
        """ Computes correlation features of a given array """
        transposed_channels = np.transpose(metric_array)
        df = pd.DataFrame(transposed_channels)
        corr_matrix = df.corr().values
        upper_diag_idxs = np.triu_indices(self.n_channels, 1)
        upper_diag_elems = corr_matrix[upper_diag_idxs]

        correlation_dict = {}
        for i, j, corr in zip(*upper_diag_idxs, upper_diag_elems):
            corr_id = 'ch_' + corr_type + '_' + str(i) + '_' + str(j)
            correlation_dict.update(
                {corr_id: corr}
            )
        return correlation_dict
    
    def time_domain_correlation(self):
        return self._compute_correlation(self.channels_signals, 'time')

    def frequency_domain_correlation(self):
        return self._compute_correlation(self.channels_spectra, 'freq')

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in FourierFeatures!")
        return features_dict