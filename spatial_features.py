import numpy as np
import pandas as pd
import networkx as nx
import pickle
from matplotlib import pyplot as plt

# Implement weighted graph. Use correlation between channels to assign edge weight values.
# Apply the following mathematical transformation in case one obtains negative correlation
# w = np.sqrt(1 + (calculated_correlation))
# Max weight = sqrt(2) (correlation = 1), min weight = 0 (correlation = -1)

# Consider distance between electrodes ?


class SpatialFeatures():
    def __init__(self, channels_signals, channels_spectra, features_list=None):
        self.features_list = features_list
        self.n_channels = len(channels_signals)
        self.channels_signals = np.array(channels_signals)
        self.channels_spectra = np.array(channels_spectra)
        self.time_correlation = self._compute_correlation(self.channels_signals, 'time')
        self.freq_correlation = self._compute_correlation(self.channels_spectra, 'freq')
        self.G = self._compute_weighted_graph() 

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

    def _compute_weighted_graph(self, corr_type='time'):
        
        # defining connectivity measure
        if corr_type == 'time':
            corr_measure = self.time_correlation
        elif corr_type == 'freq':
            corr_measure = self.freq_correlation

        print(corr_measure)
        G = nx.Graph()

        for key, channel_corr in corr_measure.items():
            # node name processing
            splitted_key = key.split('_')
            node1 = int(splitted_key[2])
            node2 = int(splitted_key[3])

            # adding edge with correlation as weight
            G.add_edge(node1, node2, weight=channel_corr)

        print(G.degree())
        return G

    def time_domain_correlation(self):
        return self.time_correlation

    def frequency_domain_correlation(self):
        return self.freq_correlation

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