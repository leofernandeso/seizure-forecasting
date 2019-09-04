import numpy as np
import pandas as pd
import networkx as nx
import pickle
import itertools
import scipy
from matplotlib import pyplot as plt


# Question : use max from cross correlation instead of simple correlation ? Idea in doing it :
# account for possible lags between electrodes


class SpatialFeatures():
    def __init__(self, channels_signals, channels_spectra, features_list=None):
        self.features_list = features_list
        self.n_channels = len(channels_signals)
        self.channels_signals = np.array(channels_signals)
        self.channels_spectra = np.array(channels_spectra)
        self.brain_conn = self.brain_connectivity()
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
        print(len(correlation_dict))
        return correlation_dict

    def brain_connectivity(self):

        channels_list = list(range(0, self.n_channels))
        count = 0

        connectivity_array = np.zeros((self.n_channels, self.n_channels))
        for c1, c2 in itertools.combinations(channels_list, 2):
            cross_corr = scipy.signal.correlate(self.channels_signals[c1], self.channels_signals[c2], method='fft')
            brain_conn = max(cross_corr)
            connectivity_array[c1][c2] = brain_conn
            connectivity_array[c2][c1] = brain_conn            
            count += 1
        
        connectivity_array = connectivity_array / np.amax(connectivity_array)
        
        connectivity_dict = {}
        for c1, c2 in itertools.combinations(channels_list, 2):
            k = "brain_conn_{}_{}".format(c1,c2)
            connectivity_dict.update(
                {k: connectivity_array[c1][c2]}
            )

        return connectivity_dict

    def _compute_weighted_graph(self):
        
        # defining connectivity measure
        conn_measure = self.brain_conn

        G = nx.Graph()

        for key, channel_conn in conn_measure.items():
            # node name processing
            splitted_key = key.split('_')
            node1 = int(splitted_key[2])
            node2 = int(splitted_key[3])
            
            # adding edge with correlation as weight
            G.add_edge(node1, node2, weight=channel_conn)

        return G

    def time_domain_correlation(self):
        return self._compute_correlation(self.channels_signals, 'time')

    def frequency_domain_correlation(self):
        return self._compute_correlation(self.channels_spectra, 'freq')

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            print(feature_name)
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in FourierFeatures!")
        return features_dict