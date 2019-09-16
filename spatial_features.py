import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time
import itertools
from scipy.signal import correlate, coherence
from scipy.integrate import trapz
import graph_utils
import visualization
import collections
from matplotlib import pyplot as plt


# Question : use max from cross correlation instead of simple correlation ? Idea in doing it :
# account for possible lags between electrodes

# For example, you
# could select a threshold of one standard deviation above the median connectivity value.


keep_weights = True
threshold = True
COHERENCE_MIN_FREQ = 0
COHERENCE_MAX_FREQ = 50

class SpatialFeatures():
    def __init__(self, channels_signals, channels_spectra, fs, features_list=None):
        self.features_list = features_list
        self.n_channels = len(channels_signals)
        self.fs = fs
        self.channels_signals = np.array(channels_signals)
        self.channels_spectra = np.array(channels_spectra)        
        self.time_correlation = self._compute_correlation(channels_signals, 'time')
        self.chann_coherence = self._compute_channel_coherence()
        self.G, self.G_thresh = self._compute_weighted_graphs()  # builds a graph and thresholds it

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

    def _compute_channel_coherence(self):
        channels_list = list(range(0, self.n_channels))
        coherence_dict = {}
        for c1, c2 in itertools.combinations(channels_list, 2):            
            f, Cxy = coherence(self.channels_signals[c1], self.channels_signals[c2], self.fs, noverlap=0)
            cropped_Cxy = Cxy[(f >= COHERENCE_MIN_FREQ) & (f < COHERENCE_MAX_FREQ)]
            freq_res = f[1] - f[0]
            band_mean_coherence = trapz(cropped_Cxy, dx=freq_res)
            coherence_id = 'ch_coher_{}_{}'.format(c1, c2)
            
            coherence_dict.update(
                {coherence_id: band_mean_coherence}
            )
        return coherence_dict

    def _compute_weighted_graphs(self):
        
        # defining connectivity measure
        conn_measure = self.chann_coherence

        G = nx.Graph()

        for key, channel_conn in conn_measure.items():
            # node name processing
            splitted_key = key.split('_')
            node1 = int(splitted_key[2])
            node2 = int(splitted_key[3])
            
            # adding edge with correlation as weight
            w = channel_conn  # perform any transformation here
            G.add_edge(node1, node2, weight=w)

        G_thresh = graph_utils.threshold_graph(G, keep_weights=keep_weights)
        return G, G_thresh

    def brain_connectivity(self):

        channels_list = list(range(0, self.n_channels))
        count = 0

        connectivity_array = np.zeros((self.n_channels, self.n_channels))
        for c1, c2 in itertools.combinations(channels_list, 2):
            cross_corr = scipy.signal.correlate(self.channels_signals[c1], self.channels_signals[c2], method='auto')
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

    def degree_entropy(self):
        return {'degree_entropy': graph_utils.degree_dist_entropy(self.G_thresh)}

    def nr_components(self):
        return {'nr_components': nx.number_connected_components(self.G_thresh)}

    def avg_shortest_path(self):
        avg_shtst_path = nx.average_shortest_path_length(self.G, weight='weight')
        return {'avg_shtst_path': avg_shtst_path}

    def nodes_after_threshold(self):
        return {'nodes_tresh': len(self.G_thresh)}
    
    #def transitivity(self):
    """ 
        Problem : disconnected components after threshold : trans = 0.
        Full graph : always 1
    """
    #    return {'transitivity': nx.transitivity(self.G)}

    def avg_clust_coeff(self):
        return {'avg_clust_coeff': nx.average_clustering(self.G, weight='weight')}

    def global_efficiency(self):
        return {'glob_efficiency': nx.global_efficiency(self.G_thresh)}

    def eigenvector_centrality(self):
        eig_cent_dict = nx.eigenvector_centrality(self.G, weight='weight')
        return {'eigenv_centr_'+str(ch): np.log2(eig_cent) for ch, eig_cent in eig_cent_dict.items()}

    # def eccentricity(self):
    #     print(nx.eccentricity(self.G))
    #     return {'eccentricty': 1}

    def density(self):
        return {'density': nx.density(self.G_thresh)}

    def nodes_degree(self):
        degrees_dict = {}
        degrees = self.G.degree(weight='weight')
        for d in degrees:
            ch, degree = d
            degrees_dict.update(
                {'degree_'+str(ch): degree}
            )
        return degrees_dict

    def clustering_coeff(self):
        clustering_coeff = nx.clustering(self.G, weight='weight')
        return {'clust_coeff_'+str(ch): np.log2(clust_coeff) for ch, clust_coeff in clustering_coeff.items()}

    def time_domain_correlation(self):
        return self.time_correlation

    def frequency_domain_correlation(self):
        return self.freq_correlation

    def channel_coherence(self):
        return self.chann_coherence

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in SpatialFeatures!")
        return features_dict