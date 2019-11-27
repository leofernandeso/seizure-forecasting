# Returns non-overlapping and equally spaced window ranges
def non_overlapping_windows(spacing, end, overlap=0):
    start = 0
    window_end = spacing
    windows = []
    while window_end <= end:
        windows += [(start, window_end)]
        if overlap:
            start += overlap
            window_end += overlap
        else:
            start += spacing
            window_end += spacing
    return windows
    

csv_train_output_path = "./../extracted_features_train.csv"
csv_val_output_path = "./../extracted_features_val.csv"
csv_test_output_path = "./../extracted_features_test.csv"
train_drop_out_filepath = "./../data/Train/drop_out_segments.txt"


epieco_parser_args = {
    'base_folder': "./../data",
    'windows_range': [*non_overlapping_windows(spacing=5, end=600, overlap=0)],     # in seconds
    'file_type': 'csv'
    #'windows_range': [(0, 200), (200, 400), (400, 600), (0, 400), (0, 600)], # defining it manually. they can also overlap!
}
epieco_folds_to_process = ['fold1Test.csv', 'fold1Train.csv', 'fold2Test.csv', 'fold2Train.csv', 
                           'fold3Test.csv', 'fold3Train.csv', 'fold4Test.csv', 'fold4Train.csv',
                           'fold5Test.csv', 'fold5Train.csv']

join_windows = False
single_channel_features_to_extract = {
    'time': ['skewness', 'peak_to_peak', 'mean', 'kurtosis', 'rms', 'zero_crossings', 
            'std', 'abs_mean', 'mobility', 'complexity'],
    'fourier': ['eeg_band_powers', 'spectral_edge_frequencies']
}

spatial_features_to_extract = {
    'spatial': ['time_domain_correlation', 'frequency_domain_correlation', 'nodes_degree']
              #'avg_clust_coeff', 'density',
              #  'global_efficiency', 'clustering_coeff', 'avg_shortest_path',
              #  'eigenvector_centrality'''
}


# chb_parser_args = dict(
#     base_folder="D:\\Faculdade\\TCC\\dados\\chb_mit",
#     preictal_window=15,
#     interictal_window=15, 
# )
