import pickle
import re
import numpy as np

import visualization

# configuration file : change it for different feature extraction or different window sizes
import feature_extractor_config as cfg

# implemented features
from fourier import FourierFeatures
from time_analysis import TimeFeatures
from spatial_features import SpatialFeatures

# epilepsy ecosystem parser
from parsing import EpiEcoParser

single_channel_feature_extractors_map = {
    'time': TimeFeatures,
    'fourier': FourierFeatures
}
spatial_feature_extractors_map = {
    'spatial': SpatialFeatures
}

# graph_features_map = {
#     'graph_theory': GraphFeatures
# }

class FeatureExtractor():
    def __init__(self, channels_array, fs, single_channel_features_to_extract=None, spatial_features_to_extract=None):
        self.channels_array = channels_array
        self.n_channels = len(channels_array)
        self.fs = fs
        self.ts = 1/fs
        self.single_channel_features_to_extract = single_channel_features_to_extract
        self.spatial_features_to_extract = spatial_features_to_extract
        self.channels_features = self._compute_single_channel_features()
        self.spatial_features = self._compute_spatial_features()

    def _compute_single_channel_features(self):
        channels_features = []
        for channel in self.channels_array:
            single_channel_features_dict = {}
            for k, val in self.single_channel_features_to_extract.items():
                single_channel_features_dict.update(
                    {k: single_channel_feature_extractors_map[k](channel, self.fs, val)}
                )
            channels_features.append(single_channel_features_dict)
        return channels_features

    def _compute_spatial_features(self):
        channels_signals = [c['time'].signal for c in self.channels_features]
        channels_spectra = [c['fourier'].power_spectral_density for c in self.channels_features]
        spatial_features_dict = {}
        for k, val in self.spatial_features_to_extract.items():
            spatial_features_dict.update(
                {k: spatial_feature_extractors_map[k](channels_signals, channels_spectra, val)}
            )
        return spatial_features_dict
    def extract_features(self):
        # Extracting single-channel features
        single_channel_features_dict = {}
        spatial_features_dict = {}
        for channel_idx, channel_feature in enumerate(self.channels_features):
            id_prefix = 'ch_' + str(channel_idx) + '_'
            for feature_type, feature_obj in channel_feature.items():
                extracted_features = feature_obj.extract_features()
                features_dict = {id_prefix+k:feature_val for k, feature_val in extracted_features.items()}
                single_channel_features_dict.update(features_dict)

        # Extracting multi-channel/spatial features
        for feature_type, feature_obj in self.spatial_features.items():
            extracted_features = feature_obj.extract_features()
            spatial_features_dict.update(extracted_features)

        return {**single_channel_features_dict, **spatial_features_dict}

def compute_windows_features(windows, fs):
    features_dict = {}
    for w_count, w in enumerate(windows):
        feature_extractor = FeatureExtractor(
                                            w, fs,
                                            single_channel_features_to_extract=cfg.single_channel_features_to_extract,
                                            spatial_features_to_extract=cfg.spatial_features_to_extract)
        window_features = feature_extractor.extract_features()
        w_prefix = 'w_' + str(w_count) + '_'
        window_features_with_updated_keys = {w_prefix+k: feature_val for k, feature_val in window_features.items()}
        features_dict.update(window_features_with_updated_keys)
    return features_dict
            
def main():

    data_parser = EpiEcoParser(**cfg.parser_args)
    fs = data_parser.fs
    segment_args = dict(
        patient_id=2,
        segment_id=655,
        _class=0
    )
    #df = data_parser.get_all_studies_data()
    channels = data_parser.get_full_train_segment(**segment_args)
    windows = data_parser.extract_windows(channels)
    features = compute_windows_features(windows, fs)
    print(features)
    print(len(features))
    #visualization.plot_eeg(channels, fs)

    #print(features)
    #print(len(features))
    
    

    
    

if __name__ == '__main__':
    main()