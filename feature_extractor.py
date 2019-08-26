import pickle
import numpy as np
from fourier import FourierFeatures
from time_analysis import TimeFeatures
from spatial_features import CorrelationFeatures

single_channel_feature_extractors_map = {
    'time': TimeFeatures,
    'fourier': FourierFeatures
}
spatial_feature_extractors_map = {
    'correlation': CorrelationFeatures
}

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
            for k, val in zip(self.single_channel_features_to_extract.keys(), self.single_channel_features_to_extract.values()):
                single_channel_features_dict.update(
                    {k: single_channel_feature_extractors_map[k](channel, self.fs, val)}
                )
            channels_features.append(single_channel_features_dict)
        return channels_features
    def _compute_spatial_features(self):
        channels_signals = [c['time'].signal for c in self.channels_features]
        channels_spectra = [c['fourier'].amplitude_spectrum for c in self.channels_features]
        spatial_features_dict = {}
        for k, val in self.spatial_features_to_extract.items():
            spatial_features_dict.update(
                {k: spatial_feature_extractors_map[k](channels_signals, channels_spectra, val)}
            )
        return spatial_features_dict
    def extract_features(self):
        # Extracting single-channel features
        features_dict = {}
        for channel_idx, channel_feature in enumerate(self.channels_features):
            id_prefix = 'ch_' + str(channel_idx) + '_'
            for feature_type, feature_obj in channel_feature.items():
                extracted_features = feature_obj.extract_features()
                print({id_prefix+k:1 for k in extracted_features.keys()})

        # Next -> extract spatial features and build dict for given segment
            
def main():

    fs = 400 # sampling frequency    
    single_channel_features_to_extract = {
        'time': ['skewness', 'peak_to_peak', 'mean', 'kurtosis'],
        'fourier': ['eeg_band_energies']
    }
    spatial_features_to_extract = {
        'correlation': ['time_domain_correlation', 'frequency_domain_correlation']
        #'graph_theory': []
    }

    # Example 32 channels EEG segment
    with open('eeg_segment_32channels.p', 'rb') as pkl_file:
        channels = pickle.load(pkl_file)

    feature_extractor = FeatureExtractor(
                                        channels, fs,
                                        single_channel_features_to_extract=single_channel_features_to_extract,
                                        spatial_features_to_extract=spatial_features_to_extract)
    print(feature_extractor.extract_features())
    #print(feature_extractor.spatial_features['correlation'].extract_features())

if __name__ == '__main__':
    main()