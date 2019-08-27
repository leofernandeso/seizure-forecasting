import pickle
import numpy as np
from fourier import FourierFeatures
from time_analysis import TimeFeatures
from spatial_features import CorrelationFeatures
from parsers.epieco import EpiEcoParser

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
        #print("single_channel_features : {}".format(len(single_channel_features_dict)))
        #print("spatial_features : {}".format(len(spatial_features_dict)))
        return {**single_channel_features_dict, **spatial_features_dict}

            
def main():
    data_parser = EpiEcoParser("D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem")
    fs = data_parser.fs
    parser_args = dict(
        patient_id=2,
        segment_id=385,
        _class=0
    )
    channels = data_parser.get_train_segment(**parser_args)

    single_channel_features_to_extract = {
        'time': ['skewness', 'peak_to_peak', 'mean', 'kurtosis', 'rms', 'zero_crossings', 
                'std', 'abs_mean'],
        'fourier': ['eeg_band_energies']
    }
    spatial_features_to_extract = {
        'correlation': ['time_domain_correlation', 'frequency_domain_correlation']
    }

    feature_extractor = FeatureExtractor(
                                        channels, fs,
                                        single_channel_features_to_extract=single_channel_features_to_extract,
                                        spatial_features_to_extract=spatial_features_to_extract)
    extracted_features = feature_extractor.extract_features()
    print(extracted_features)
    print(len(extracted_features.keys()))

if __name__ == '__main__':
    main()