import pickle
import numpy as np
from fourier import FourierFeatures
from time_analysis import TimeFeatures

feature_extractors_map = {
    'time': TimeFeatures,
    'fourier': FourierFeatures
}

class FeatureExtractor():
    def __init__(self, channels_array, fs, features_to_extract=None):
        self.channels_array = channels_array
        self.n_channels = len(channels_array)
        self.fs = fs
        self.ts = 1/fs
        self.features_to_extract = features_to_extract
        self.feature_classes = list(features_to_extract.keys())

    def compute_single_channel_features(self):
        self.channels_features = []
        for channel in self.channels_array:
            single_channel_features_dict = {}
            for k, val in zip(self.features_to_extract.keys(), self.features_to_extract.values()):
                single_channel_features_dict.update(
                    {k: feature_extractors_map[k](channel, self.fs, val)}
                )
            self.channels_features.append(single_channel_features_dict)

        # channels_fourier_features = []
        # channels_time_features = []
        # for channel in self.channels_array:
        #     channels_fourier_features.append(
        #         FourierFeatures(channel, self.fs, self.features_to_extract['fourier'])
        #     )
        #     channels_time_features.append(
        #         TimeFeatures(channel, self.fs, self.features_to_extract['time'])
        #     )
        # self.channels_fourier_features = np.array(channels_fourier_features)
        # self.channels_time_features = np.array(channels_time_features)
    #def extract_features(self):



def main():

    fs = 400 # sampling frequency

    with open('eeg_segment_32channels.p', 'rb') as pkl_file:
        channels = pickle.load(pkl_file)

    features_to_extract = {
        'time': ['skewness', 'peak_to_peak', 'mean'],
        'fourier': ['eeg_band_energies']
    }

    feature_extractor = FeatureExtractor(channels, fs, features_to_extract=features_to_extract)
    feature_extractor.compute_single_channel_features()
    print(feature_extractor.channels_features[0]['fourier'].extract_features())

if __name__ == '__main__':
    main()