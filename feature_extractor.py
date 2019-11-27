import pickle
import pandas as pd
import numpy as np
import time

import visualization

# configuration file : change it for different feature extraction or different window sizes
import feature_extractor_config as cfg

# implemented features
from fourier import FourierFeatures
from time_analysis import TimeFeatures
from spatial_features import SpatialFeatures

# epilepsy ecosystem parser
import parsing

single_channel_feature_extractors_map = {
    'time': TimeFeatures,
    'fourier': FourierFeatures
}
spatial_feature_extractors_map = {
    'spatial': SpatialFeatures
}


class FeatureExtractor():
    def __init__(self, channels_array, fs, single_channel_features_to_extract=None, spatial_features_to_extract=None):
        self.channels_array = channels_array
        self.n_channels = len(channels_array)
        self.fs = fs
        #visualization.plot_eeg(self.channels_array, self.fs, scale='linear')
        #visualization.plot_channel(channels_array[8], self.fs)
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
                    {k: single_channel_feature_extractors_map[k](
                        channel, self.fs, val)}
                )
            channels_features.append(single_channel_features_dict)
        return channels_features

    def _compute_spatial_features(self):
        channels_signals = self.channels_array
        channels_spectra = [
            c['fourier'].power_spectral_density for c in self.channels_features]

        spatial_features_dict = {}
        for k, val in self.spatial_features_to_extract.items():
            spatial_features_dict.update(
                {k: spatial_feature_extractors_map[k](
                    channels_signals, channels_spectra, self.fs, val)}
            )
        return spatial_features_dict

    def extract_features(self):

        single_channel_features_dict = {}
        spatial_features_dict = {}

        # Extracting single-channel features
        for channel_idx, channel_feature in enumerate(self.channels_features):
            id_prefix = 'ch_' + str(channel_idx) + '_'
            for feature_type, feature_obj in channel_feature.items():
                extracted_features = feature_obj.extract_features()
                features_dict = {id_prefix+k: feature_val for k,
                                 feature_val in extracted_features.items()}
                single_channel_features_dict.update(features_dict)

        # Extracting multi-channel/spatial features
        for feature_type, feature_obj in self.spatial_features.items():
            extracted_features = feature_obj.extract_features()
            spatial_features_dict.update(extracted_features)

        return {**single_channel_features_dict, **spatial_features_dict}


def compute_windows_features(windows, fs, join_windows=True):

    if join_windows:
        features_to_return = {}
    else:
        features_to_return = []

    for w_count, w in enumerate(windows):
        feature_extractor = FeatureExtractor(
            w, fs,
            single_channel_features_to_extract=cfg.single_channel_features_to_extract,
            spatial_features_to_extract=cfg.spatial_features_to_extract)
        window_features = feature_extractor.extract_features()

        if join_windows:
            w_prefix = 'w_' + str(w_count) + '_'
            window_features_with_updated_keys = {
                w_prefix+k: feature_val for k, feature_val in window_features.items()}
            features_to_return.update(window_features_with_updated_keys)
        else:
            features_to_return.append(window_features)

    return features_to_return


def generate_features(paths_df, data_parser, output_fn, dropout_path, join_windows=True):

    count = 1
    with open(output_fn, 'a') as csv_file:
        for idx, row in paths_df.iterrows():

            print('Writing file {}/{} - {}...\n'.format(count,
                                                        len(paths_df), row['base_filepath']))

            # Getting separated windows and extracting features

            channels = data_parser.load_segment_from_path(row['abs_filepath'])
            windows = data_parser.extract_windows(channels)

            start = time.time()
            features = compute_windows_features(
                windows, data_parser.fs, join_windows=join_windows)
            end = time.time()
            print("Features computation time : {}\n\n".format(end-start))

            if join_windows:
                if not None in features.values():

                    # Appending final information
                    features['class'] = row['class']
                    features['patient_id'] = row['patient']
                    features['segment_id'] = row['segment_id']

                    features_df = pd.DataFrame(features, index=[0])
                    if count == 1:
                        features_df.to_csv(csv_file, index=False)
                    else:
                        features_df.to_csv(
                            csv_file, index=False, header=False, chunksize=300)
                else:
                    print("== Discarding segment due to dropouts. == ")

            else:
                # Checking if there are NaNs (this means dropouts!)
                for feature_dict in features:
                    if None in feature_dict.values():
                        contains_dropouts = True
                
                if not contains_dropouts:
                    features_df = pd.DataFrame(features)
                    features_df['class'] = row['class']
                    features_df['patient'] = row['patient']
                    features_df['segment_id'] = row['segment_id']

                    if count == 1:
                        features_df.to_csv(csv_file, index=False)
                    else:
                        features_df.to_csv(
                            csv_file, index=False, header=False, chunksize=300)
                else:
                    print("== Discarding segment due to dropouts. == ")
            count += 1


def main():
    data_parser = parsing.EpiEcoParser(**cfg.epieco_parser_args)
    fs = data_parser.fs
    segment_args = dict(
        patient_id=3,
        segment_id=298,
        _class=0,
    )
    channels = data_parser.get_full_train_segment(**segment_args)
    windows = data_parser.extract_windows(channels)
    features = compute_windows_features(windows, fs, join_windows=False)
    # print(features[0])
    # print(len(features[0]))


if __name__ == '__main__':
    main()
