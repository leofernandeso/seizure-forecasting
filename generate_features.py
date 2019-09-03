import numpy as np
import pandas as pd

# main feature extractor 
from feature_extractor import *

# epilepsy ecosystem parser
from parsing import EpiEcoParser

def generate_features(paths_df, data_parser, output_fn):


    features_df = pd.DataFrame()
    count = 0
    for idx, row in paths_df.iterrows():

        # Getting separated windows and extracting features
        channels = data_parser.load_segment_from_path(row['abs_filepath'])
        windows = data_parser.extract_windows(channels)
        features = compute_windows_features(windows, data_parser.fs)

        # Appending final information
        features['class'] = row['class']
        features['patient'] = row['patient']

        features_df = pd.DataFrame(features, index=[0])
        features_df.to_hdf(output_fn, key='df', mode='a', format='table', append=True)
        count += 1
        print('Writing row {}/{}...\n'.format(count, len(paths_df)))




def main():

    data_parser = EpiEcoParser(**cfg.parser_args)
    fs = data_parser.fs
    segment_args = dict(
        patient_id=2,
        segment_id=140,
        _class=0
    )
    df = data_parser.process_dataset()
    generate_features(df, data_parser, cfg.h5_train_output_path)
    #windows = data_parser.extract_windows(channels)
    #features = compute_windows_features(windows, fs)
    # print(features)
    # print(len(features))
    
    

    
    

if __name__ == '__main__':
    main()