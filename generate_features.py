import numpy as np
import pandas as pd

# main feature extractor 
from feature_extractor import *

# epilepsy ecosystem parser
from parsing import EpiEcoParser

def generate_features(paths_df, data_parser, output_fn, dropout_path):

    count = 1
    with open(output_fn, 'a') as csv_file:
        for idx, row in paths_df.iterrows():

                print('Writing row {}/{} - File {}...\n'.format(count, len(paths_df), row['base_filepath']))

                # Getting separated windows and extracting features
                channels = data_parser.load_segment_from_path(row['abs_filepath'])
                windows = data_parser.extract_windows(channels)
                features = compute_windows_features(windows, data_parser.fs)
                #print(features)
                
                if not None in features.values():
                        # Appending final information
                        features['class'] = row['class']
                        features['patient'] = row['patient']

                        features_df = pd.DataFrame(features, index=[0])
                        if count == 1:
                                features_df.to_csv(csv_file, index=False)
                        else:
                                features_df.to_csv(csv_file, index=False, header=False, chunksize=300)
                else:
                        with open(dropout_path, 'a') as drop_file:
                                drop_file.write(row['abs_filepath']+'\n')
                count += 1





def main():

    data_parser = EpiEcoParser(**cfg.parser_args)
    df = data_parser.process_dataset()
    generate_features(df, data_parser, cfg.h5_train_output_path, cfg.train_drop_out_filepath)      
    

if __name__ == '__main__':
    main()