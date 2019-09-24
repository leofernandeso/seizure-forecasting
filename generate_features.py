import numpy as np
import pandas as pd
import pickle
import time

# main feature extractor 
from feature_extractor import *

# epilepsy ecosystem parser
from parsing import EpiEcoParser

def generate_features(paths_df, data_parser, output_fn, dropout_path, join_windows=True):

    count = 1
    with open(output_fn, 'a') as csv_file:
        for idx, row in paths_df.iterrows():

                print('Writing file {}/{} - {}...\n'.format(count, len(paths_df), row['base_filepath']))

                # Getting separated windows and extracting features
        

                channels = data_parser.load_segment_from_path(row['abs_filepath'])
                windows = data_parser.extract_windows(channels)
                features = compute_windows_features(windows, data_parser.fs, join_windows=join_windows)
                
                if join_windows:        
                        if not None in features.values():
                                # Appending final information
                                features['class'] = row['class']
                                features['patient'] = row['patient']
                                features['segment_id'] = row['segment_id']

                                features_df = pd.DataFrame(features, index=[0])
                                if count == 1:
                                        features_df.to_csv(csv_file, index=False)
                                else:
                                        features_df.to_csv(csv_file, index=False, header=False, chunksize=300)
                        elif dropout_path:
                                with open(dropout_path, 'a') as drop_file:
                                        drop_file.write(row['abs_filepath']+'\n')
                        count += 1
                else:
                        features_df = pd.DataFrame(features)
                        features_df['class'] = row['class']
                        features_df['patient'] = row['patient']
                        features_df['segment_id'] = row['segment_id']
                        
                        if count == 1:
                                features_df.to_csv(csv_file, index=False)
                        else:
                                features_df.to_csv(csv_file, index=False, header=False, chunksize=300)
                        count += 1


def main():

    train_path = "C:\\Users\\Leonardo\\Documents\\Faculdade\\TCC\\processed_data\\subsets\\train.p"
    val_path = "C:\\Users\\Leonardo\\Documents\\Faculdade\\TCC\\processed_data\\subsets\\validation.p"
    

    data_parser = EpiEcoParser(**cfg.parser_args)


    # Generate folds
    # data_parser.generate_k_folds("D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem\\folds")



    # Run this line only to generate new subsets!
    # df_train, df_val = data_parser.process_dataset()


        #     with open(train_path, 'rb') as df_file:
        #             df_train = pickle.load(df_file)
        #     with open(val_path, 'rb') as df_file:
        #             df_val = pickle.load(df_file)

    
    #generate_features(df_train, data_parser, cfg.csv_train_output_path, cfg.train_drop_out_filepath, cfg.join_windows)      
    #generate_features(df_val, data_parser, cfg.csv_val_output_path, cfg.train_drop_out_filepath, cfg.join_windows)      


    

if __name__ == '__main__':
    main()