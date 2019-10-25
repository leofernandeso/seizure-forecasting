import os
import shutil
import pickle
import time
import pandas as pd
import numpy as np
import feature_extractor_config as cfg
from feature_extractor import compute_windows_features
from parsing import get_segment_windows
from sklearn.model_selection import train_test_split, StratifiedKFold

class CHBDataGenerator():
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.paths_folder = os.path.join(self.base_folder, 'paths')
        self.patients_folders = [f for f in os.listdir(self.paths_folder) if os.path.isdir( os.path.join( self.paths_folder, f ) )]
        self.fs = 256

    def generate_all_patients_train_test(self, test_size=0.25):
        path_files = [f for f in os.listdir(self.paths_folder) if os.path.isfile( os.path.join( self.paths_folder, f ) )]
        for pat_path_file in path_files:
            patient_id = pat_path_file[4:6]
            self.generate_patient_train_test(patient_id, test_size)

    def load_segment_from_path(self, path):
        segment_file = open(path, 'rb')
        signal_array = pickle.load(segment_file)
        return signal_array

    def generate_patient_train_test(self, patient_id, test_size):
        
        dest_folder = os.path.join(
            self.paths_folder, 'Data_Pat_{}'.format(patient_id)
        )

        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder, ignore_errors=True)
        os.mkdir(dest_folder)
        
        patient_df = pd.read_csv(os.path.join(self.paths_folder, 'Pat_{}_paths.csv'.format(patient_id)))
        df_train, df_test = train_test_split(patient_df, test_size=test_size, shuffle=True, stratify=patient_df['class'])

        df_train.to_csv( os.path.join( dest_folder, 'Pat_{}_Train.csv'.format(patient_id) ), index=False)
        df_test.to_csv( os.path.join( dest_folder, 'Pat_{}_Test.csv'.format(patient_id) ), index=False)
        
    def generate_features_from_csv(self, csv_fn, dest_csv, windows_interval=10):
        
        
        with open(dest_csv, 'a') as csv_file:
            paths_df = pd.read_csv(csv_fn)
            count = 1
            for idx, row in paths_df.iterrows():

                # Verbose
                print('Writing file {}/{} - {}...\n'.format(count, len(paths_df), row['path']))

                channels = self.load_segment_from_path(row['path'])
                duration = (1/self.fs) * channels.shape[1]  # ts * number of signal samples
                windows_ranges = cfg.non_overlapping_windows(windows_interval, int(duration))
                signal_windows = get_segment_windows(self.fs, channels, windows_ranges)

                start = time.time()
                features_list = compute_windows_features(signal_windows, self.fs, join_windows=cfg.join_windows)
                end = time.time()
                print('Time in feature computation {}\n'.format(end-start))

                for features_dict in features_list:                    
                    features_dict['patient_id'] = row['path'].split('_')[-2].split('Pat')[-1]
                    features_dict['class'] = row['class']
                    features_dict['segment_id'] = row['segment_id']

                features_df = pd.DataFrame(features_list)
                if count == 1:
                        features_df.to_csv(csv_file, index=False)
                else:
                        features_df.to_csv(csv_file, index=False, header=False, chunksize=300)
                count += 1

    def generate_kFolds_features(self, window_interval=10):


        for pat_folder in self.patients_folders:
            abs_pat_folder = os.path.join(self.paths_folder, pat_folder)
            folds_folder = os.path.join(abs_pat_folder, 'folds')
            for fold_file in os.listdir(folds_folder):
                if fold_file not in cfg.chb_processed_folds and not fold_file.endswith('_features.csv'):
                    print("==== Processing file {} ====\n".format(fold_file))
                    abs_fold_fp = os.path.join(folds_folder, fold_file)
                    abs_features_dest = os.path.join(folds_folder, fold_file.split('.')[0] + '_features.csv')
                    self.generate_features_from_csv(abs_fold_fp, abs_features_dest)
                
    def generate_k_folds_paths(self, k=3):
        for pat_folder in self.patients_folders:

            # Manipulating paths
            patient_id = pat_folder.split('_')[2]
            train_fp = os.path.join(pat_folder, 'Pat_{}_Train.csv'.format(patient_id) )
            train_fp = os.path.join(self.paths_folder, train_fp)
            folds_folder = os.path.join( self.paths_folder, pat_folder, 'folds' )

            if os.path.exists(folds_folder):
                shutil.rmtree(folds_folder, ignore_errors=True)
            else:
                os.mkdir(folds_folder)

            # Reading df
            pat_train_df = pd.read_csv(train_fp)
            X = pat_train_df['path'] # X -> not really variables. just paths
            y = pat_train_df['class'] 
            segment_ids = pat_train_df['segment_id']

            # Building Folds
            skf = StratifiedKFold(n_splits=k, random_state=42)
            counter = 1
            for train_index, eval_index in skf.split(X, y):
                X_train, X_eval = X.iloc[train_index], X.iloc[eval_index]
                y_train, y_eval = y.iloc[train_index], y.iloc[eval_index]
                segment_ids_train = segment_ids.iloc[train_index]                
                segment_ids_eval = segment_ids.iloc[eval_index]                

                fold_df_train = pd.DataFrame()
                fold_df_train['path'] = X_train
                fold_df_train['class'] = y_train
                fold_df_train['segment_id'] = segment_ids_train
                
                fold_df_eval = pd.DataFrame()
                fold_df_eval['path'] = X_eval
                fold_df_eval['class'] = y_eval
                fold_df_eval['segment_id'] = segment_ids_eval
                
                fold_df_train.to_csv( os.path.join(
                    folds_folder, "Pat_{}_Fold_{}_Train.csv".format(patient_id, counter)
                    ) 
                )
                fold_df_eval.to_csv( os.path.join(
                    folds_folder, "Pat_{}_Fold_{}_Eval.csv".format(patient_id, counter)
                    )
                )
                counter += 1
            

def main():
    chb_data_gen = CHBDataGenerator(cfg.chb_parser_args['base_folder'])
    # chb_data_gen.generate_all_patients_train_test()
    # chb_data_gen.generate_k_folds_paths(k=5)
    chb_data_gen.generate_kFolds_features()

if __name__ == '__main__':
    main()
        
