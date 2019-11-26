import sys
import re
import numpy as np
import pandas as pd
import h5py
import pickle
import os
import shutil
import feature_extractor_config as cfg
import feature_extractor

from sklearn.model_selection import train_test_split, StratifiedKFold
# from chb_label_wrapper import ChbLabelWrapper

df_subset_path = "C:\\Users\\Leonardo\\Documents\\Faculdade\\TCC\\processed_data\\subsets"

def get_segment_windows(fs, segment_array, window_ranges):
    transposed_array = np.transpose(segment_array)
    for w_range in window_ranges:
        lower_limit, upper_limit = w_range
        lower_id = lower_limit*fs
        upper_id = upper_limit*fs
        cropped_array = transposed_array[lower_id:upper_id]
        cropped_array = np.transpose(cropped_array)
        yield cropped_array

def get_edf_data(path):
    edf_raw_data = mne.io.read_raw_edf(path, verbose=False)
    eeg_data = edf_raw_data.get_data(picks=cfg.chb_channels_list)
    return eeg_data

def crop_preictal_segment(fs, data, seizure_interval, preictal_window, horizon=5*60):
    preictal_window = preictal_window * 60  # converting to seconds
    for interval in seizure_interval:
        seizure_begin = interval[0]    # in seconds
        segment_end_time = seizure_begin - horizon
        segment_start_time = segment_end_time - preictal_window
        N1_preictal = segment_start_time * fs
        if N1_preictal < 0:
            N1_preictal = 0

        N2_preictal = segment_end_time * fs     
        cropped_data = data[:, N1_preictal:N2_preictal]
        yield cropped_data
    
def divide_interictal_files(fs, data, interictal_window):
    duration = int(data.shape[1] * (1/fs))
    windows = cfg.non_overlapping_windows(interictal_window*60, duration)
    for w in windows:
        N_w1 = w[0] * fs
        N_w2 = w[1] * fs
        yield data[:, N_w1:N_w2]

class EpiEcoParser():
    def __init__(self, base_folder, windows_range, file_type, study='all'):
        self.base_folder = base_folder
        self.folds_folder = self.base_folder + '/folds'
        self.train_folder = self.base_folder + '/Train'
        self.test_folder = self.base_folder + '/Test'
        self.file_type = file_type
        self.fs = 400 
        self.windows_range = windows_range
        self.study = study

    def _get_all_studies_data(self, subset='Train'):

        # Getting all files inside patient folders
        files_list = []
        base_path_list = []
        for _root, _, files in os.walk(os.path.join(self.base_folder, subset)):
            for _file in files:
                if _file.endswith(self.file_type):
                    files_list.append(os.path.join(_root, _file))
                    base_path_list.append(_file)

        files_list = np.array(files_list)
        base_path_list = np.array(base_path_list)

        files_list = np.expand_dims(files_list, axis=1)
        base_path_list = np.expand_dims(base_path_list, axis=1)

        # Dataframe manipulation
        df_arr = np.concatenate((files_list, base_path_list), axis=1)
        df = pd.DataFrame(np.concatenate((files_list, base_path_list), axis=1), columns=['abs_filepath', 'base_filepath'])
        splitted = df['base_filepath'].str.split('_|\.', expand=True)
        df['patient'] = splitted[0].str.extract('(\d)')
        df['segment_id'] = splitted[1]
        df['class'] = splitted[2]
        return df

    def _get_segment_path(self, subset, patient_id, segment_id, _class, file_type='csv'):
        patient_folder = subset + "/Pat{}Train".format(patient_id)
        segment_path = "Pat" + str(patient_id) + "Train_" + str(segment_id) + "_" + str(_class) + ".{}".format(file_type)
        segment_path = "Pat{}Train_{}_{}.{}".format(patient_id, segment_id, _class, file_type)
        file_path = os.path.join(self.base_folder, patient_folder, segment_path)
        return file_path

    def process_dataset(self):
        df_train = self._get_all_studies_data()
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['class'])
        with open(df_subset_path+'\\train.p', 'wb') as train_df_file:
            pickle.dump(df_train, train_df_file)     
        with open(df_subset_path+'\\validation.p', 'wb') as val_df_file:            
            pickle.dump(df_val, val_df_file)
        return df_train, df_val        
    
    def generate_k_folds_features(self):
        for fold_nr, fold_dir in enumerate(  sorted(os.listdir(self.folds_folder))  ):



            train_fp = 'fold{}Train.csv'.format(fold_nr+1)
            test_fp = 'fold{}Test.csv'.format(fold_nr+1)

            abs_fold_dir = os.path.join(self.folds_folder, fold_dir)
            train_fold_file = os.path.join(abs_fold_dir, train_fp)
            test_fold_file = os.path.join(abs_fold_dir,  test_fp)

            train_fold_df = pd.read_csv(train_fold_file)
            test_fold_df = pd.read_csv(test_fold_file)

            if train_fp in cfg.epieco_folds_to_process:
                print("=== Generating train fold - {} === \n".format(fold_nr+1))
                feature_extractor.generate_features(train_fold_df, self, abs_fold_dir+'/fold{}Train_features.csv'.format(fold_nr+1),
                                        None, join_windows=False)
                                        
            if test_fp in cfg.epieco_folds_to_process:
                print("=== Generating test fold - {} === \n".format(fold_nr+1))                                     
                feature_extractor.generate_features(test_fold_df, self, abs_fold_dir+'/fold{}Eval_features.csv'.format(fold_nr+1),
                                        None, join_windows=False)

    def generate_k_folds(self, k=5):
        ''' 
            Generates folds with separate csv files, where each csv file contains the segment paths
            and other relevant data in order to perform the evaluation
        '''

        if not os.path.exists(self.folds_folder):
            os.mkdir(self.folds_folder)

        data_df = self._get_all_studies_data()
        data_df = data_df.sample(frac=1).reset_index(drop=True)     # shuffle
        X = data_df.drop(columns=['class'])
        y = data_df['class']

        skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
        counter = 1
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            fold_df_train = pd.DataFrame(X_train, columns=X.columns)
            fold_df_train['class'] = y_train
        
            fold_df_test = pd.DataFrame(X_test, columns=X.columns)
            fold_df_test['class'] = y_test

            folder_path = os.path.join(self.folds_folder, "fold{}".format(counter))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            fold_df_train.to_csv(os.path.join(folder_path, "fold{}Train.csv".format(counter)))
            fold_df_test.to_csv(os.path.join(folder_path, "fold{}Test.csv".format(counter)))
            
            counter += 1

        if self.study == 'all':
            return self._get_all_studies_data()
        else:
            # call self._process_study()
            return None

    def load_segment_from_path(self, path):
        print(path)
        if self.file_type == 'csv':
            eeg_data = pd.read_csv(path)
            signal_array = eeg_data.iloc[:, 4:]
        elif self.file_type == 'hdf5':
            eeg_file = h5py.File(path, 'r')
            eeg_data = np.array(eeg_file['data'].get('table'))
            signal_array = [row[3] for row in eeg_data]
        return np.array(np.transpose(signal_array))

    def get_full_train_segment(self, patient_id, segment_id, _class=1):
        file_path = self._get_segment_path('Train', patient_id, segment_id, _class)
        return self.load_segment_from_path(file_path)

    def extract_windows(self, segment_array):
        return get_segment_windows(self.fs, segment_array, self.windows_range)






# class CHBParser():
#     def __init__(self, base_folder, preictal_window, interictal_window):
#         self.base_folder = base_folder
#         self.preictal_window = preictal_window
#         self.interictal_window = interictal_window  

#     """ Function used to generate patient segments - using pickle files. """
#     def fetch_all_patients_data(self):
#         for patient_folder in os.listdir(self.base_folder):
#             patient_id = patient_folder.split('chb')[1]
#             print("Fetching patient {} data...".format(patient_id))
#             self.generate_patient_segments(patient_id)

#     """ Generates dataframe with patient specific segment paths """
#     def generate_paths_df(self):

#         preictal_path_template = "Pat{}_preictal"
#         interictal_path_template = "Pat{}_interictal"

#         # Processing folders that have dependencies
#         dest_folder = os.path.join(self.base_folder, "paths")
#         if os.path.exists(dest_folder):
#             shutil.rmtree(dest_folder, ignore_errors=True)

#         for patient_folder in os.listdir(self.base_folder):

#             preictal_data = {}
#             interictal_data = {}

#             # Folder and subfolder processing
#             patient_id = patient_folder.split('chb')[1]
#             abs_pat_folder_path = os.path.join(self.base_folder, patient_folder)
#             pat_preictal_folderpath = os.path.join(abs_pat_folder_path, preictal_path_template.format(patient_id))
#             pat_interictal_folderpath = os.path.join(abs_pat_folder_path, interictal_path_template.format(patient_id))
#             preictal_files = os.listdir(pat_preictal_folderpath)
#             interictal_files = os.listdir(pat_interictal_folderpath)

#             # Building a dict and a pandas dataframe from dict
#             preictal_data.update(
#                 {
#                     'path': [os.path.join(pat_preictal_folderpath, seg_path) for seg_path in preictal_files],
#                     'class': [1] * len(preictal_files),
#                     'segment_id': list(range(1, len(preictal_files) + 1 ))
#                 }
#             )
#             interictal_data.update(
#                 {
#                     'path': [os.path.join(pat_interictal_folderpath, seg_path) for seg_path in interictal_files],
#                     'class': [0] * len(interictal_files),
#                     'segment_id': list(range(1, len(interictal_files) + 1 ))
#                 }
#             )

#             print("\n\n ** Patient {} - Writing paths CSV ** \n\n".format(patient_id))
#             patient_df = pd.concat([
#                     pd.DataFrame(preictal_data, columns=['path', 'class', 'segment_id']),
#                     pd.DataFrame(interictal_data, columns=['path', 'class', 'segment_id'])
#                 ], axis=0
#             )

#             # Saving csv with paths
#             if not os.path.exists(dest_folder):
#                 os.mkdir(dest_folder)
#             patient_df.to_csv(
#                 os.path.join(dest_folder, "Pat_{}_paths.csv".format(patient_id)), index=False
#             )
            
#     """ Reads all segment raw files for a given patient, dividing the segments as preictal or interictal.
#         Saves the processed data as pickle files """
#     def generate_patient_segments(self, patient_id):
#         patient_folder = os.path.join(self.base_folder, 'chb{}'.format(patient_id))
#         wrapper = ChbLabelWrapper(os.path.join(patient_folder, 'chb{}-summary.txt'.format(patient_id)))
#         seizure_list = wrapper.get_seizure_list()
#         edf_files = [f for f in os.listdir(patient_folder) if f.endswith('.edf')]


#         preictal_dest_folder = os.path.join(patient_folder, "Pat{}_preictal".format(patient_id))
#         interictal_dest_folder = os.path.join(patient_folder, "Pat{}_interictal".format(patient_id))

#         if not os.path.exists(preictal_dest_folder):
#             os.mkdir(preictal_dest_folder)
#         if not os.path.exists(interictal_dest_folder):
#             os.mkdir(interictal_dest_folder)

#         for index, file in enumerate(edf_files):
#             if 'chb' + patient_id not in cfg.chb_ignore_patients:
#                 filepath = os.path.join(patient_folder, file)
#                 eeg_data = get_edf_data(filepath)
#                 if seizure_list[index]:
#                     cropped_preictal_arrays = crop_preictal_segment(self.fs, eeg_data, seizure_list[index], self.preictal_window)
#                     for seg in cropped_preictal_arrays:
#                         print(file)
#                         dest_file_name = "preictal_Pat{}_{}.p".format(patient_id, len(os.listdir(preictal_dest_folder))+1)
#                         dest_file_path = os.path.join(preictal_dest_folder, dest_file_name)
#                         with open(dest_file_path, 'wb') as dest_file:
#                             pickle.dump(seg, dest_file)
#                 else:
#                     cropped_interictal_arrays = divide_interictal_files(self.fs, eeg_data, self.interictal_window)
#                     print(file)
#                     for seg in cropped_interictal_arrays:
#                         dest_file_name = "interictal_Pat{}_{}.p".format(patient_id, len(os.listdir(interictal_dest_folder))+1)
#                         dest_file_path = os.path.join(interictal_dest_folder, dest_file_name)
#                         with open(dest_file_path, 'wb') as dest_file:
#                             pickle.dump(seg, dest_file)



def main():
    parser = EpiEcoParser(**cfg.epieco_parser_args)
    parser.generate_k_folds()
    parser.generate_k_folds_features()

if __name__ == '__main__':
    main()
