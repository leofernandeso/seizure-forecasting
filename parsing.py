import sys
import re
import numpy as np
import pandas as pd
import h5py
import os

class EpiEcoParser():
    def __init__(self, base_folder, windows_range, study='all'):
        self.base_folder = base_folder
        self.train_folder = self.base_folder + '\\Train'
        self.test_folder = self.base_folder + '\\Test'
        self.fs = 400 
        self.windows_range = windows_range
        self.study = study

    def _get_all_studies_data(self, subset='Train'):

        # Getting all files inside patient folders
        files_list = []
        base_path_list = []
        for _root, _, files in os.walk(os.path.join(self.base_folder, subset)):
            for file in files:
                if file.endswith('.hdf5'):
                    files_list.append(os.path.join(_root, file))
                    base_path_list.append(file)

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

    def _get_segment_path(self, subset, patient_id, segment_id, _class):
        patient_folder = subset + "\\Pat{}Train".format(patient_id)
        segment_path = "Pat" + str(patient_id) + "Train_" + str(segment_id) + "_" + str(_class) + ".hdf5"
        segment_path = "Pat{}Train_{}_{}.hdf5".format(patient_id, segment_id, _class)
        file_path = os.path.join(self.base_folder, patient_folder, segment_path)
        return file_path

    def process_dataset(self):
        if self.study == 'all':
            return self._get_all_studies_data()
        else:
            # call self._process_study()
            return None
    
    def load_segment_from_path(self, path):
        eeg_file = h5py.File(path, 'r')
        eeg_data = np.array(eeg_file['data'].get('table'))
        signal_array = [row[3] for row in eeg_data]
        return np.array(np.transpose(signal_array))

    def get_full_train_segment(self, patient_id, segment_id, _class=1):
        file_path = self._get_segment_path('Train', patient_id, segment_id, _class)
        return self.load_segment_from_path(file_path)

    def extract_windows(self, segment_array):
        return get_segment_windows(self.fs, segment_array, self.windows_range)

def get_segment_windows(fs, segment_array, window_ranges):
    transposed_array = np.transpose(segment_array)
    for w_range in window_ranges:
        lower_limit, upper_limit = w_range
        lower_id = lower_limit*fs
        upper_id = upper_limit*fs
        cropped_array = transposed_array[lower_id:upper_id]
        cropped_array = np.transpose(cropped_array)
        yield cropped_array

        

