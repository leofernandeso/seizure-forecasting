import numpy as np
import h5py
import os

class EpiEcoParser():
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.fs = 400 
    def get_train_segment(self, patient_id, segment_id, _class=1):
        patient_folder = "Pat{}Train".format(patient_id)
        segment_path = "Pat" + str(patient_id) + "Train_" + str(segment_id) + "_" + str(_class) + ".hdf5"
        segment_path = "Pat{}Train_{}_{}.hdf5".format(patient_id, segment_id, _class)
        file_path = os.path.join(self.base_folder, patient_folder, segment_path)
        eeg_file = h5py.File(file_path, 'r')
        eeg_data = np.array(eeg_file['data'].get('table'))
        signal_array = [row[3] for row in eeg_data]
        return np.array(np.transpose(signal_array))
    
