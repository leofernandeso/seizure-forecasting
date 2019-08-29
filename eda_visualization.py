from matplotlib import pyplot as plt
import numpy as np
from parsing import EpiEcoParser

def plot_eeg_comparison(channels_array1, channels_array2, fs, scale='log'):
    count = 1
    nr_samples = channels_array1.shape[1]
    t = np.arange(1, nr_samples+1) / fs
    f1 = plt.figure(1)
    for channel in channels_array1:
        plt.subplot(len(channels_array1), 1, count)
        plt.plot(t, channel)
        plt.xscale(scale)
        count += 1
    f1.show()
    count = 1
    f2 = plt.figure(2)
    for channel in channels_array2:
        plt.subplot(len(channels_array2), 1, count)
        plt.plot(t, channel)
        plt.xscale(scale)
        count += 1
    f2.show()
    input()

# def main():
#     data_parser = EpiEcoParser("D:\\Faculdade\\TCC\\dados\\epilepsy_ecosystem")
#     fs = data_parser.fs
#     parser_args_interictal = dict(
#         patient_id=2,
#         segment_id=137,
#         _class=0
#     )
#     parser_args_preictal = dict(
#         patient_id=2,
#         segment_id=137,
#         _class=1
#     )
#     channels_array_interictal = data_parser.get_train_segment(**parser_args_interictal)
#     channels_array_preictal = data_parser.get_train_segment(**parser_args_preictal)
#     plot_eeg_comparison(channels_array_interictal, channels_array_preictal, fs, scale='linear')
    


# if __name__ == '__main__':
#     main()