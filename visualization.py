from matplotlib import pyplot as plt
import numpy as np

def plot_eeg(channels_array, fs, scale='linear', savepath=None):
    count = 1
    nr_samples = channels_array.shape[1]
    t = np.arange(1, nr_samples+1) / fs
    for channel in channels_array:
        plt.subplot(len(channels_array), 1, count)
        plt.plot(t, channel)
        plt.xscale(scale)
        count += 1
    if savepath:
        plt.savefig(savepath)        
    else:
        plt.show()
    plt.clf()

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


if __name__ == '__main__':
    main()