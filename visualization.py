from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

def plot_eeg(channels_array, fs=1, scale='linear', savepath=None):
    count = 1
    nr_samples = channels_array.shape[1]
    t = np.arange(1, nr_samples+1) / fs
    for channel in channels_array:
        plt.subplot(len(channels_array), 1, count)
        plt.plot(t, channel)
        plt.xscale(scale)
        plt.ylim([-500, 500])
        count += 1
    if savepath:
        plt.savefig(savepath)        
    else:
        plt.show()
    plt.clf()

def plot_power_spectrum(freqs, psd):
    plt.plot(freqs, psd, color='gray', linewidth=0.8)
    plt.fill_between(freqs, psd, where=freqs<=80, color='steelblue')
    plt.fill_between(freqs, psd, where=freqs<=30, color='cadetblue')
    plt.fill_between(freqs, psd, where=freqs<=12, color='cornflowerblue')
    plt.fill_between(freqs, psd, where=freqs<=8, color='royalblue')
    plt.fill_between(freqs, psd, where=freqs<=4, color='lightskyblue')
    plt.xticks(np.arange(min(freqs), max(freqs), 5))
    plt.xlim([0, 60])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V ^ 2 / Hz)")
    plt.show()

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

def plot_channel(channel, fs=1):
    count = 1
    nr_samples = len(channel)
    t = np.arange(1, nr_samples+1) / fs
    plt.plot(t, channel)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.ylim([-500, 500])
    plt.show()
    plt.clf()

def plot_graph(G):
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='r', edgelist=edges, edge_color=weights,
                 width=1.0, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()