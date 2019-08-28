import numpy as np
from scipy import signal as sci_signal
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
import math
import pickle
from matplotlib import pyplot as plt


"""
    Possible features to be implemented : 
    ===   Spectral edge frequency and power   ===
        Most of the spectral power is comprised in the 0.5-40Hz band. We define spectral
        edge frequency as the frequency below which 50% of the total power of the signal
        is located. The spectral edge power is the value of the power existing below the
        spectral edge frequency.
    ===   Energy of wavelet coefficients  ===
        Along with DFT, the Discrete Wavelet Transform is also applied using a 7-level decomposition and the
        Daubechies 4 (db4) as the mother wavelet, to extract the detail
        (64–128 Hz, 32–64 Hz, 16–32 Hz, 8–16 Hz, 4–8 Hz, 2–4 Hz, 1–2 Hz) and
        approximation coefficients (<1 Hz). The 7-level decomposition is
        selected based on the 256 Hz sampling frequency used when assembling
        the CHB-MIT database, as it is the minimum depth required to cover the
        fundamental frequency bands and allow for the separation of bellow the
        1 Hz that is most predominantly occupied by artifacts.
"""

EEG_BANDS = {   
            'delta': (0, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma1': (30, 80),
            'gamma2': (80, 150)
            }
SPECTRAL_EDGE_FREQUENCIES_LIST = [80, 90, 95]


class FourierFeatures():
    def __init__(self, signal, fs, features_list=None):
        self.features_list = features_list
        self.signal = signal
        self.fs = fs
        self.ts = 1/fs
        self.Np = len(signal)
        self.samples_in_segment = 2 * (1/0.5) * self.fs
        self._compute_base_variables()

    def _compute_base_variables(self):
        self.f_axis, self.power_spectral_density = self._power_spectral_density()
        self._freq_res = self.f_axis[1] - self.f_axis[0] # window frequency resolution 
        self.total_power = trapz(self.power_spectral_density, dx=self._freq_res)

    def _power_spectral_density(self):
        freqs, psd = sci_signal.welch(self.signal, self.fs, nperseg=self.samples_in_segment)
        return freqs, psd

    def spectral_edge_frequencies(self):
        cum_power = cumtrapz(self.power_spectral_density, dx=self._freq_res)
        edge_frequencies_dict = {}
        for edge_factor in SPECTRAL_EDGE_FREQUENCIES_LIST:
            superior_freqs = np.array(np.where(cum_power > (edge_factor/100) * self.total_power)) * self._freq_res
            edge_frequency = superior_freqs[0][0] # first superior frequency
            edge_frequencies_dict.update(
                {'spectral_edge_freq_'+str(edge_factor): edge_frequency}
            )
        return edge_frequencies_dict

    def eeg_band_powers(self):
        band_powers = {}
        band_powers['total_power'] = self.total_power
        for band in EEG_BANDS:
            lower_freq, upper_freq = EEG_BANDS[band]
            band_power = self.power_spectral_density[(self.f_axis >= lower_freq) & (self.f_axis < upper_freq)]
            relative_band_energy = trapz(band_power, dx=self._freq_res) / self.total_power
            band_powers[band] = relative_band_energy
        return band_powers

    def extract_features(self):
        features_dict = {}
        for feature_name in self.features_list:
            try:
                method_to_call = getattr(self, feature_name)
                output_params = method_to_call()
                features_dict.update(output_params)
            except AttributeError:
                print(f"Feature **{feature_name}** calculation method not implemented in FourierFeatures!")
        return features_dict

# fs = 400
# ts = 1/fs
# with open('eeg_segment.p', 'rb') as pkl_file:
#     signal = pickle.load(pkl_file)

# t = np.arange(0, len(signal)) * ts
# ff = FourierFeatures(signal, fs, features_list=['eeg_band_powers', 'spectral_edge_frequencies'])
# computed_features = ff.extract_features()
# print(computed_features)
# plt.plot(ff.f_axis, ff.power_spectral_density)
# plt.show()