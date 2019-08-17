import numpy as np
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


class FourierFeatures():
    def __init__(self, signal, fs, features_list=None):
        self.features_list = features_list
        self.signal = signal
        self.fs = fs
        self.ts = 1/fs
        self.Np = len(signal)
        self._compute_base_variables()

    def _compute_base_variables(self):
        self.f_axis = self.fs * ((np.arange(0, self.Np) - self.Np/2) / self.Np)
        self.f_axis = self.f_axis[self.f_axis >= 0] # ignoring negative frequencies
        self.amplitude_spectrum = self._amplitude_spectrum()
        self.power_spectrum = self._power_spectrum()

    def _amplitude_spectrum(self):
        """ Computes the amplitude spectrum ignoring amplitude values associated with negative frequencies """
        amplitude_spectrum = np.fft.fft(self.signal)
        amplitude_spectrum = np.abs( np.fft.fftshift(amplitude_spectrum) ) / self.fs
        spectrum_middle = int( np.ceil(self.Np/2) ) 
        amplitude_spectrum = amplitude_spectrum[spectrum_middle:]  
        return amplitude_spectrum 

    def _power_spectrum(self):
        power_spectrum = self.amplitude_spectrum ** 2
        return power_spectrum

    def eeg_band_energies(self):
    
        # Approximate energy calculation method (integral of power * dt) : 
        # energy = ts * sum(power_spectrum)
        # Since we're interested in calculating the percentage of the band energies, the term *self.ts* is irrelevant
        # (because it cancels out in the percentage formula)

        band_energies = {}
        total_energy = self.ts * np.sum(self.power_spectrum)
        band_energies['total_energy'] = total_energy
        for band in EEG_BANDS:
            lower_freq, upper_freq = EEG_BANDS[band]
            band_power_spectrum = self.power_spectrum[(self.f_axis >= lower_freq) & (self.f_axis < upper_freq)]
            relative_band_energy = self.ts * np.sum(band_power_spectrum) / total_energy
            band_energies[band] = relative_band_energy
        return band_energies

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

# test eeg segment
# fs = 400
# ts = 1/fs
# with open('eeg_segment.p', 'rb') as pkl_file:
#     signal = pickle.load(pkl_file)

# t = np.arange(0, len(signal)) * ts
# ff = FourierFeatures(signal, fs, features_list=['eeg_band_energies'])
# computed_features = ff.extract_features()
# print(computed_features)