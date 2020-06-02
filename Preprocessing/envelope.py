import numpy as np
from scipy.signal import butter, lfilter, hilbert

from scipy.signal import butter, lfilter, hilbert


class envl_freq:
    """
    Class to calculate the envelope and spectum of a waveform
    Args:
        sampling_frequency (int): Sampling frequency of the waveform
        lowcut (int): Lower band pass filter
        highcut (int): Higher band pass filter
    Attributes:
        fs (int): Sampling frequency of the waveform
        lowcut (int): Lower band pass filter
        highcut (int): Higher band pass filter
    """
    def __init__(self, sampling_frequency, lowcut, highcut):
        self.fs = sampling_frequency
        self.lowcut = lowcut
        self.highcut = highcut

    def centering(self, data):
        """
        Centering the waveform data
        Args:
            data (numpy array): Original waveform
        Returns:
            (numpy array): Centered data
        """
        return data - np.mean(data)

    def bandpass_filter(self, wave, order=5):
        """
        Computes the band pass filter
        Args:
            wave (numpy array): Original waveform
            order (int): Order of the Butterworth filter
        Returns:
            y (numpy array): Filtered waveform
        """
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, wave)
        return y

    def envelope_hilbert(self, signal):
        """
        Calculate waveform envelope using Hilber filter
        Attributes:
            signal (numpy array): Waveform
        Returns:
            amplitude envelope (numpy array): Envelope of the signal
        """
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def calculate_envelope_hilbert(self, wave):
        """
        Calculates the envelope using Hilbert method
        Attributes:
            wave (numpy array): Original waveform
        Returns:
            envelope (numpy array): Envelope of waveform
        """
        filtered = self.bandpass_filter(wave)  # Above in the frequence we are interested in the data from 2500 - 5000
        envelope = self.envelope_hilbert(filtered)
        return envelope

    def acc_spectrum(self, wave, w_length, rate, cutoff_freq=0):
        """
        Compute acceleration spectrum [g] from measured wave [g]
        Args:
            wave (numpy.array): measured wave [g] for each temporal sample
            w_length (int): Number of samples of FFT
            rate (int): sampling rate [samples/seg]
            cutoff_freq (float, optional): Cutoff frequency for the ideal highpass filter
        Returns:
            acceleration: array with the acceleration [g] for each frecuency in freqs
            frecs: array with the evaluated frecuencies [Hz]
            amplitude_correction: Amplitude correction value
        """
        n_length = len(wave)
        if w_length < n_length:
            ana_wave = wave[0:w_length]
            n_length = w_length
        else:
            ana_wave = wave
        window = np.hanning(n_length)  # np.ones(n_length) #np.hanning(n_length)
        wave_fft = np.fft.rfft(window * ana_wave, n=w_length)
        abs_wave_fft = np.abs(wave_fft)
        freqs = np.fft.rfftfreq(w_length, 1 / rate)
        correccion_amplitud = 1 / sum(window)
        scaled_fft = 2 * correccion_amplitud * abs_wave_fft
        scaled_fft[freqs < cutoff_freq] = 0
        return scaled_fft, freqs, correccion_amplitud

    def envel(self, wave):
        """
        Calculates the spectrum of the envelope of the waveform.
        Attributes:
            wave (numpy array): Original waveform
        Returns:
            fft_env (numpy array): Amplitude of the envelope spectrum
            freq (numpy array): Frequencies of the envelope spectrum
        """
        x = self.centering(wave)
        envelope_hilbertvar = self.calculate_envelope_hilbert(x)
        envelope_hilbertvar_centered = self.centering(envelope_hilbertvar)
        fft_env, freq, _ = self.acc_spectrum(envelope_hilbertvar_centered, len(envelope_hilbertvar_centered),
                                             self.fs, cutoff_freq=10)
        return fft_env, freq


