import numpy as np
from scipy.signal import butter, lfilter
from enum import Enum, auto


class FilterType(Enum):
    LOWPASS = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    BANDSTOP = auto()


class Filters:
    def __init__(self, order=5):
        self.order = order

    def butter_filter(self, cutoff, fs, btype="low"):
        nyq = 0.5 * fs
        if isinstance(cutoff, list) or isinstance(cutoff, tuple):
            normal_cutoff = [c / nyq for c in cutoff]
        else:
            normal_cutoff = cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype=btype, analog=False)
        return b, a

    def apply_filter(self, data, cutoff, fs, filter_type=FilterType.LOWPASS):
        if filter_type == FilterType.LOWPASS:
            b, a = self.butter_filter(cutoff, fs, "low")
        elif filter_type == FilterType.HIGHPASS:
            b, a = self.butter_filter(cutoff, fs, "high")
        elif filter_type == FilterType.BANDPASS:
            b, a = self.butter_filter(cutoff, fs, "bandpass")
        elif filter_type == FilterType.BANDSTOP:
            b, a = self.butter_filter(cutoff, fs, "bandstop")
        else:
            raise ValueError("Invalid filter type")
        return lfilter(b, a, data)

    def apply_filter_to_spectrogram(
        self, spectrogram, cutoff, fs, filter_type=FilterType.LOWPASS
    ):
        filtered_spectrogram = np.zeros_like(spectrogram)
        for i in range(spectrogram.shape[1]):
            time_slice = spectrogram[:, i]
            filtered_time_slice = self.apply_filter(time_slice, cutoff, fs, filter_type)
            filtered_spectrogram[:, i] = filtered_time_slice
        return filtered_spectrogram
