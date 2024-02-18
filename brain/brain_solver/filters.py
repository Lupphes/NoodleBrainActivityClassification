import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter


class Filters:
    def __init__(self, order=5):
        self.order = order

    def butter_lowpass(self, cutoff, fs):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_highpass(self, cutoff, fs):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype="high", analog=False)
        return b, a

    def apply_filter(self, data, cutoff_low, cutoff_high, fs):
        b_low, a_low = self.butter_lowpass(cutoff_low, fs)
        b_high, a_high = self.butter_highpass(cutoff_high, fs)
        y_low = lfilter(b_low, a_low, data)
        y_high = lfilter(b_high, a_high, y_low)
        return y_high
