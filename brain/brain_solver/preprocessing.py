import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def apply_filter(data, cutoff_low, cutoff_high, fs, order=5):
    b_low, a_low = butter_lowpass(cutoff_low, fs, order=order)
    b_high, a_high = butter_highpass(cutoff_high, fs, order=order)
    y_low = lfilter(b_low, a_low, data)
    y_high = lfilter(b_high, a_high, y_low)
    return y_high


def preprocess_eeg_data(df, targets, consensus_col="expert_consensus"):
    """
    Preprocess EEG data by aggregating features within each eeg_id group.

    Parameters:
    - df: pandas DataFrame containing the EEG data.
    - targets: list of strings, names of the target columns to aggregate.
    - consensus_col: string, name of the column containing expert consensus.

    Returns:
    - A pandas DataFrame after preprocessing.
    """

    # Aggregate first spectrogram_id and min spectrogram_label_offset_seconds
    train = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
    train.columns = ["spec_id", "min_offset"]

    # Aggregate max spectrogram_label_offset_seconds
    train["max_offset"] = df.groupby("eeg_id")["spectrogram_label_offset_seconds"].max()

    # Aggregate first patient_id
    train["patient_id"] = df.groupby("eeg_id")["patient_id"].first()

    # Sum and normalize target columns
    tmp = df.groupby("eeg_id")[targets].agg("sum")
    for target in targets:
        train[target] = tmp[target].values
    y_data = train[targets].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[targets] = y_data

    # Aggregate first expert_consensus
    train["target"] = df.groupby("eeg_id")[consensus_col].first()

    # Reset index
    train = train.reset_index()

    # Print shape for confirmation
    print("Train non-overlap eeg_id shape:", train.shape)

    return train
