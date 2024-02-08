import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Helpers:
    @staticmethod
    def check_file_existence(file_path):
        """
        Checks if a file exists at the specified path.

        Parameters:
        - file_path (str): The path to the file.

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        return os.path.exists(file_path) and os.path.isfile(file_path)

    @staticmethod
    def load_csv(file_path, **kwargs):
        """
        Loads a CSV file into a pandas DataFrame if the file exists.

        Parameters:
        - file_path (str): The path to the CSV file.
        - kwargs: Additional keyword arguments to pass to pd.read_csv().

        Returns:
        - DataFrame if file exists and is loaded successfully, None otherwise.
        """
        if Helpers.check_file_existence(file_path):
            try:
                df = pd.read_csv(file_path, **kwargs)
                return df
            except Exception as e:
                print(f"Failed to load file due to: {e}")
                return None
        else:
            print("The file does not exist. Check the path and try again.")
            return None

    @staticmethod
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
        train["max_offset"] = df.groupby("eeg_id")[
            "spectrogram_label_offset_seconds"
        ].max()

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

    @staticmethod
    def eeg_from_parquet(parquet_path, FEATS, display=False):

        # EXTRACT MIDDLE 50 SECONDS
        eeg = pd.read_parquet(parquet_path, columns=FEATS)
        rows = len(eeg)
        offset = (rows - 10_000) // 2
        eeg = eeg.iloc[offset : offset + 10_000]

        if display:
            plt.figure(figsize=(10, 5))
            offset = 0

        # CONVERT TO NUMPY
        data = np.zeros((10_000, len(FEATS)))
        for j, col in enumerate(FEATS):

            # FILL NAN
            x = eeg[col].values.astype("float32")
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            data[:, j] = x

            if display:
                if j != 0:
                    offset += x.max()
                plt.plot(range(10_000), x - offset, label=col)
                offset -= x.min()

        if display:
            plt.legend()
            name = parquet_path.split("/")[-1]
            name = name.split(".")[0]
            plt.title(f"EEG {name}", size=16)
            plt.show()

        return data
