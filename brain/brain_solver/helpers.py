import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pywt
import librosa

import torch


class Helpers:
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]

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

    @staticmethod
    def read_spectrograms(
        path, data_path_train_on_brain_spectograms_dataset_specs, read_files=False
    ):
        """
        Reads spectrogram data from a given path. Either reads individual parquet files
        or loads a pre-processed numpy file based on the flag provided.

        Parameters:
        - path: str, path to the directory containing spectrogram files or to the numpy file.
        - read_files: bool, if True reads parquet files, if False loads a numpy file.

        Returns:
        - A dictionary with keys as the spectrogram names (integers) and values as numpy arrays of the spectrogram data.
        """
        if read_files:
            files = os.listdir(path)[:500]
            print(f"There are {len(files)} spectrogram parquets")
            spectrograms = {}
            for i, f in enumerate(files):
                if i % 100 == 0:
                    print(i, ", ", end="")
                try:
                    tmp = pd.read_parquet(os.path.join(path, f))
                except:
                    pass

                name = int(f.split(".")[0])
                spectrograms[name] = tmp.iloc[:, 1:].values
        else:
            # Assuming the numpy file is stored one level up from the given path
            npy_path = os.path.join(
                os.path.dirname(path),
                data_path_train_on_brain_spectograms_dataset_specs,
            )
            spectrograms = np.load(npy_path, allow_pickle=True).item()

        return spectrograms

    @staticmethod
    def read_eeg_spectrograms(
        train_df, eeg_spectrogram_path, eeg_specs_npy_path, read_files
    ):
        """
        Reads EEG spectrogram data either from individual .npy files based on EEG IDs
        or loads a pre-processed numpy file containing all EEG spectrograms.

        Parameters:
        - train_df: pandas.DataFrame, DataFrame with a column 'eeg_id' containing EEG IDs.
        - read_files: bool, if True reads .npy files for each EEG ID, if False loads a pre-processed .npy file.
        - eeg_spectrogram_path: str, path to the directory containing individual .npy EEG spectrogram files.
        - eeg_specs_npy_path: str, path to the pre-processed .npy file containing all EEG spectrograms.

        Returns:
        - A dictionary with keys as EEG IDs and values as numpy arrays of the EEG spectrogram data.
        """
        all_eegs = {}
        if read_files:
            for i, e in enumerate(train_df["eeg_id"].values):
                if i % 100 == 0:
                    print(f"{i}, ", end="")
                file_path = f"{eeg_spectrogram_path}{e}.npy"
                x = np.load(file_path)
                all_eegs[e] = x
        else:
            all_eegs = np.load(eeg_specs_npy_path, allow_pickle=True).item()

        return all_eegs

    @staticmethod
    # DENOISE FUNCTION
    def maddest(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    @staticmethod
    def denoise(x, wavelet="haar", level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1 / 0.6745) * Helpers.maddest(coeff[-level])

        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

        ret = pywt.waverec(coeff, wavelet, mode="per")

        return ret

    @staticmethod
    def spectrogram_from_eeg(parquet_path, display=False, USE_WAVELET=None):

        # LOAD MIDDLE 50 SECONDS OF EEG SERIES
        eeg = pd.read_parquet(parquet_path)
        middle = (len(eeg) - 10_000) // 2
        eeg = eeg.iloc[middle : middle + 10_000]

        # VARIABLE TO HOLD SPECTROGRAM
        img = np.zeros((128, 256, 4), dtype="float32")

        if display:
            plt.figure(figsize=(10, 7))
        signals = []
        for k in range(4):
            COLS = Helpers.FEATS[k]

            for kk in range(4):

                # COMPUTE PAIR DIFFERENCES
                x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

                # FILL NANS
                m = np.nanmean(x)
                if np.isnan(x).mean() < 1:
                    x = np.nan_to_num(x, nan=m)
                else:
                    x[:] = 0

                # DENOISE
                if USE_WAVELET:
                    x = Helpers.denoise(x, wavelet=USE_WAVELET)
                signals.append(x)

                # RAW SPECTROGRAM
                mel_spec = librosa.feature.melspectrogram(
                    y=x,
                    sr=200,
                    hop_length=len(x) // 256,
                    n_fft=1024,
                    n_mels=128,
                    fmin=0,
                    fmax=20,
                    win_length=128,
                )

                # LOG TRANSFORM
                width = (mel_spec.shape[1] // 32) * 32
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                    np.float32
                )[:, :width]

                # STANDARDIZE TO -1 TO 1
                mel_spec_db = (mel_spec_db + 40) / 40
                img[:, :, k] += mel_spec_db

            # AVERAGE THE 4 MONTAGE DIFFERENCES
            img[:, :, k] /= 4.0

            if display:
                plt.subplot(2, 2, k + 1)
                plt.imshow(img[:, :, k], aspect="auto", origin="lower")
                plt.title(f"EEG {1} - Spectrogram {Helpers.NAMES[k]}")

        if display:
            plt.show()
            plt.figure(figsize=(10, 5))
            offset = 0
            for k in range(4):
                if k > 0:
                    offset -= signals[3 - k].min()
                plt.plot(range(10_000), signals[k] + offset, label=Helpers.NAMES[3 - k])
                offset += signals[3 - k].max()
            plt.legend()
            plt.title(f"EEG {2} Signals")
            plt.show()
            print()
            print("#" * 25)
            print()

        return img

    @staticmethod
    def plot_spectrograms(
        dataloader, train_data_preprocessed, ROWS=2, COLS=3, BATCHES=2
    ):
        """
        Plots spectrograms from the dataloader batches along with their corresponding labels and EEG IDs.

        Parameters:
        - dataloader: DataLoader instance from which to draw batches of data.
        - train_data_preprocessed: DataFrame containing preprocessed training data, including EEG IDs.
        - ROWS: int, number of rows per figure.
        - COLS: int, number of columns per figure.
        - BATCHES: int, number of batches to plot before stopping.
        """
        for i, (x, y) in enumerate(dataloader):
            plt.figure(figsize=(20, 8))
            for j in range(ROWS):
                for k in range(COLS):
                    index = j * COLS + k
                    if (
                        index >= x.shape[0]
                    ):  # Check to avoid index error in the last batch
                        break
                    plt.subplot(ROWS, COLS, index + 1)
                    t = y[index]
                    img = torch.flip(x[index, :, :, 0], (0,)).T
                    mn = img.flatten().min()
                    mx = img.flatten().max()
                    img = (img - mn) / (mx - mn)
                    plt.plot(img)

                    tars = f"[{t[0]:0.2f}]"
                    for s in t[1:]:
                        tars += f", {s:0.2f}"
                    # Adjust the calculation of `eeg_id` index if necessary
                    eeg_id_index = i * (ROWS * COLS) + index
                    eeg = train_data_preprocessed.eeg_id.values[eeg_id_index]
                    plt.title(f"EEG = {eeg}\nTarget = {tars}", size=12)
                    plt.ylabel("y", size=14)
                    plt.xlabel("i", size=14)
            plt.show()

            if i == BATCHES - 1:
                break
