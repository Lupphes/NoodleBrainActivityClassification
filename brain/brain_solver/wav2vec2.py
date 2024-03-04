import numpy as np
import torch
from torchaudio.transforms import Resample
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
import os


class Wav2Vec2:
    CHANNELS = [
        "Fp1",
        "F7",
        "T3",
        "T5",
        "O1",
        "F3",
        "C3",
        "P3",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "O2",
        "F8",
        "T4",
        "T6",
    ]
    SAMPLING_RATE = 200
    TARGET_SAMPLING_RATE = 16000

    @staticmethod
    def preprocess_eeg_data(
        data, model_path="facebook/wav2vec2-base-960h", min_length=16000
    ):
        """
        Pre-processes the data for use by wav2vec2, ensuring data meets minimum length requirements.
        """
        resampler = Resample(
            orig_freq=Wav2Vec2.SAMPLING_RATE,
            new_freq=Wav2Vec2.TARGET_SAMPLING_RATE,
            dtype=torch.float32,
        )
        model = Wav2Vec2Processor.from_pretrained(model_path)

        data_preprocessed = []
        for i in range(len(Wav2Vec2.CHANNELS)):
            # Resample data
            data_resampled = resampler(
                torch.tensor(data[i], dtype=torch.float32).unsqueeze(0)
            ).numpy()[0]

            # Ensure data meets minimum length
            if data_resampled.shape[1] < min_length:
                # Calculate padding (total and for each side)
                total_padding = min_length - data_resampled.shape[1]
                padding_left = total_padding // 2
                padding_right = total_padding - padding_left

                # Pad data
                data_resampled = np.pad(
                    data_resampled,
                    ((0, 0), (padding_left, padding_right)),
                    "constant",
                    constant_values=0,
                )

            # Normalize data
            data_normalized = model(
                data_resampled,
                sampling_rate=Wav2Vec2.TARGET_SAMPLING_RATE,
                return_tensors="pt",
                padding=True,
            )
            data_preprocessed.append(data_normalized)

        return data_preprocessed

    @staticmethod
    def preprocess_spec_data(data, model_path="facebook/wav2vec2-base-960h"):
        """
        Pre-processes the data for use by wav2vec2.
        """

        model = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
        data_normalized = model(
            data,
            sampling_rate=Wav2Vec2.TARGET_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )

        return [data_normalized]

    @staticmethod
    def process_with_wav2vec2(data, model_path="facebook/wav2vec2-base-960h"):
        """
        Applies wav2vec2 to the given data and returns the values in the last hidden layer.
        """

        model = Wav2Vec2ForCTC.from_pretrained(model_path)

        w2v_output = []
        for ch_data in data:
            outputs = model(**ch_data, output_hidden_states=True)
            features = outputs.hidden_states[-1][0]
            w2v_output.append(pd.DataFrame(features.detach().numpy()))

        return w2v_output

    @staticmethod
    def wav2vec2(
        parquet_file, proc_eegs=False, model_path="facebook/wav2vec2-base-960h"
    ):
        """
        Applies the Wav2Vec2 algorithm to the raw EEG data given as argument
        and writes the result to a folder data/w2v_eegs

        Arguments:
        - data (ndarray): raw EEG data from a parquet file.
        - data_path: the path leading to the data folder.
        - out_filename: the desired name of the output file (preferrably the parquet filename without .parquet).
        """

        if proc_eegs:
            data_preprocessed = Wav2Vec2.preprocess_eeg_data(parquet_file, model_path)
        else:
            data_preprocessed = Wav2Vec2.preprocess_spec_data(parquet_file, model_path)
        w2v_output = Wav2Vec2.process_with_wav2vec2(data_preprocessed, model_path)
        return w2v_output
