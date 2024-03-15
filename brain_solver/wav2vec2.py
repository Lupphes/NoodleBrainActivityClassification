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
        Pre-processes the EEG data for use by wav2vec2.

        Arguments:
        - data (ndarray): raw EEG data.
        - model_path (str): the path to the wav2vec2 model to use.
        - min_length (int): the minimum length of the data.

        Returns:
        - list: the pre-processed data.
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
        Pre-processes the spectograms for use by wav2vec2.

        Arguments:
        - data: spectograms.
        - model_path: the path to the wav2vec2 model to use.

        Returns:
        - list: the pre-processed data.
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

        Arguments:
        - data: the pre-processed data.
        - model_path: the path to the wav2vec2 model to use.

        Returns:
        - list: the output from the last hidden layer.
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
        Preprocesses the data and applies the Wav2Vec2 algorithm to the raw EEG data or spectograms.

        Arguments:
        - parquet_file: the parquet file containing the data.
        - proc_eegs: whether to process EEG data or spectograms.
        - model_path: the path to the wav2vec2 model to use.

        Returns:
        - list: the output from the last hidden layer.
        """
        if proc_eegs:
            data_preprocessed = Wav2Vec2.preprocess_eeg_data(parquet_file, model_path)
        else:
            data_preprocessed = Wav2Vec2.preprocess_spec_data(parquet_file, model_path)
        w2v_output = Wav2Vec2.process_with_wav2vec2(data_preprocessed, model_path)
        return w2v_output
