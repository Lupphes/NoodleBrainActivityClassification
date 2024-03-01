import torch
from torchaudio.transforms import Resample
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
import os
import numpy as np

class Wav2Vec2:
    CHANNELS = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    SAMPLING_RATE = 200
    TARGET_SAMPLING_RATE = 16000

    @staticmethod
    def preprocess_eeg_data(data):
        """
        Pre-processes the data for use by wav2vec2.
        """
    
        resampler = Resample(orig_freq=Wav2Vec2.SAMPLING_RATE, new_freq=Wav2Vec2.TARGET_SAMPLING_RATE, dtype=torch.float64)
        model = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        data_preprocessed = []
        for i in range(len(Wav2Vec2.CHANNELS)):
            data_resampled = resampler(torch.tensor(data[i]).unsqueeze(0)).numpy()[0]
            data_normalized = model(data_resampled, sampling_rate=Wav2Vec2.TARGET_SAMPLING_RATE, return_tensors="pt", padding=True)
            data_preprocessed.append(data_normalized)

        return data_preprocessed
    
    @staticmethod
    def preprocess_spec_data(data):
        """
        Pre-processes the data for use by wav2vec2.
        """
    
        model = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        data_normalized = model(data, sampling_rate=Wav2Vec2.TARGET_SAMPLING_RATE, return_tensors="pt", padding=True)

        return [data_normalized]
    
    @staticmethod
    def process_with_wav2vec2(data):
        """
        Applies wav2vec2 to the given data and returns the values in the last hidden layer.
        """

        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        w2v_output = []
        for ch_data in data:
            outputs = model(**ch_data, output_hidden_states=True)
            features = outputs.hidden_states[-1][0]
            w2v_output.append(pd.DataFrame(features.detach().numpy()))
        
        return w2v_output

    @staticmethod
    def wav2vec2(data, out_path, proc_eegs=False):
        """
        Applies the Wav2Vec2 algorithm to the raw EEG data given as argument
        and writes the result to a folder data/w2v_eegs

        Arguments:
        - data: the data which should be processed with wav2vec
        - out_path: the path to which the files should be output
        - proc_eegs: whether the function should process the input as EEGs or spectrograms
        """
        w2v_processed = {}
        for name, it in data.items():
            if proc_eegs:
                data_preprocessed = Wav2Vec2.preprocess_eeg_data(it)
            else:
                data_preprocessed = Wav2Vec2.preprocess_spec_data(it)
            out = Wav2Vec2.process_with_wav2vec2(data_preprocessed)
            np.save(out_path + str(name), out)
            w2v_processed[name] = out
        np.save('w2v_specs2', w2v_processed)