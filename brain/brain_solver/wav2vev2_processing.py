import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import pandas as pd
from brain_solver import helpers as hp, Config


class Wav2VecProcessing:
    pass


def wav2vec2(spectrograms):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    decoded = {}
    for file_name, specs in spectrograms.items():
        features = processor(spectrograms, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(features.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded[file_name] = processor.batch_decode(predicted_ids)
    return decoded