import os
import numpy as np
import pandas as pd
import concurrent.futures
import torch
from brain_solver import Filters, FilterType, Wav2Vec2 as w2v
import pytorch_lightning as pl
from transformers.utils import logging
from tqdm import tqdm

# Suppress warnings if desired
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.CRITICAL)

# Setup for CUDA device selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Configuration
full_path = "/home/osloup/NoodleNappers/brain/data/"  # Luppo
from brain_solver import Config

config = Config(
    full_path,
    full_path + "out/",
    USE_EEG_SPECTROGRAMS=True,
    USE_KAGGLE_SPECTROGRAMS=True,
    should_read_brain_spectograms=False,
    should_read_eeg_spectrogram_files=False,
    USE_PRETRAINED_MODEL=False,
)

import sys

sys.path.append(full_path + "kaggle-kl-div")

# Ensure the output directory exists
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

# Initialize random environment
pl.seed_everything(config.seed, workers=True)

# Filter parameters
cutoffs = {
    FilterType.LOWPASS: 50.0,
    FilterType.HIGHPASS: 0.1,
    FilterType.BANDPASS: [0.1, 50.0],
    FilterType.BANDSTOP: [45.0, 55.0],
}
fs = 250  # Sampling rate (Hz)

# Initialize Filters class
ft = Filters(order=5)


def process_file(file_name):
    name = file_name[:-8]
    force_regenerate = False

    w2v_only_dir = os.path.join(config.data_w2v_specs, "wav2vec_only")
    os.makedirs(w2v_only_dir, exist_ok=True)
    w2v_only_output_filename = os.path.join(w2v_only_dir, f"{name}.npy")

    if not os.path.exists(w2v_only_output_filename) or force_regenerate:
        try:
            parquet_file = pd.read_parquet(
                os.path.join(config.data_spectograms, file_name)
            )
            data_for_processing = parquet_file.iloc[:, 1:].values
            w2v_only_output = w2v.wav2vec2(data_for_processing)
            np.save(w2v_only_output_filename, w2v_only_output)
        except Exception as e:
            return f"ERROR: An unexpected error occurred for {name} (wav2vec only): {e}"

    results = []
    for filter_type, cutoff in cutoffs.items():
        raw_dir = os.path.join(config.data_w2v_specs, filter_type.name, "raw")
        w2v_dir = os.path.join(config.data_w2v_specs, filter_type.name, "w2v")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(w2v_dir, exist_ok=True)
        raw_output_filename = os.path.join(raw_dir, f"{name}.npy")
        w2v_output_filename = os.path.join(w2v_dir, f"{name}.npy")

        if (
            not os.path.exists(raw_output_filename)
            or force_regenerate
            or not os.path.exists(w2v_output_filename)
            or force_regenerate
        ):
            try:
                if "data_for_processing" not in locals():
                    parquet_file = pd.read_parquet(
                        os.path.join(config.data_spectograms, file_name)
                    )
                    data_for_processing = parquet_file.iloc[:, 1:].values

                filtered_spectrogram = ft.apply_filter_to_spectrogram(
                    data_for_processing, cutoff, fs, filter_type
                )
                np.save(raw_output_filename, filtered_spectrogram)
                w2v_output = w2v.wav2vec2(filtered_spectrogram)
                np.save(w2v_output_filename, w2v_output)
            except Exception as e:
                results.append(f"ERROR: An unexpected error occurred for {name}: {e}")
    return results


def main():
    files = os.listdir(config.data_spectograms)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files)))
    for result in results:
        if result:
            print(result)


if __name__ == "__main__":
    main()
