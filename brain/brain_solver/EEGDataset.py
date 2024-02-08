from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F


class EEGDataset(Dataset):
    def __init__(self, df, data_eeg, transform=None):
        self.df = df
        self.data_eeg = data_eeg
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_file_path = os.path.join(self.data_eeg, f"{row['eeg_id']}.parquet")
        eeg_data = pd.read_parquet(eeg_file_path)

        # Convert to tensor, ensure dtype is float32, and add a channel dimension
        X = torch.from_numpy(eeg_data.values.astype(np.float32))
        X = X.unsqueeze(0)  # Correctly add a channel dimension without re-wrapping

        if self.transform:
            X = self.transform(X)

        y = torch.tensor(row["target_encoded"], dtype=torch.long)

        return X, y
