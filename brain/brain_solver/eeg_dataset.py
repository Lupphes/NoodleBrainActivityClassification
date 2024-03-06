import numpy as np
from torch.utils.data import Dataset
import albumentations as albu
class EEGDataset(Dataset):
    def __init__(self, data, specs, eeg_specs, targets, augment=False, mode="train", w2v_enabled=False, model_eegs=True, raw_eegs=False):
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.targets = targets
        self.augment = augment
        self.mode = mode
        self.w2v_enabled = w2v_enabled
        self.raw_eegs = raw_eegs
        self.model_eegs = model_eegs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
        if self.augment:
            X = self.__augment(X)
        if self.mode == "train":
            return list(zip(X, y))
        else:
            return X

    def _generate_data(self, indexes):

        if self.raw_eegs:
            X = np.zeros((len(indexes), 19, 4, 768, 8), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")
            
            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                else:
                    r = int((row["min_offset"] + row["max_offset"]) // 4)

                for k in range(4):
                    X[j, :, :, k] = self.specs[row.spec_id]

        if self.w2v_enabled:

            X = np.zeros((len(indexes), 1, 768, 8), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")
            
            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                else:
                    r = int((row["min_offset"] + row["max_offset"]) // 4)

                for k in range(4):
                    X[j, :, :, :, k] = self.specs[row.spec_id]

        else:
            X = np.zeros((len(indexes), 128, 256, 8), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")
            img = np.ones((128, 256), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                else:
                    r = int((row["min_offset"] + row["max_offset"]) // 4)

                for k in range(4):
                    # EXTRACT 300 ROWS OF SPECTROGRAM
                    img = self.specs[row.spec_id][r : r + 300, k * 100 : (k + 1) * 100].T

                    # LOG TRANSFORM SPECTROGRAM
                    img = np.clip(img, np.exp(-4), np.exp(8))
                    img = np.log(img)

                    # STANDARDIZE PER IMAGE
                    ep = 1e-6
                    m = np.nanmean(img.flatten())
                    s = np.nanstd(img.flatten())
                    img = (img - m) / (s + ep)
                    img = np.nan_to_num(img, nan=0.0)

                    # CROP TO 256 TIME STEPS
                    img = img[:, 22:-22]
                    
                    # pad image if necessary
                    if img.shape[1] < 256:
                        img = np.pad(img, ((0, 0), (0, 256-img.shape[1])))

                    X[j, 14:-14, :, k] = img / 2.0

        # EEG SPECTROGRAMS
        if self.model_eegs:
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

        if self.mode != "test":
            y[j,] = row[self.targets]

        return X, y

    def _random_transform(self, img):
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                # albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
            ]
        )
        return composition(image=img)["image"]

    def __augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        return img_batch