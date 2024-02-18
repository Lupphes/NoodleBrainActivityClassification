import numpy as np
import torch

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import gc

from .eeg_dataset import EEGDataset
from .trainer import Trainer as tr


class BrainModel:
    @staticmethod
    def cross_validate_eeg(
        config,
        train_data_preprocessed,
        spectrograms,
        data_eeg_spectograms,
        TARGETS,
        n_splits=5,
        batch_size_train=32,
        batch_size_valid=64,
        max_epochs=4,
        num_workers=3,
    ):
        """
        Performs cross-validation on EEG data using GroupKFold.

        Parameters:
        - train_data_preprocessed: DataFrame containing the training data.
        - spectrograms: The preloaded spectrogram data.
        - data_eeg_spectograms: Path or container with EEG spectrogram data.
        - TARGETS: List of target columns in the training data.
        - n_splits: Number of splits for cross-validation.
        - batch_size_train: Batch size for training dataloader.
        - batch_size_valid: Batch size for validation dataloader.
        - max_epochs: Maximum number of epochs for training.
        - num_workers: Number of workers for DataLoader.
        """
        all_oof = []
        all_true = []
        valid_loaders = []

        gkf = GroupKFold(n_splits=n_splits)
        for i, (train_index, valid_index) in enumerate(
            gkf.split(
                train_data_preprocessed,
                train_data_preprocessed.target,
                train_data_preprocessed.patient_id,
            )
        ):
            print("#" * 25)
            print(f"### Fold {i+1}")

            train_ds = EEGDataset(
                train_data_preprocessed.iloc[train_index],
                spectrograms,
                data_eeg_spectograms,
                TARGETS,
            )
            train_loader = DataLoader(
                train_ds,
                shuffle=True,
                batch_size=batch_size_train,
                num_workers=num_workers,
            )
            valid_ds = EEGDataset(
                train_data_preprocessed.iloc[valid_index],
                spectrograms,
                data_eeg_spectograms,
                TARGETS,
                mode="valid",
            )
            valid_loader = DataLoader(
                valid_ds,
                shuffle=False,
                batch_size=batch_size_valid,
                num_workers=num_workers,
            )

            print(f"### Train size: {len(train_index)}, Valid size: {len(valid_index)}")
            print("#" * 25)

            trainer = pl.Trainer(max_epochs=max_epochs)
            model = tr(
                config.trained_weight_file,
                config.USE_KAGGLE_SPECTROGRAMS,
                config.USE_EEG_SPECTROGRAMS,
            )
            if config.trained_model_path is None:
                trainer.fit(model=model, train_dataloaders=train_loader)
                trainer.save_checkpoint(f"EffNet_v{config.VER}_f{i}.ckpt")

            valid_loaders.append(valid_loader)
            all_true.append(train_data_preprocessed.iloc[valid_index][TARGETS].values)

            del trainer, model
            gc.collect()

        return all_oof, all_true, valid_loaders

    @staticmethod
    def validate_model_across_folds(config, device, all_oof, all_true, valid_loaders):
        """
        Validates a model across different folds and returns the out-of-fold predictions.

        Parameters:
        - config: Configuration object with attributes like VER (version), trained_model_path, and trained_weight_file.
        - device: The device (CPU or GPU) to run the validation on.
        - valid_loaders: A list of DataLoader objects for validation, one for each fold.

        Returns:
        - all_oof: Numpy array of concatenated out-of-fold predictions.
        """
        for i in range(5):
            print("#" * 25)
            print(f"### Validating Fold {i+1}")

            ckpt_file = (
                f"EffNet_v{config.VER}_f{i}.ckpt"
                if config.trained_model_path is None
                else f"{config.trained_model_path}/EffNet_v{config.VER}_f{i}.ckpt"
            )
            model = tr.load_from_checkpoint(
                ckpt_file,
                weight_file=config.trained_weight_file,
                use_kaggle_spectrograms=config.USE_KAGGLE_SPECTROGRAMS,
                use_eeg_spectrograms=config.USE_EEG_SPECTROGRAMS,
            )
            model = model.to(device).eval()
            with torch.inference_mode():  # Use inference mode for efficiency
                for val_batch in valid_loaders[i]:
                    val_batch = val_batch.to(
                        device
                    )  # Move validation batch to the correct device
                    oof = (
                        torch.softmax(model(val_batch), dim=1).cpu().numpy()
                    )  # Get predictions
                    all_oof.append(oof)  # Collect predictions
            del model
            gc.collect()
            torch.cuda.empty_cache()

        all_oof = np.concatenate(all_oof)
        all_true = np.concatenate(all_true)

        return all_oof, all_true
