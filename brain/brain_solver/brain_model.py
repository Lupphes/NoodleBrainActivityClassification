import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import gc

from .eeg_dataset import EEGDataset
from .trainer import Trainer as tr
from .network import Network


class BrainModel:
    @staticmethod
    def cross_validate_eeg(
        config,
        device,
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
            valid_ds_training = EEGDataset(
                train_data_preprocessed.iloc[valid_index],
                spectrograms,
                data_eeg_spectograms,
                TARGETS,
            )
            valid_loader_training = DataLoader(
                valid_ds_training,
                shuffle=False,
                batch_size=batch_size_valid,
                num_workers=num_workers,
            )

            print(f"### Train size: {len(train_index)}, Valid size: {len(valid_index)}")
            print("#" * 25)

            # trainer = pl.Trainer(max_epochs=max_epochs)
            model = Network(
                config.trained_weight_file,
                config.USE_KAGGLE_SPECTROGRAMS,
                config.USE_EEG_SPECTROGRAMS,
            ).to(device)
            if config.trained_model_path is None:
                # trainer.fit(model=model, train_dataloaders=train_loader)
                # trainer.save_checkpoint(f"EffNet_v{config.VER}_f{i}.ckpt")
                criterion = nn.KLDivLoss(reduction="batchmean")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                BrainModel.train(
                    model,
                    max_epochs,
                    criterion,
                    optimizer,
                    train_loader,
                    valid_loader_training,
                    device,
                )
                # STORE THE MODEL EACH EPOCH AND MAKE GRAPH

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

    @staticmethod
    def training_step(model, batch, criterion, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        out = torch.log_softmax(out, dim=1)
        loss = criterion(out, y)
        return loss

    # @staticmethod
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return torch.softmax(self.forward(batch), dim=1)
    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    @staticmethod
    def validation_step(model, batch, criterion, device):
        # Prepare batch data
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # Generate predictions
        out = model(x)
        out = torch.log_softmax(out, dim=1)
        # Calculate Loss
        loss = criterion(out, y)
        # Calculate Accuracy
        acc = BrainModel.accuracy(out, y)
        return {"val_loss": loss, "val_acc": acc}

    @staticmethod
    def validate(model, val_loader, criterion, device):
        with torch.no_grad():
            model.eval()
            outputs = [
                BrainModel.validation_step(model, batch, criterion, device)
                for batch in tqdm(val_loader)
            ]
            batch_losses = [x["val_loss"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()
            batch_accs = [x["val_acc"] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()
            return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    @staticmethod
    def train(
        model, num_epochs, criterion, optimizer, train_loader, val_loader, device
    ):
        for epoch in range(num_epochs):
            print("Epoch: ", epoch + 1)
            # Training Phase
            for batch in tqdm(train_loader):
                # Calculate Loss
                loss = BrainModel.training_step(model, batch, criterion, device)
                # Compute Gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # Reset Gradients
                optimizer.zero_grad()

            # Validation Phase
            result = BrainModel.validate(model, val_loader, criterion, device)
            print(
                f"val_loss: {result['val_loss']:.2f}, val_acc: {result['val_acc']:.2f}"
            )
