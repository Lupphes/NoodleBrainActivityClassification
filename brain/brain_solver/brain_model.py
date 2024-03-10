import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from d2l import torch as d2l

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import gc

from .eeg_dataset import EEGDataset
from .network import Network
from .trainer import Trainer


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
        max_epochs_first_stage=5,
        max_epochs_second_stage=3,
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

            model = Network(
                config.trained_weight_file,
                config.USE_KAGGLE_SPECTROGRAMS,
                config.USE_EEG_SPECTROGRAMS,
            ).to(device)

            if config.trained_model_path is None or config.FINE_TUNE:
                if config.FINE_TUNE:
                    # Freeze layers?
                    lr = 1e-2
                    # add more layers to model
                    for param in model.base_model.parameters():
                        param.requires_grad = False

                    for param in model.base_model.avgpool.parameters():
                        param.requires_grad = True

                    for param in model.base_model.classifier.parameters():
                        param.requires_grad = True
                else:
                    lr = 1e-3

                # First training stage
                criterion = nn.KLDivLoss(reduction="batchmean")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                lr_1 = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=BrainModel.lrfn
                )
                BrainModel.train(
                    model,
                    max_epochs_first_stage,
                    criterion,
                    optimizer,
                    train_loader,
                    valid_loader_training,
                    device,
                    lr_1,
                )

                # Second training stage
                train_ds2 = EEGDataset(
                    train_data_preprocessed.iloc[train_index][
                        train_data_preprocessed.iloc[train_index]["kl"] < 5.5
                    ],
                    spectrograms,
                    data_eeg_spectograms,
                    TARGETS,
                )
                train_loader2 = DataLoader(
                    train_ds2,
                    shuffle=True,
                    batch_size=batch_size_train,
                    num_workers=num_workers,
                )
                print(
                    f"### Second stage train size {len(data)}, valid size {len(valid_index)}"
                )
                print("#" * 25)
                lr = 1e-5
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                lr_2 = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=BrainModel.lrfn2
                )
                BrainModel.train(
                    model,
                    max_epochs_second_stage,
                    criterion,
                    optimizer,
                    train_loader2,
                    valid_loader_training,
                    device,
                    lr_2,
                )

                # Save model
                torch.save(
                    model,
                    config.output_path + f"EffNet_version{config.VER}_fold{i+1}.pth",
                )

            valid_loaders.append(valid_loader)
            all_true.append(train_data_preprocessed.iloc[valid_index][TARGETS].values)

            del model
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
                config.output_path + f"EffNet_version{config.VER}_fold{i+1}.pth"
                if config.trained_model_path is None or config.FINE_TUNE
                else f"{config.trained_model_path}EffNet_v{config.VER}_f{i}.ckpt"
            )
            if config.trained_model_path is None or config.FINE_TUNE:
                model = torch.load(ckpt_file)
            else:
                model = Trainer.load_from_checkpoint(
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
        acc = BrainModel.custom_accuracy(out, y)
        return loss, acc, len(y)

    @staticmethod
    def custom_accuracy(predicted_probs, true_probs):
        # Convert predicted probabilities and true probabilities to numpy arrays
        predicted_probs = torch.softmax(predicted_probs, dim=-1).detach().cpu().numpy()
        true_probs = true_probs.detach().cpu().numpy()

        # Calculate the absolute difference between predicted and true probabilities
        abs_diff = np.abs(predicted_probs - true_probs)

        # Calculate the accuracy as the percentage of overlap or similarity
        # 1.0 minus the mean absolute difference normalized by the sum of true probabilities
        accuracies = 1.0 - np.mean(abs_diff, axis=-1) / np.sum(true_probs, axis=-1)

        return torch.tensor(np.mean(accuracies))

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
        acc = BrainModel.custom_accuracy(out, y)
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

    def lrfn(epoch):
        return [1e-3, 1e-3, 1e-4, 1e-4, 1e-5][epoch]

    def lrfn2(epoch):
        return [1e-5, 1e-5, 1e-6][epoch]

    @staticmethod
    def train(
        model,
        num_epochs,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        lr_scheduler,
    ):
        animator = d2l.Animator(
            xlabel="epoch",
            xlim=[1, num_epochs],
            figsize=(10, 5),
            legend=[
                "train loss",
                "train accuracy",
                "validation loss",
                "validation accuracy",
            ],
        )

        for epoch in range(num_epochs):
            print("Epoch: ", epoch + 1)
            total_count = 0
            total_loss = 0
            total_accuracy = 0
            # Training Phase
            for batch in tqdm(train_loader):
                # Calculate Loss
                loss, acc, len_y = BrainModel.training_step(
                    model, batch, criterion, device
                )
                # Compute Gradients
                loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()
                # Reset Gradients
                optimizer.zero_grad()
                total_count += len_y
                total_loss += len_y * loss.item()
                total_accuracy += len_y * acc

            # Update learning rate
            lr_scheduler.step()

            # Validation Phase
            train_loss = total_loss / total_count
            train_accuracy = total_accuracy / total_count
            result = BrainModel.validate(model, val_loader, criterion, device)
            print(
                f"val_loss: {result['val_loss']:.2f}, val_acc: {result['val_acc']:.2f}"
            )
            animator.add(
                epoch + 1,
                (train_loss, train_accuracy, result["val_loss"], result["val_acc"]),
            )
