import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# This class is a simple wrapper around the EfficientNet B0 model but compared to Network,
# this one is a PyTorch Lightning Module so can be trained using PyTorch Lightning
class Trainer(pl.LightningModule):

    def __init__(
        self, weight_file, use_kaggle_spectrograms=False, use_eeg_spectrograms=True
    ):
        super().__init__()
        self.use_kaggle_spectrograms = use_kaggle_spectrograms
        self.use_eeg_spectrograms = use_eeg_spectrograms
        self.base_model = efficientnet_b0()
        self.base_model.load_state_dict(torch.load(weight_file))
        # Update the classifier layer to match the number of target classes
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, 6
        )
        self.prob_out = nn.Softmax(dim=1)

    def forward(self, x):
        # Split the input into two groups
        x1 = [x[:, :, :, i : i + 1] for i in range(4)]
        x1 = torch.concat(x1, dim=1)
        x2 = [x[:, :, :, i + 4 : i + 5] for i in range(4)]
        x2 = torch.concat(x2, dim=1)

        # Select the appropriate input based on initialization parameters
        if self.use_kaggle_spectrograms and self.use_eeg_spectrograms:
            x = torch.concat([x1, x2], dim=2)
        elif self.use_eeg_spectrograms:
            x = x2
        else:
            x = x1

        # Expand the input to have 3 channels
        x = torch.concat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)

        out = self.base_model(x)
        return out

    # One training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = torch.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(out, y)
        return loss

    # One validation step
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.softmax(self.forward(batch), dim=1)

    # Configure the optimizer
    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
