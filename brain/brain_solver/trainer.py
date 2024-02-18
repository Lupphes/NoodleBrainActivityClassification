import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class Trainer(pl.LightningModule):
    def __init__(self, weight_file, use_kaggle_spectrograms, use_eeg_spectrograms):
        super().__init__()
        self.use_kaggle_spectrograms = use_kaggle_spectrograms
        self.use_eeg_spectrograms = use_eeg_spectrograms
        self.base_model = efficientnet_b0()
        self.base_model.load_state_dict(torch.load(weight_file))
        # Update the classifier layer to match the number of target classes
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, 6, dtype=torch.float32
        )
        self.prob_out = nn.Softmax()

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = torch.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(out, y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.softmax(self.forward(batch), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
