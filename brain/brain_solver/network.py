import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class Network(nn.Module):
    def __init__(self, weight_file, use_kaggle_spectrograms, use_eeg_spectrograms):
        super().__init__()
        self.use_kaggle_spectrograms = use_kaggle_spectrograms
        self.use_eeg_spectrograms = use_eeg_spectrograms
        self.base_model = efficientnet_b0()
        if weight_file:
            self.base_model.load_state_dict(torch.load(weight_file))
        # Update the classifier layer to match the number of target classes
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, 6, dtype=torch.float32
        )
        self.prob_out = nn.Softmax()

    def forward(self, x):
        # Split the input into two groups
        # TEMPORARY SOLUTION
        x1 = x
        x2 = x
        #x1 = [x[:, :, :, i : i + 1] for i in range(4)]
        #x1 = torch.concat(x1, dim=1)
        #x2 = [x[:, :, :, i + 4 : i + 5] for i in range(4)]
        #x2 = torch.concat(x2, dim=1)

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
