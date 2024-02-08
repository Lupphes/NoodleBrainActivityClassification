import torch.optim as optim
import torch.nn as nn
import torch


class Trainer:
    def __init__(self, model, data_loaders, num_classes, lr=0.01):
        self.model = model
        self.data_loaders = data_loaders
        self.lr = lr
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modify the model's final layer to match the number of classes
        num_ftrs = model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=10):
        for epoch in range(epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.data_loaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(
                    self.data_loaders[phase].dataset
                )
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
