import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# Define the model architecture
class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor
        x = torch.relu(self.layer_1(x))
        return self.layer_2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # Ensure input shape is correct for the model
        return x.view(28 * 28), y  # Flatten the input to match the model input

# Generate synthetic data (Replace this with your actual data)
data = torch.randn(1000, 1, 28, 28)  # Simulating a dataset of 28x28 images
labels = torch.randint(0, 10, (1000,))  # Simulating random labels from 0-9

# Create dataset and dataloader
custom_dataset = CustomDataset(data, labels)

# IMPORTANT: Set num_workers=0 for Windows to avoid issues
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=0)

# Initialize model
model = LitModel()

# Initialize trainer with CPU settings
trainer = pl.Trainer(max_epochs=10, accelerator='cpu')  # Use CPU

# Train the model
trainer.fit(model, train_loader)
