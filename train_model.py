# Importing MNIST dataset from torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# Importing torch.nn and torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F

# Importing torch.optim
import torch
import torch.optim as optim

import tqdm
from datetime import datetime
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block of convolutional layers (input is 3 channels now)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.fc_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fc_conv(x)
        # x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)


def train():
    # Set device and random seed
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset with augmentation
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model, criterion, optimizer
    model = Net().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Save model with timestamp and git info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha[:7]
        model_name = f'mnist_model_{timestamp}_commit_{commit_hash}.pth'
    except:
        model_name = f'mnist_model_{timestamp}.pth'
    
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/{model_name}')
    print(f"\nModel saved as: {model_name}")
    
if __name__ == '__main__':
    train() 
