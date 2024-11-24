import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from datetime import datetime
import os
import tqdm
import random
import numpy as np

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    # Set seeds for reproducibility
    set_seed(42)
    
    # Force CPU
    device = torch.device('cpu')
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check if data exists
    data_dir = 'data'
    download_required = not os.path.exists(os.path.join(data_dir, 'MNIST'))
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=download_required, transform=transform),
        batch_size=64, shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
    
    # Model, optimizer and criterion
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 1
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            pbar.set_description(
                f'Loss={loss.item():.4f} | Acc={100 * correct / total:.2f}%'
            )
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(models_dir, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')
    return save_path

if __name__ == '__main__':
    train()
