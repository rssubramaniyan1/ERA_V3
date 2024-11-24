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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # Set seeds for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))  # Set seed for dataloader workers
    
    # Model, optimizer and criterion
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
    criterion = nn.CrossEntropyLoss()
    
    def calculate_accuracy(loader, model):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total
    
    epochs = 1
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        pbar = tqdm.tqdm(train_loader)

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
                desc=f'Loss={loss.item():.4f} | Train Accuracy={100 * correct / total:.2f}% | Epoch={epoch + 1}'
            )
    
    # Save model with timestamp in models directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(models_dir, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == '__main__':
    train()
