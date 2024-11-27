import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
from torch.utils.data import Subset
import numpy as np
from model import Net
from datetime import datetime
import os
import tqdm
import random
from augmentation import DataAugmentation

def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(model, learning_rate=0.01, weight_decay=0.0001):
    """Setup loss, optimizer, and scheduler"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None  # Will define this in the training loop with OneCycleLR
    return criterion, optimizer, scheduler


def train():
    # Set seeds for reproducibility
    set_seed(42)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check if data exists
    data_dir = 'data'
    download_required = not os.path.exists(os.path.join(data_dir, 'MNIST'))
    
    # Initialize augmentation
    augmenter = DataAugmentation()
    train_transform = augmenter.get_transforms(train=True)
  
    
    # Load datasets
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=download_required, transform=train_transform
    )
  
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
    )
    
    
    
    # Model setup
    model = Net().to(device)
    
    # Get optimizer and initial scheduler
    criterion, optimizer, _ = get_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0001
    )
    
    # Define number of epochs
    train_epochs = 20
    
    # Create OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=train_epochs,
        pct_start=0.3,
        div_factor=10,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    # Training Phase
    print("Starting Training Phase...")
    best_train_acc = 0
    for epoch in range(train_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        # Use tqdm with adjusted settings for cleaner output
        pbar = tqdm.tqdm(train_loader, 
                         desc=f'Train Epoch {epoch + 1}/{train_epochs}',
                         disable=False,  # Enable the progress bar
                         leave=True,     # Leave the progress bar after completion
                         ncols=80,       # Fixed width for better formatting
                         ascii=True)     # Use ASCII characters for compatibility

        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()

            # Update progress bar description
            pbar.set_postfix({
                'Loss': f'{running_loss / (total / target.size(0)):.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Print epoch summary
        train_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}/{train_epochs} Summary | '
              f'Loss: {running_loss / len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            # Save best training model
            save_path = os.path.join(models_dir, f'best_train_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'mean': augmenter.mean,
                'std': augmenter.std
            }, save_path)
    
    print(f"\nTraining completed. Best training accuracy: {best_train_acc:.2f}%")
    return save_path

if __name__ == '__main__':
    train()
