import torch
from test_model import test_performance
from torchsummary import summary
from model_v5 import Net, NormalizationTypes
from torchvision import datasets
from augmentation import DataAugmentation
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from hyperparametertunig import best_params
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import random
import os
import torch.nn as nn
import test_model
import torch.nn.functional as F
print("Using test_model from:", test_model.__file__)
TOTAL_EPOCHS = 15

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        loss = -torch.sum(smooth_one_hot * F.log_softmax(pred, dim=1), dim=1).mean()
        return loss

def train(model, train_loader, optimizer, scheduler, criterion, device):
    """Training function that can be called from main.py"""
    set_seed(42)
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = LabelSmoothingLoss(smoothing=0.05)
    # criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    iterator = tqdm(train_loader, 
                        desc='Training',
                        leave=True,
                        ncols=100)

    for data, target in iterator:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if scheduler is not None:
        #     scheduler.step()

        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()

        current_acc = 100. * correct / total
        iterator.set_description(f'Training | Acc: {current_acc:.2f}% | Loss: {running_loss/total:.4f}')

    train_acc = 100. * correct / total
    print(f'\nTraining Summary | Accuracy: {train_acc:.2f}% | Loss: {running_loss/len(train_loader):.4f}')
    
    # if torch.cuda.is_available():
    #     print_gpu_memory()
    
    return train_acc

# Utility functions
def set_seed(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# def print_gpu_memory():
#     if torch.cuda.is_available():
#         print(f'Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
#         print(f'Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB')

def is_ci_environment():
    """Check if we're running in a CI environment"""
    return os.environ.get('CI') is not None or os.environ.get('GITHUB_ACTIONS') is not None

def get_optimizer(model, learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'], steps_per_epoch=469, batch_size=128):
    """Setup loss, optimizer, and scheduler with weight decay"""
    criterion = LabelSmoothingLoss(smoothing=0.05)  # Reduced smoothing
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    # More aggressive learning rate schedule
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.01, patience=1, verbose=True,
                                                      min_lr=1e-8)
    return criterion, optimizer, scheduler

def train_standalone(epochs=TOTAL_EPOCHS, test_loader=None):
    """Standalone training function that handles all setup and training"""
    # Set seeds for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # Model setup
    BATCH_SIZE = best_params['batch_size']
    model = Net(norm_type=NormalizationTypes.BATCH).to(device)
    summary(model, (1, 28, 28))
    
    # Data setup
    augmenter = DataAugmentation()
    train_transform = augmenter.get_transforms(train=True)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transform),
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
     # Setup optimizer and criterion
   
    # Keep track of best accuracy
    best_acc = 0
    criterion, optimizer, scheduler = get_optimizer(model)
    # Training loop
    for epoch in range(epochs):
        print(f"\nEPOCH: {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_acc = train(model=model, 
                         train_loader=train_loader, 
                         optimizer=optimizer, 
                         scheduler=scheduler,
                         criterion=criterion, 
                         device=device)
        
        # Test if test_loader is provided
        if test_loader is not None:
            print("Calling test_performance with epoch:", epoch)
            test_acc = test_performance(model=model, 
                                      test_loader=test_loader, 
                                      device=device,
                                      epoch=epoch)
            
             # Monitor improvement
            if test_acc > best_acc:
                best_acc = test_acc
            scheduler.step(test_acc)
            plateau_counter = 0
            if test_acc > best_acc:
                plateau_counter = 0
            else:
                plateau_counter += 1

            
            print(f"Epoch Summary | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
            print(f"Best Acc: {best_acc:.2f}% | Epochs without improvement: {plateau_counter}")
        else:
            print(f"Epoch Summary | Train Acc: {train_acc:.2f}%")
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
    
    return model

def main():
    # Data setup for test set
    augmenter = DataAugmentation()
    test_transform = augmenter.get_transforms(train=False)
    
    # Test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=test_transform),
        batch_size=1000, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Call train_standalone with test_loader
    model = train_standalone(epochs=TOTAL_EPOCHS, test_loader=test_loader)

if __name__ == '__main__':
    main() 