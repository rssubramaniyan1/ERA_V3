import optuna
from optuna.trial import Trial
import torch.nn as nn
import torch
from torchvision import datasets
from augmentation import DataAugmentation
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from model_v5 import Net, NormalizationTypes
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import random
import os
import torch.nn.functional as F

TOTAL_EPOCHS = 2
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


def objective(trial: Trial):
    """Optuna objective function for hyperparameter optimization"""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters to optimize
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.01, 0.1),
        'dropout_rate1': trial.suggest_float('dropout_rate1', 0.01, 0.05),
        'dropout_rate2': trial.suggest_float('dropout_rate2', 0.01, 0.05),
        'dropout_rate3': trial.suggest_float('dropout_rate3', 0.02, 0.06),
        'pct_start': trial.suggest_float('pct_start', 0.1, 0.3),
        'div_factor': trial.suggest_int('div_factor', 5, 15),
        'final_div_factor': trial.suggest_int('final_div_factor', 30, 100)
    }
    
    # Data setup
    dataset = datasets.MNIST('data', train=True, download=True, transform=DataAugmentation().get_transforms(train=True))
    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    train_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True,
                      transform=DataAugmentation().get_transforms(train=False)),
        batch_size=1000,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model setup
    model = Net(norm_type=NormalizationTypes.BATCH)
    model.to(device)
    
    # Training setup
    criterion = LabelSmoothingLoss(smoothing=params['label_smoothing'])
    optimizer = optim.SGD(
        model.parameters(),
        lr=params['learning_rate'],
        momentum=0.9,
        weight_decay=params['weight_decay'],
        nesterov=True
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params['learning_rate'],
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=params['pct_start'],
        anneal_strategy='cos',
        div_factor=params['div_factor'],
        final_div_factor=params['final_div_factor']
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(TOTAL_EPOCHS):
        # Train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        
        # Update best accuracy
        best_acc = max(best_acc, accuracy)
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_acc

# Default parameters if optimization hasn't been run
default_params = {
    'learning_rate': 0.02,
    'weight_decay': 0.0001,
    'batch_size': 128,
    'label_smoothing': 0.05,
    'dropout_rate1': 0.05,
    'dropout_rate2': 0.05,
    'dropout_rate3': 0.06,
    'pct_start': 0.2,
    'div_factor': 10,
    'final_div_factor': 50
}

# This will be updated if optimization is run
best_params = default_params.copy()

def run_optuna_study(n_trials=50):
    """Run Optuna hyperparameter optimization"""
    global best_params  # Allow updating the best_params
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(
        n_startup_trials=2,  # Reduce from 5
        n_warmup_steps=2,    # Reduce from 5
        interval_steps=1     # Reduce from 3
    ))
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params  # Update best_params with optimization results
    return best_params

# Usage in main:
if __name__ == '__main__':
    best_params = run_optuna_study(n_trials=50)
    
    # Use best parameters for final training
    