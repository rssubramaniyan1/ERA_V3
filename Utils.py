import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(model, learning_rate=0.01, weight_decay=0.0001):
    """Setup loss, optimizer, and scheduler"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )
    scheduler = None  # Will define this in the training loop with OneCycleLR
    return criterion, optimizer, scheduler

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)  # Number of classes

        # Convert target to long if needed
        target = target.long()

        # Create a tensor of shape (batch_size, num_classes)
        one_hot = torch.zeros_like(pred)

        # Ensure target has correct dimensions
        if target.dim() > 1:
            target = target.squeeze()

        # Scatter the 1s
        one_hot.scatter_(1, target.unsqueeze(1), 1)

        # Apply label smoothing
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute the loss
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_one_hot * log_probs).sum(dim=1).mean()

        return loss


def save_checkpoint(model, optimizer, epoch, best_acc, scheduler=None, swa_model=None, filename='checkpoint.pth'):
    """Save model checkpoint with optional SWA and scheduler states"""
    os.makedirs('checkpoints', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add SWA state if provided
    if swa_model is not None:
        checkpoint['swa_state'] = {
            'model': swa_model.state_dict(),
            'n_averaged': swa_model.n_averaged
        }
    
    path = os.path.join('checkpoints', filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, scheduler=None, swa_model=None, filename='checkpoint.pth'):
    """Load model checkpoint with optional SWA and scheduler states"""
    path = os.path.join('checkpoints', filename)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided and available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load SWA state if provided and available
        if swa_model is not None and 'swa_state' in checkpoint:
            swa_model.load_state_dict(checkpoint['swa_state']['model'])
            swa_model.n_averaged = checkpoint['swa_state']['n_averaged']
        
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        print(f"Loaded checkpoint from epoch {start_epoch} with accuracy {best_acc:.2f}%")
        return start_epoch, best_acc
    return 0, 0