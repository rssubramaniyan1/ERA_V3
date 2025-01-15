import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import os
import numpy as np
from torch.amp import autocast, GradScaler
import warnings
from torch.optim.swa_utils import AveragedModel, SWALR

# Suppress all PIL warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Import our custom modules
from dataset import get_data_loaders
from Utils import save_checkpoint, load_checkpoint, set_seed, LabelSmoothingLoss, get_optimizer

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

def train_epoch(model, train_loader, criterion, optimizer, epoch, scheduler, scaler, swa_model=None, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Scale loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights if we've accumulated enough gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased from 1.0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
            if swa_model is not None and epoch >= 75:
                swa_model.update_parameters(model)
        
        # Update metrics
        running_loss += loss.item() * accumulation_steps  # Scale loss back
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
        # Free up memory periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Free up memory
            del outputs, loss
            torch.cuda.empty_cache()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    
    print(f"\nValidation Stats:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct/Total: {correct}/{total}")
    
    return avg_loss, accuracy


def main():
    global best_acc, scaler
    
    print(f"\nDevice: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    print("\nInitializing ResNet50...")
    model = models.resnet50(weights=None)
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    model = model.to(device, memory_format=torch.channels_last)
    
    swa_model = AveragedModel(model)
    
    scaler = GradScaler()
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    batch_size = min(256, int(gpu_mem * 32))
    num_workers = min(os.cpu_count(), 16)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Use our own LabelSmoothingLoss from Utils.py
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003 * batch_size/256,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    # Setup SWA scheduler
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=1e-2,
        anneal_epochs=5,
        anneal_strategy='cos'
    )
    
    # Reduce accumulation steps to speed up training
    accumulation_steps = 4
    effective_batch_size = batch_size * accumulation_steps
    print(f"\nEffective batch size with accumulation: {effective_batch_size}")
    
    print("\nStarting training...")
    best_acc = 0.0
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0
    early_stop_accuracy = 70.0  # Early stopping threshold
    
    for epoch in range(100):
        # Training
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler,
            swa_model=swa_model if epoch >= 75 else None,
            accumulation_steps=accumulation_steps
        )
        
        # Validation
        if epoch >= 75:  # Use SWA model for validation after epoch 75
            val_loss, val_acc = validate(swa_model, val_loader, criterion)
        else:
            val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoints
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model=model if epoch < 75 else swa_model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                scheduler=scheduler,
                filename='best_model.pth'
            )
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            save_checkpoint(
                model=model if epoch < 75 else swa_model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                scheduler=scheduler,
                filename=f'checkpoint_epoch_{epoch}.pth'
            )
        
        # Early stopping check
        if val_acc >= early_stop_accuracy:
            print(f"\nReached target accuracy of {early_stop_accuracy}%!")
            print(f"Early stopping at epoch {epoch+1}")
            # Save final model
            save_checkpoint(
                model=model if epoch < 75 else swa_model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                scheduler=scheduler,
                filename='final_model.pth'
            )
            break
        
        # Early stopping due to no improvement
        if patience_counter >= patience:
            print(f"\nNo improvement for {patience} epochs. Early stopping!")
            break

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()