# Importing MNIST dataset from torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# Importing torch.nn and torch.optim
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from datetime import datetime
import os
from model import Net  # Import Net from model.py

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
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        # Update progress bar with both loss and accuracy
        if batch_idx % 10 == 0:  # Update every 10 batches
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
            
        # Print detailed stats every 100 batches
        if batch_idx % 100 == 0:
            print(f'\nBatch {batch_idx}/{len(train_loader)}:')
            print(f'Loss: {avg_loss:.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
            print(f'Correct/Total: {correct}/{total}')
    
    # Final training stats
    final_accuracy = 100. * correct / total
    final_loss = running_loss / len(train_loader)
    print(f'\nTraining completed:')
    print(f'Final Loss: {final_loss:.4f}')
    print(f'Final Accuracy: {final_accuracy:.2f}%')
    
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
