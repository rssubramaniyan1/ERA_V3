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
from model import Net

def check_data_exists():
    data_path = 'data/MNIST/raw'
    required_files = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]
    
    if not os.path.exists(data_path):
        print("Data directory does not exist. Will download data.")
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            print(f"Missing {file}. Will download data.")
            return False
    
    print("MNIST data already exists. Skipping download.")
    return True

def train():
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST specific normalization
    ])
    
    # Load dataset with augmentation
    download_required = not check_data_exists()
    train_dataset = datasets.MNIST('data', train=True, download=download_required, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model, criterion, optimizer
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()  # Changed from NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    num_epochs = 1
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
        
        # Epoch-end statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {epoch_accuracy:.2f}%')
    
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
