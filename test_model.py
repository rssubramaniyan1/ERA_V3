import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from train_model import Net

def test_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    
    # Load the trained model
    model_path = '/media/ravis/D/ERA_V3/Assignment5/models/mnist_model_20241124_104253.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has too many parameters: {total_params}"
    
    # Setup data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train for one epoch
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"\nTrain accuracy after one epoch: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"

if __name__ == "__main__":
    test_model()

