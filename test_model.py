import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import glob
import os
from model import Net

def test_model():
    # Set reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Net().to(device)
    
    # Get current working directory and find model files
    current_dir = os.getcwd()
    model_dir = os.path.join(current_dir, 'models')
    model_files = glob.glob(os.path.join(model_dir, 'mnist_model_*.pth'))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    latest_model = max(model_files)
    print(f"Found model files: {model_files}")
    print(f"Loading latest model: {latest_model}")
    
    try:
        model.load_state_dict(torch.load(latest_model, weights_only=True, map_location=device))
        print(f"Successfully loaded model from: {latest_model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Current working directory: {current_dir}")
        print(f"Model directory contents: {os.listdir(model_dir)}")
        raise
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has too many parameters: {total_params}"
    
    # Setup data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    # Test loop
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Overall accuracy
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Class-wise accuracy
            for t, p in zip(target, pred):
                class_correct[t.item()] += (p == t).item()
                class_total[t.item()] += 1
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix(accuracy=f"{accuracy:.2f}%")
    
    # Print final results
    final_accuracy = 100. * correct / total
    print(f"\nOverall Test accuracy: {final_accuracy:.2f}%")
    
    # Print class-wise accuracy
    print("\nClass-wise accuracy:")
    for i in range(10):
        class_acc = 100. * class_correct[i] / class_total[i]
        print(f"Class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    assert final_accuracy > 95, f"Model accuracy is {final_accuracy:.2f}%, should be > 95%"

if __name__ == "__main__":
    test_model()

