import torch
from torchvision import datasets, transforms
from model import Net
import sys
import os
import glob
import tqdm
from train_model import is_ci_environment

def get_latest_model():
    """Get the latest model from the models directory"""
    models = glob.glob('models/*.pth')
    if not models:
        raise FileNotFoundError("No model files found in models directory")
    return max(models, key=os.path.getctime)

def test(model_path=None):
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If no model specified, use the latest one
    if model_path is None:
        model_path = get_latest_model()
    elif not os.path.dirname(model_path):
        model_path = os.path.join('models', model_path)
    
    print(f"Using model: {model_path}")
    
    # Check if data exists
    data_dir = 'data'
    download_required = not os.path.exists(os.path.join(data_dir, 'MNIST'))
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load model and normalization values
    checkpoint = torch.load(model_path, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mean = checkpoint.get('mean', 0.5)  # fallback to 0.5 if not found
    std = checkpoint.get('std', 0.5)    # fallback to 0.5 if not found
    
    # After loading checkpoint
    print(f"Loaded mean: {mean}, std: {std}")
    print(f"Model state keys: {checkpoint.keys()}")
    
    # Load test data with same normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=download_required, transform=test_transform),
        batch_size=1000, shuffle=True)
    
    model.eval()
    
    # Test model parameters
    param_count = model.count_parameters()
    # print(f"Model architecture:\n{model}")
    print(f"Parameter count: {param_count}")
    assert param_count < 20000, "Model has too many parameters"
    
    # Check for either GAP or FC layer
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected layer"
    print('✓ Architecture test passed (Using ' + ('GAP' if has_gap else 'FC layer') + ' for classification)')
    return True

def test_performance(model, device=None, test_loader=None, is_ci=None):
    """Test the model performance"""
    if is_ci is None:
        is_ci = is_ci_environment()
    
    # Device selection based on environment
    if device is None:
        if is_ci:
            device = torch.device('cpu')
            print("CI Environment: Using CPU")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                print(f"Local Environment: Using GPU - {torch.cuda.get_device_name(0)}")
            else:
                print("Local Environment: Using CPU")
    
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        iterator = test_loader
        if not is_ci:
            iterator = tqdm.tqdm(test_loader, 
                               desc=f'Test Epoch 1/1',
                               leave=True,
                               ncols=100)
        
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if not is_ci:
                current_acc = 100. * correct / total
                iterator.set_description(f'Test Epoch 1/1 | Accuracy={current_acc:.2f}%')

    final_accuracy = 100. * correct / total
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'Raw values - Correct: {correct}, Total: {total}')
    
    required_accuracy = 99.4
    if final_accuracy < required_accuracy:
        raise AssertionError(
            f'Model accuracy {final_accuracy:.2f}% is below required {required_accuracy:.1f}% '
            f'(Correct: {correct}/{total})'
        )
    
    return final_accuracy

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python test_model.py [model_filename]")
        print("Example: python test_model.py model_20230615_143022.pth")
        print("If no model specified, the latest model will be used")
        sys.exit(1)
    
    model_file = sys.argv[1] if len(sys.argv) == 2 else None
    test(model_file) 
