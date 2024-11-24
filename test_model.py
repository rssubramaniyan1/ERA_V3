import torch
from torchvision import datasets, transforms
from model import Net
import sys
import os
import glob

def get_latest_model():
    """Get the latest model from the models directory"""
    models = glob.glob('models/*.pth')
    if not models:
        raise FileNotFoundError("No model files found in models directory")
    return max(models, key=os.path.getctime)

def test(model_path=None):
    # Force CPU
    device = torch.device('cpu')
    
    # If no model specified, use the latest one
    if model_path is None:
        model_path = get_latest_model()
    elif not os.path.dirname(model_path):
        model_path = os.path.join('models', model_path)
    
    print(f"Using model: {model_path}")
    
    # Check if data exists
    data_dir = 'data'
    download_required = not os.path.exists(os.path.join(data_dir, 'MNIST'))
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=download_required, transform=transform),
        batch_size=1000, shuffle=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load model
    model = Net().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    model.eval()
    
    # Test model parameters
    param_count = model.count_parameters()
    print(f"Parameter count: {param_count}")
    assert param_count < 25000, "Model has too many parameters"
    
    # Test input shape
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        output = model(dummy_input)
        assert output.shape[1] == 10, "Model output should have 10 classes"
    except Exception as e:
        print(f"Input shape test failed: {str(e)}")
        sys.exit(1)
    
    # Test accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    assert accuracy > 95, "Model accuracy is below 95%"
    
    return True

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python test_model.py [model_filename]")
        print("Example: python test_model.py model_20230615_143022.pth")
        print("If no model specified, the latest model will be used")
        sys.exit(1)
    
    model_file = sys.argv[1] if len(sys.argv) == 2 else None
    test(model_file) 